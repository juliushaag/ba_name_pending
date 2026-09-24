import time
from abc import ABC, abstractmethod
from functools import partial
from typing import Optional

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import barl_jax.executors as executors
from barl_jax.action_chunking import normalize_action_chunk
from barl_jax.components import (
  TRAINABLE,
  Buffer,
  Component,
  _Static,
  interned_static,
  sanitize_action,
)
from barl_jax.env import EnvStats
from barl_jax.metrics import Recorder
from barl_jax.optimization import OptimizerMixin


@partial(jax.jit, static_argnums=(0,))
def _jitted_infer(graphdef, state, observation, key):
  """The compiled body behind `Policy.inference_fn`. Module-level so its jit
  cache survives across calls: repeated `nnx.split` of the same policy yields an
  equal-and-equally-hashing GraphDef, so a rollout or an eval wave rebinding its
  weights hits the cache instead of recompiling."""
  action, extras = nnx.merge(graphdef.value, state).inference_step(observation, key=key)
  return action, extras


@partial(jax.jit, static_argnums=(0, 1))
def _jitted_infer_exec(graphdef, executor_config, state, observation, key):
  """`_jitted_infer`'s counterpart for every non-direct executor. Kept
  separate rather than folded into `_jitted_infer` so the plain direct path
  (every caller that never opts into an execution strategy) keeps its own
  jit cache, unperturbed by `executor_config` becoming part of the trace.

  Only `executor_config` -- a frozen, all-primitive-field dataclass, hashable
  for free -- crosses the jit boundary as a static arg; the executor instance
  itself never needs to (see `executors.base.executor_from_config`)."""
  policy = nnx.merge(graphdef.value, state)
  executor = executors.executor_from_config(executor_config)
  infer_step = lambda obs, key: policy.inference_step(obs, key=key)
  score_fn = policy.score if executor.needs_score else None
  return executor.select(infer_step, score_fn, observation, key)


class Policy(OptimizerMixin, nnx.Module, ABC):

  _classes = dict()

  # The component slots this Policy builds, by attribute name -- the contract a
  # model variant's `components:` must match. An omitted slot already fails at
  # build_component(); naming an *extra* one would resolve silently into the
  # checkpoint metadata with nothing behind it, so validation checks both
  # directions against this.
  SLOTS: tuple[str, ...] = ()
  SLOT_OBJECTIVES: dict[str, str] = {}

  # ------------------------------------------------------------------
  # Registry
  # ------------------------------------------------------------------

  @staticmethod
  def make(name: str, env: gym.Env, *args, **kwargs) -> "Policy":
    from barl_jax.config.model import ModelSpec
    config = kwargs.pop("config", None)
    spec = kwargs.pop("spec", None)
    profile = kwargs.pop("profile", None)
    if spec is None:
      # `config` is a RunConfig when present; otherwise resolve bare defaults.
      spec = ModelSpec.resolve(name, config, profile)
    cls = Policy.class_for(spec.policy_class)
    return cls(env, *args, model_name=spec.name, profile=spec.profile, spec=spec, **kwargs)

  @staticmethod
  def class_for(name):
    from barl_jax.registration import available_names, import_type, load_builtins, resolve_named
    load_builtins("policies")
    if ":" in name:
      cls = import_type(name)
    else:
      cls = resolve_named("policies", name, Policy._classes, Policy)
      if cls is None:
        raise KeyError(f"{name!r} is not a registered Policy or plugin "
                       f"({', '.join(available_names('policies', Policy._classes))})")
    if not issubclass(cls, Policy):
      raise TypeError(f"{name!r} must implement Policy")
    return cls

  @staticmethod
  def available() -> list[str]:
    """Every model name a `model:` string can resolve to -- not the raw class
    registry, which is an implementation detail once one class can back several
    models."""
    from barl_jax.config.schemas import MODEL_ALIASES, MODEL_SCHEMAS
    return sorted([*MODEL_SCHEMAS, *MODEL_ALIASES])

  # ------------------------------------------------------------------
  # Construction
  # ------------------------------------------------------------------

  def __init__(self, env: gym.Env, profile: str = None, env_stats: EnvStats = None,
               rngs: nnx.Rngs = None, seed: int = 0, model_name: str = None, spec=None, **kwargs):
    from barl_jax.config.model import ModelSpec

    # The `model:` string this Policy was built from -- distinct from
    # `type(self).__name__` once one class backs several models (RLPolicy /
    # MaxEntRLPolicy). Plain str, so it is static in nnx: it lands in the
    # GraphDef (SAC and AMDP get distinct jit cache entries) and never enters
    # the state tree `multiseed.stack()` compares. Defaults to the class name
    # so a policy built directly (a test, a notebook) still resolves.
    self.model_name = model_name or type(self).__name__
    self._profile  = profile
    # Held behind _Static for hashability (see _Static): these are dicts/lists,
    # and the GraphDef they end up in is a static jit argument.
    self._raw_kwargs_s = _Static(dict(kwargs))
    # Captured so a checkpoint reconstructs under the same config it was built
    # from -- see checkpoints.py.
    self._spec = spec if spec is not None else ModelSpec.resolve(self.model_name, None, profile)

    # Every module that needs randomness draws from this; nnx threads and
    # advances the key state across the jit boundary, so no call signature has
    # to grow a `key` argument the way a bare-jax port would need.
    self.rngs = rngs if rngs is not None else nnx.Rngs(seed)

    stats = env_stats if env_stats is not None else EnvStats.from_env(env)
    self._env_stats = _Static(stats)

    # Build order of the slots build_component() constructs -- names only. The
    # components themselves are *not* stored here: they already live in the
    # module tree under their own attributes (self.critic, self.actor, ...),
    # and a second hard reference would survive nnx.merge() unchanged, so the
    # lifecycle hooks would mutate the pre-trace objects instead of the merged
    # ones (TraceContextError). `components` re-derives the mapping from live
    # attributes every time.
    self._optimizer_slots_cache = _Static({"value": None})

    action_space      = env.action_space
    observation_space = env.observation_space

    if isinstance(action_space, gym.spaces.Discrete):
      self.action_dim      = int(action_space.n)
      self.action_discrete = True
    elif isinstance(action_space, gym.spaces.Box):
      self.action_dim      = int(np.prod(action_space.shape))
      self.action_discrete = False

      low  = jnp.asarray(action_space.low,  dtype=jnp.float32)
      high = jnp.asarray(action_space.high, dtype=jnp.float32)
      self.act_low   = Buffer(low)
      self.act_high  = Buffer(high)
      self.act_scale = Buffer((high - low) / 2)
      self.act_bias  = Buffer((high + low) / 2)
    else:
      raise NotImplementedError(f"{action_space} type not supported ... yet")

    self.observation_shape = tuple(observation_space.shape)

    self.obs_std  = Buffer(jnp.ones(observation_space.shape[-1]))
    self.obs_mean = Buffer(jnp.zeros(observation_space.shape[-1]))
    self.obs_count = Buffer(jnp.zeros((), jnp.float32))

    # Gradient steps taken, for algorithms whose update cadence is `step % k`.
    self.train_steps = Buffer(jnp.zeros((), jnp.int32))

    # The Policy's own params come from the typed model spec, with explicit
    # constructor kwargs (e.g. trainer's `chunk_size`) winning. Subclasses
    # read them off `self` (self.policy_delay, ...) after calling
    # `_absorb_params`.
    self._absorb_params(self.resolve_params())
    self._env_stats = _Static(stats.derive(
      chunk_size=self.chunk_size,
      chunked_act_dim=self.chunk_size * self.action_dim))

  @property
  def env_stats(self) -> EnvStats:
    return self._env_stats.value

  # A Policy sits at the root of the tree, so its own diagnostics have no path
  # to take a name from -- these are the two namespaces it owns outright.
  @property
  def metrics(self) -> Recorder:
    return Recorder(self, self.algo_name())

  @property
  def batch_metrics(self) -> Recorder:
    return Recorder(self, "batch")

  @property
  def _raw_kwargs(self) -> dict:
    return self._raw_kwargs_s.value

  def _absorb_params(self, params: dict) -> None:
    """Copy the typed `params` mapping onto plain attributes.

    Only the two keys every Policy owns live here; algorithm-specific keys
    (`policy_delay`, `entropy_scale`, ...) are absorbed by the subclass that
    owns them, so a typo'd param fails at the subclass rather than silently
    landing on the base.
    """
    self.chunk_size = int(params.get("chunk_size", 1))
    self.normalize_obs = bool(params.get("normalize_obs", True))

  def resolve_component(self, slot: str) -> tuple[type, dict]:
    """Resolves a named slot's class + canonical mapping from the spec.

    Policy constructor kwargs are deliberately not broadcast into every slot:
    slot overrides use the normal `components.<slot>.<key>` path, so a policy
    parameter cannot accidentally retune every component with the same name."""
    return self._spec.component(slot)

  def resolve_params(self) -> dict:
    """This Policy's own top-level params from the typed spec, with explicit
    constructor kwargs winning."""
    return {**self._spec.params(), **self._raw_kwargs}

  @classmethod
  def slot_objective(cls, slot, actor_cls=None):
    return cls.SLOT_OBJECTIVES.get(slot)

  def build_component(self, slot: str, gate_cls: type, *args,
                      env_stats: EnvStats = None, config_overrides: dict = None,
                      **explicit_kwargs):
    """Resolves + constructs a named slot (actor/critic/squash/...), gated to
    `gate_cls`: the resolved class must be `gate_cls` or a subclass. `env_stats`
    overrides the run's own for a slot another component wraps and feeds a
    transformed input -- pass an `EnvStats.derive()` of the run's, never an
    unrelated one.

    `explicit_kwargs` are for values computed at run time that never live in
    config (an entropy dual's `target`) -- everything config supplies reaches
    the component as a materialized typed instance, never through this.
    `config_overrides` pins a config-shaped tunable regardless of the stored
    value."""
    cls, mapping = self.resolve_component(slot)
    if config_overrides:
      mapping = {**mapping, **config_overrides}
    if not issubclass(cls, gate_cls):
      raise TypeError(f"{self.model_name}.{slot}: {cls.__name__} must implement {gate_cls.__name__}")
    config = _materialize_component_config(cls, mapping, owner=f"{self.model_name}.{slot}")
    objective = self.slot_objective(
      slot, self.resolve_component("actor")[0] if "actor" in self.SLOTS else None)
    if objective is not None and cls.learning_objective_for(config) != objective:
      raise TypeError(f"{self.model_name}.{slot}: {cls.__name__} does not support {objective!r}")
    stats = env_stats if env_stats is not None else self.env_stats
    explicit_kwargs.setdefault("rngs", self.rngs)
    component = cls(stats, config, *args, **explicit_kwargs)
    component.slot = slot
    return component

  def derived_config(self) -> dict:
    """Every slot's construction-derived values, keyed `<slot>.<name>` -- the
    part of what a run actually built that config/ alone does not state. Merged
    into the run's wandb config next to the resolved kwargs."""
    return {f"{slot}.{key}": value
            for slot, component in self.components.items()
            for key, value in component.derived().items()}

  # ------------------------------------------------------------------
  # Metrics
  # ------------------------------------------------------------------

  def algo_name(self) -> str:
    """This algorithm's logging namespace -- `SAC` -> `sac`, `AMDP` -> `amdp`.
    From `model_name`, not `type(self).__name__`: several models can share one
    Policy class (RLPolicy / MaxEntRLPolicy), and each keeps its own namespace."""
    name = self.model_name.lower()
    return name[:-len("policy")] if name.endswith("policy") and len(name) > len("policy") else name

  def record_batch(self, batch: dict):
    """The `batch/*` family -- what every algorithm trains against, recorded
    once here rather than restated by each train_step."""
    self.batch_metrics.dist("reward", batch["chunk_return"])
    self.batch_metrics.dist("action", batch["act"])

  # ------------------------------------------------------------------
  # Component lifecycle
  # ------------------------------------------------------------------

  @property
  def components(self) -> dict[str, "Component"]:
    """slot -> component, walked out of the module graph rather than tracked in
    a parallel registry. Two things follow: it is always the merged/traced
    objects (a stored reference would survive nnx.merge() and the lifecycle
    hooks would mutate the pre-trace ones), and a slot's name is just its
    attribute name, so nothing has to be kept in sync.

    Direct children only -- those are the Policy's slots. A Squash nested
    inside an actor is that actor's business, not a slot."""
    return {str(path[0]): node
            for path, node in nnx.iter_graph(self)
            if len(path) == 1 and isinstance(node, Component)}

  def before_step(self):
    """Per-step bookkeeping each component owns (a critic's Polyak update),
    fanned out so no train_step has to drive it by hand."""
    for component in self.components.values():
      component.before_step()

  def after_step(self):
    self.train_steps[...] += 1
    for component in self.components.values():
      component.after_step()

  # ------------------------------------------------------------------
  # Checkpointing
  # ------------------------------------------------------------------

  # ------------------------------------------------------------------
  # Observation normalisation
  # ------------------------------------------------------------------

  def update_observation_stats(self, obs):
    if not self.normalize_obs:
      return
    count = jnp.asarray(obs.shape[0], jnp.float32)
    total = self.obs_count[...] + count
    mean = jnp.mean(obs, axis=0)
    variance = jnp.var(obs, axis=0)
    delta = mean - self.obs_mean[...]
    combined = (self.obs_std[...] ** 2 * self.obs_count[...] + variance * count
                + delta ** 2 * self.obs_count[...] * count / total)
    self.obs_mean[...] += delta * count / total
    self.obs_std[...] = jnp.sqrt(jnp.maximum(combined / total, 0.0))
    self.obs_count[...] = total

  def normalize_observations(self, obs):
    if not self.normalize_obs:
      return obs
    return (obs - self.obs_mean[...]) / jnp.maximum(self.obs_std[...], 1e-6)

  @property
  def discount(self):
    return 1.0

  # ------------------------------------------------------------------
  # Action scaling
  # ------------------------------------------------------------------

  # Every continuous actor trains against a canonical per-dimension [-1, 1]
  # output, independent of the env's real bounds; scale_action is the one-place
  # affine remap, used only at env.step() call sites.

  def scale_action(self, raw_action):
    """Canonical [-1, 1]-per-dimension actor output -> this env's actual action bounds.

    Sanitizes first, so the affine map cannot carry a NaN or an out-of-box value
    into the env: outside [-1, 1] it would land outside the env's own bounds too,
    which gymnasium does not check for MuJoCo."""
    if self.action_discrete:
      return raw_action
    return sanitize_action(raw_action) * self.act_scale[...] + self.act_bias[...]

  def action_bounds_numpy(self) -> tuple:
    """Host-side (scale, bias), for the rollout/eval loops -- scaling on the
    device would cost a second round-trip per step for an affine map on an
    array that is already back on the host."""
    if self.action_discrete:
      return None, None
    return np.asarray(self.act_scale[...]), np.asarray(self.act_bias[...])

  def action_chunk(self, action):
    """Canonical runtime action shape, always `(batch, chunk_size, act_dim)`."""
    return normalize_action_chunk(action, chunk_size=self.chunk_size, act_dim=self.action_dim, xp=jnp)

  # ------------------------------------------------------------------
  # Misc
  # ------------------------------------------------------------------

  def num_params(self) -> int:
    # TRAINABLE, not nnx.Param: a Polyak target is a full second copy of the
    # critic and counting it would inflate this by ~40%. It is the analogue of
    # torch's `if p.requires_grad`, which excluded the frozen EMA copy.
    return sum(int(np.prod(x.shape)) for x in jax.tree.leaves(nnx.state(self, TRAINABLE)))

  # ------------------------------------------------------------------
  # Jitted inference
  # ------------------------------------------------------------------

  # inference_step runs once per env step in both rollout and eval, so this is
  # the hot path. Called directly it dispatches op-by-op in eager JAX (~17x the
  # jitted cost); called through nnx.jit it pays nnx's per-call Python graph
  # traversal, which at batch 1 dominates everything else. So it goes through a
  # plain jax.jit over a split graphdef/state instead, with the RNG passed in
  # explicitly rather than mutated on the module -- that keeps the traced
  # function pure and lets the parameter state be captured once per rollout
  # instead of round-tripped every step.

  def inference_fn(self, executor: "executors.ActionExecutor" = None):
    """A pure, jitted `(observation, key) -> (action, extras)` pinned to this
    policy's *current* weights. Rebuild it after training (the rollout loop
    does, once per iteration); it deliberately does not see later updates.
    One compile per batch shape (and, when `executor` is a multi-candidate
    strategy, per its config -- every field is a static jit arg, see
    `barl_jax.executors.base.ExecutorConfig`).

    The key is an argument rather than drawn from the module's own `nnx.Rngs`:
    that keeps the traced function pure, lets the parameter state be captured
    once per rollout instead of round-tripped every step, and leaves the caller
    holding the one piece of state that actually has to advance.

    `executor=None` (or a `DirectExecutor`, its default) is the original code
    path -- unchanged, and sharing `_jitted_infer`'s compile cache with every
    other caller that has not opted into a non-default execution strategy.

    Must be called from eager (non-jitted) context -- it temporarily mutates
    the module to set `use_running_average=True` on BatchNorm/BatchRenorm layers
    so the compiled inference uses running statistics."""
    # Acting uses running statistics, not the batch's -- a rollout step is a
    # batch of `num_envs`, and a BatchNorm/BatchRenorm critic normalized by that
    # would price actions differently than it was trained to. `use_running_average`
    # is captured in the GraphDef at split time, so we set eval mode on a clone
    # to avoid mutating the live module.
    clone = nnx.clone(self)
    clone.eval()
    graphdef, state = nnx.split(clone)

    if executor is None or executor.is_direct:
      return partial(_jitted_infer, interned_static(graphdef), state)
    return partial(_jitted_infer_exec, interned_static(graphdef), executor.config, state)

  # ------------------------------------------------------------------
  # Best-of-N scoring (optional capability -- see barl_jax.executors)
  # ------------------------------------------------------------------

  #: Whether `score()` is a real critic-backed scorer. False by default;
  #: `RLPolicy` (SAC/AMDP/DIME/SDAC) sets this True. Checked eagerly by
  #: `ActionExecutor.validate` before any `needs_score` executor is ever
  #: traced.
  supports_bon = False

  def score(self, observation, action_chunk):
    """`(observation, action_chunk) -> score`, one scalar per batch row,
    pinned to this policy's *current* weights (same lifecycle as
    `inference_fn`). Only meaningful when `supports_bon` is True."""
    raise NotImplementedError(f"{type(self).__name__} does not support Best-of-N scoring")

  def measure_latency(self, n: int = 10) -> float:
    """Rough per-epoch regression signal, not a rigorous benchmark."""
    dummy = jnp.zeros((1, *self.observation_shape))
    act = self.inference_fn()
    key = jax.random.PRNGKey(0)
    action, _ = act(dummy, key)  # warm-up: triggers the jit compile
    jax.block_until_ready(action)

    start = time.perf_counter()
    for _ in range(n):
      action, _ = act(dummy, key)
    jax.block_until_ready(action)
    elapsed = time.perf_counter() - start

    return 1000 * elapsed / n

  # ------------------------------------------------------------------
  # Abstract interface
  # ------------------------------------------------------------------

  @abstractmethod
  def train_step(self, batch: dict) -> dict:
    """Returns `losses`, whose keys must exactly match create_optimizer()'s.
    Diagnostics go to `self.metrics` / a component's own `self.metrics`, not a
    return value -- the training loop drains them once per step."""
    ...

  @abstractmethod
  def inference_step(self, observation, step_info: Optional[dict] = None, key=None) -> tuple:
    """Returns `(action_chunk, extras)` where `action_chunk` always has shape
    `(batch, chunk_size, act_dim)` and `extras` is a dict of named arrays to
    log/persist alongside the currently active chunk."""
    ...



def _materialize_component_config(cls: type, mapping: dict, *, owner: str):
  """Build the typed config a component declares via `config_type`.

  The stored mapping is canonical (`class` selector + nested values). A `None`
  config (e.g. a squash built with no knobs) passes through; a component
  without `config_type` receives the raw mapping for backwards-compatible
  plugin constructors.
  """
  if mapping is None:
    return None
  config_type = getattr(cls, "config_type", None)
  if config_type is None:
    return None if not mapping else dict(mapping)
  values = dict(mapping)
  # Public selector is `class`; dataclass field is `class_name`. The class is
  # already known (`cls`), so neither needs passing to the constructor -- pop
  # both to avoid requiring every impl schema to declare identity metadata it
  # never reads (entropy/ categorical bases historically omitted it).
  values.pop("class", None)
  values.pop("class_name", None)
  from barl_jax.config.serialization import construct
  try:
    return construct(config_type, values, path=owner)
  except (KeyError, TypeError) as exc:
    raise KeyError(f"{owner}: cannot materialize {config_type.__name__}: {exc}") from exc



def register_policy(cls: type):
  name = cls.__name__
  if name in Policy._classes:
    raise ValueError(f"Policy {name} already registered")
  Policy._classes[name] = cls
  return cls
