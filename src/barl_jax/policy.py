import json
import os
import time
from functools import partial
from abc import ABC, abstractmethod
from typing import Optional

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx

from barl_jax.env_stats import EnvStats
from barl_jax.metrics import Recorder


class Buffer(nnx.Variable):
  """Non-trainable state: normalization statistics, action bounds, target-net
  weights. Part of the checkpoint and threaded through jit like any other
  state, but never handed to an optimizer (see TRAINABLE)."""


TARGET_SUFFIX = "_target"


def is_target(path, _value) -> bool:
  """An nnx filter: does this parameter belong to a frozen target copy?

  Target nets are `nnx.clone`s and their leaves stay `nnx.Param` -- nnx.update()
  preserves a Variable's original type, so they cannot be retyped into something
  a type filter would skip. The naming convention carries the information, so it
  is stated once here rather than spelled out at each use."""
  return any(str(p).endswith(TARGET_SUFFIX) for p in path)


# What every optimizer differentiates and updates.
TRAINABLE = nnx.All(nnx.Param, nnx.Not(is_target))


def frozen(module):
  """A view of `module` that neither trains nor updates: its parameters carry
  no gradient, and any batch-statistic normalization inside uses its running
  averages instead of accumulating new ones. Inputs still carry gradient, which
  is what an actor improving against a critic needs.

  An nnx Module is a pytree, so the gradient half is just stop_gradient over its
  leaves. (The split/stop_gradient/merge dance this replaces is not usable
  inside a captured or nested transform at all.)

  Freezing the statistics matters as much as freezing the weights. An actor loss
  prices actions the critic was never trained on, so letting those move the
  critic's normalization moments would shift the function under the very
  regression that defines it -- CrossQ evaluates the critic in inference mode
  in its actor update for the same reason. It also keeps the mutation out of a
  nested transform, which nnx rejects outright ("Cannot mutate BatchStat from a
  different trace level") -- AMDP takes its action-score gradient that way."""
  view = jax.lax.stop_gradient(module)
  view.eval()
  return view


class _Static:
  """Opaque holder for Python bookkeeping that must stay out of the state tree
  *and* keep the GraphDef hashable.

  Neither nnx-native option covers this, which is why the holder exists:

  - a bare attribute trips nnx's "data in a static attribute" check as soon as
    the value contains an array (EnvStats is a dict of numpy arrays);
  - `nnx.data()` would make it state and thread it through every jit;
  - `nnx.static()` accepts it but leaves the GraphDef unhashable, and the
    GraphDef is a static argument of the jitted step -- `jax.jit` then raises
    `unhashable type: 'dict'`.

  Identity hashing is the right semantics here: these objects live for the whole
  run, so the jit cache keys on them stably."""

  __slots__ = ("value",)

  def __init__(self, value):
    self.value = value

  def __repr__(self):
    return f"_Static({self.value!r})"


class Component:
  """Mixin for anything Policy.build_component() constructs. Three jobs: consume
  the run's EnvStats, which every component takes as its first positional
  argument; give the owning Policy a uniform lifecycle to fan out to -- it
  registers each slot it builds and forwards all four hooks below, so no Policy
  has to hand-roll a critic's Polyak update from its own train_step; and hold a
  `metrics` sink already namespaced to this component's slot, so writing
  `self.metrics.dist("q", pred)` lands at `critic/q/*` without the component
  ever naming itself.

  Unlike the torch original, the EnvStats itself is *not* retained -- only the
  two scalars every component wants. Anything else a component needs from it
  (HLGaussCritic's return range) is read in __init__ and kept as a plain float."""

  def __init__(self, env_stats: EnvStats = None, **kwargs):
    # Not forwarded: nnx.Module's __init__ is object's, and every component has
    # a **kwargs catch-all that config/components.yaml leftovers land in (that
    # is what model_config.validate() exists to police). Passing them on would
    # just be a TypeError from object.__init__.
    super().__init__()
    env_stats = env_stats if env_stats is not None else EnvStats()
    self.obs_dim    = env_stats.get("obs_dim")
    self.action_dim = env_stats.get("act_dim")
    # Both are overwritten by build_component with the real slot; the defaults
    # only keep a component constructed outside a Policy usable.
    self.slot       = type(self).__name__.lower()

  @property
  def metrics(self) -> Recorder:
    """Diagnostics sink. A facade bound to this module, rebuilt per access and
    holding no state -- the logged key comes from where this component sits in
    the module tree, so it never names itself."""
    return Recorder(self)

  def before_step(self):
    """Called once per gradient step, before the Policy's train_step."""
    ...

  def after_step(self):
    """Called once per gradient step, after every optimizer has stepped."""
    ...

  def on_epoch_start(self):
    ...

  def on_epoch_end(self):
    ...

  def create_optimizer(self) -> dict:
    """Optimizer key -> optax transform. The keys must match the keys this
    component's owning Policy returns from train_step."""
    return {}


class Actor(Component, nnx.Module, ABC):
  """A network that can produce actions -- pure architecture; every
  algorithm-level decision (loss, squash choice, etc.) stays on the owning
  Policy's train_step."""

  @abstractmethod
  def sample(self, s, key, critic: "Critic" = None):
    """`critic` is unused by generative actors; implicit/Q-derived actors need
    it to compute an action at all. `key` is explicit -- there is no global RNG."""
    ...


class Critic(Component, nnx.Module, ABC):
  """A network that can price (s,a) -- pure architecture; Bellman targets
  and the regression loss itself live in each Policy's train_step."""

  @abstractmethod
  def value(self, s, a, target: bool = False):
    """The only accessor an algorithm needs: a scalar price for (s, a), off the
    live net or (`target=True`) the Polyak-averaged one. How that scalar is
    arrived at -- ensemble reduction, quantile mean, categorical expectation,
    pessimism -- is the implementation's business, never the caller's."""
    ...


class Policy(nnx.Module, ABC):

  _classes = dict()

  # True for strictly on-policy algorithms (e.g. PPO) -- the training script
  # clears the replay buffer after each rollout's training pass when set.
  on_policy = False

  # Parameters the Policy owns directly rather than through a slot, mapped to
  # the optimizer key that trains them: `{"log_std": "explore"}`. Without an
  # entry a stray parameter is a hard error rather than a silently untrained
  # one (torch's add_param_group had the same job, minus the check).
  policy_params: dict[str, str] = {}

  # ------------------------------------------------------------------
  # Registry
  # ------------------------------------------------------------------

  @staticmethod
  def make(name: str, env: gym.Env, *args, **kwargs) -> "Policy":
    from barl_jax.model_config import resolve_model_name
    name, profile = resolve_model_name(name)
    if profile is not None:
      kwargs.setdefault("profile", profile)

    if name not in Policy._classes:
      raise KeyError(f"{name} not in registered policies: {list(Policy._classes.keys())}")
    return Policy._classes[name](env, *args, **kwargs)

  @staticmethod
  def available() -> list[str]:
    return list(Policy._classes.keys())

  # ------------------------------------------------------------------
  # Construction
  # ------------------------------------------------------------------

  def __init__(self, env: gym.Env, profile: str = None, env_stats: EnvStats = None,
               rngs: nnx.Rngs = None, seed: int = 0, **kwargs):
    from barl_jax.model_config import get_overrides

    self._profile  = profile
    # Held behind _Static for hashability (see _Static): these are dicts/lists,
    # and the GraphDef they end up in is a static jit argument.
    self._raw_kwargs_s = _Static(dict(kwargs))
    # Captured so a checkpoint reconstructs under the same overrides too.
    self._overrides_s  = _Static(get_overrides())

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
    self.observation_dim   = int(np.prod(observation_space.shape))

    self.obs_std  = Buffer(jnp.ones(observation_space.shape[-1]))
    self.obs_mean = Buffer(jnp.zeros(observation_space.shape[-1]))

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

  @property
  def _overrides(self) -> dict:
    return self._overrides_s.value



  def resolve_component(self, slot: str, **explicit_kwargs) -> tuple[type, dict]:
    """Resolves a named slot's class + kwargs from config/models.yaml (see
    model_config.py), then layers `self._raw_kwargs` and `explicit_kwargs`
    on top (highest precedence -- for algorithm-fixed choices that must never
    be tunable)."""
    from barl_jax.model_config import resolve_slot
    cls, kwargs = resolve_slot(type(self).__name__, slot, self._profile)
    return cls, {**kwargs, **self._raw_kwargs, **explicit_kwargs}

  def resolve_params(self, **explicit_kwargs) -> dict:
    """Resolves this Policy's own top-level tunable kwargs (read directly off
    `self` in train_step, not part of any actor/critic slot) from
    config/models.yaml's `params:` block -- same precedence as
    resolve_component(), minus the components.yaml layer."""
    from barl_jax.model_config import resolve_params
    params = resolve_params(type(self).__name__, self._profile)
    return {**params, **self._raw_kwargs, **explicit_kwargs}

  def build_component(self, slot: str, gate_cls: type, *args,
                      env_stats: EnvStats = None, **explicit_kwargs):
    """Resolves + constructs a named slot (actor/critic/squash/...), gated to
    `gate_cls`: the resolved class must be `gate_cls` or a subclass. `env_stats`
    overrides the run's own for a slot another component wraps and feeds a
    transformed input -- pass an `EnvStats.derive()` of the run's, never an
    unrelated one."""
    cls, kwargs = self.resolve_component(slot, **explicit_kwargs)
    assert issubclass(cls, gate_cls), (
      f"config/models.yaml: {type(self).__name__}'s {slot!r} slot names {cls.__name__}, "
      f"which is not a subclass of {gate_cls.__name__} -- {type(self).__name__} can only "
      f"work with {gate_cls.__name__} (or a subclass of it) there."
    )
    stats = env_stats if env_stats is not None else self.env_stats
    kwargs.setdefault("rngs", self.rngs)
    component = cls(stats, *args, **kwargs)
    component.slot = slot
    return component

  # ------------------------------------------------------------------
  # Metrics
  # ------------------------------------------------------------------

  @classmethod
  def algo_name(cls) -> str:
    """This algorithm's logging namespace -- `SACPolicy` -> `sac`, `AMDP` -> `amdp`."""
    name = cls.__name__.lower()
    return name[:-len("policy")] if name.endswith("policy") and len(name) > len("policy") else name

  def optimizer_slots(self) -> dict[str, str]:
    """Optimizer key -> owning component slot, by asking each slot which keys
    it creates. Keys owned by no component map to this algorithm's own
    namespace. Building the transforms is free -- optax allocates no state
    until init()."""
    cache = self._optimizer_slots_cache.value
    if cache["value"] is None:
      owners = {}
      for slot, component in self.components.items():
        for key in component.create_optimizer():
          owners[key] = slot
      cache["value"] = owners
    return cache["value"]

  def metrics_key(self, optimizer_key: str, field: str) -> str:
    """`critic/loss` when the slot owns one optimizer key, `actor/loss/flow`
    when it owns several -- so the common case stays flat."""
    owners = self.optimizer_slots()
    if optimizer_key not in owners:  # no component owns it -- always name it
      return f"{self.algo_name()}/{field}/{optimizer_key}"
    slot   = owners[optimizer_key]
    shared = sum(1 for o in owners.values() if o == slot) > 1
    return f"{slot}/{field}/{optimizer_key}" if shared else f"{slot}/{field}"

  def record_batch(self, batch: dict):
    """The `batch/*` family -- what every algorithm trains against, recorded
    once here rather than restated by each train_step."""
    self.batch_metrics.dist("reward", batch["rew"])
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
    for component in self.components.values():
      component.after_step()

  def on_epoch_start(self):
    self._epoch_start_time = time.perf_counter()
    for component in self.components.values():
      component.on_epoch_start()

  def on_epoch_end(self):
    for component in self.components.values():
      component.on_epoch_end()

  def epoch_time(self) -> float:
    """Wall-clock seconds since the last on_epoch_start() call."""
    return time.perf_counter() - self._epoch_start_time

  # ------------------------------------------------------------------
  # Optimizers
  # ------------------------------------------------------------------

  def create_optimizer(self) -> dict:
    """Optimizer key -> optax transform, unioned over every slot. A Policy with
    a tunable of its own (an exploration std owned by no component) overrides
    this and adds its key."""
    out = {}
    for component in self.components.values():
      out.update(component.create_optimizer())
    return out

  def optimizer_labels(self) -> dict:
    """The `param path -> optimizer key` tree optax.multi_transform partitions
    on. A slot's name *is* its attribute name (see `components`), so a
    parameter's owner is just the head of its path -- no identity matching
    against the instance dict, and nothing to keep in sync.

    A component owning more than one key overrides `param_labels(subpath)` to
    say which of its sub-attributes maps to which."""
    owners = self.optimizer_slots()
    default_key = {}
    for key, slot in owners.items():
      default_key.setdefault(slot, key)

    components = self.components
    labels = []
    for path, _ in nnx.to_flat_state(nnx.state(self, TRAINABLE)):
      slot = str(path[0])
      component = components.get(slot)
      key = None
      if component is not None:
        split = getattr(component, "param_labels", None)
        key = (split(path[1:]) if split else None) or default_key.get(slot)
      labels.append((path, key or self._fallback_label(path)))
    return nnx.from_flat_state(labels)


  def _fallback_label(self, path) -> str:
    key = self.policy_params.get(str(path[0]))
    if key is not None:
      return key
    raise KeyError(
      f"{type(self).__name__}: parameter {'.'.join(map(str, path))} belongs to no "
      f"component slot, so no optimizer key owns it. Either build it through "
      f"build_component(), or declare it in `policy_params`."
    )

  def build_optimizer(self, lr_schedule=None, max_grad_norm: float = 1.0):
    """One optax.multi_transform over the whole param tree. multi_transform
    applies each transform to its own partition, so clip_by_global_norm is
    computed *per optimizer key* -- exactly what the torch loop's
    clip_grad_norm_(params_of_key_k, 1.0) did, in one backward pass instead of
    one per key."""
    transforms = {}
    for key, tx in self.create_optimizer().items():
      chain = [optax.clip_by_global_norm(max_grad_norm), tx]
      if lr_schedule is not None:
        chain.append(optax.scale_by_schedule(lr_schedule))
      transforms[key] = optax.chain(*chain)
    return optax.multi_transform(transforms, self.optimizer_labels())

  # ------------------------------------------------------------------
  # Checkpointing
  # ------------------------------------------------------------------

  def save(self, path: str):
    """`path` names a directory: orbax writes the state tree into it, and a
    JSON sidecar carries what is needed to reconstruct the architecture."""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta = {
      "policy_class": type(self).__name__,
      "profile":      self._profile,
      "overrides":    self._overrides,
    }
    with ocp.StandardCheckpointer() as ckpt:
      ckpt.save(os.path.join(path, "state"), nnx.state(self), force=True)
    with open(os.path.join(path, "meta.json"), "w") as f:
      json.dump(meta, f, indent=2)

  @staticmethod
  def load(type_name: str, path: str, env: gym.Env, env_stats: EnvStats = None) -> "Policy":
    from barl_jax.model_config import set_overrides
    path = os.path.abspath(path)
    with open(os.path.join(path, "meta.json")) as f:
      meta = json.load(f)
    # The run's dotted overrides are part of the architecture, so they have to
    # be back in place before the components are constructed -- and they beat
    # the resuming run's own, which the weights would not match.
    if meta.get("overrides"):
      set_overrides(meta["overrides"])
    # env_stats is a function of the env, so it is rebuilt rather than persisted.
    policy = Policy._classes[type_name](env, profile=meta.get("profile"), env_stats=env_stats)
    with ocp.StandardCheckpointer() as ckpt:
      restored = ckpt.restore(os.path.join(path, "state"), nnx.state(policy))
    nnx.update(policy, restored)
    return policy

  @staticmethod
  def load_any(path: str, env: gym.Env, env_stats: EnvStats = None) -> "Policy":
    with open(os.path.join(os.path.abspath(path), "meta.json")) as f:
      meta = json.load(f)
    return Policy.load(meta["policy_class"], path, env, env_stats)

  # ------------------------------------------------------------------
  # Observation normalisation
  # ------------------------------------------------------------------

  def set_obs_normalization(self, mean, std):
    self.obs_mean[...] = jnp.asarray(mean, dtype=jnp.float32)
    self.obs_std[...]  = jnp.asarray(std,  dtype=jnp.float32)

  def normalize_obs(self, obs):
    return (obs - self.obs_mean[...]) / self.obs_std[...]

  # ------------------------------------------------------------------
  # Action scaling
  # ------------------------------------------------------------------

  # Every continuous actor trains against a canonical per-dimension [-1, 1]
  # output, independent of the env's real bounds; scale/unscale_action are
  # the one-place affine remap, used only at env.step() call sites.

  def scale_action(self, raw_action):
    """Canonical [-1, 1]-per-dimension actor output -> this env's actual action bounds."""
    if self.action_discrete:
      return raw_action
    return raw_action * self.act_scale[...] + self.act_bias[...]

  def unscale_action(self, action):
    """Inverse of scale_action: real env-scale action -> canonical [-1, 1]."""
    if self.action_discrete:
      return action
    return (action - self.act_bias[...]) / self.act_scale[...]

  def action_bounds_numpy(self) -> tuple:
    """Host-side (scale, bias), for the rollout/eval loops -- scaling on the
    device would cost a second round-trip per step for an affine map on an
    array that is already back on the host."""
    if self.action_discrete:
      return None, None
    return np.asarray(self.act_scale[...]), np.asarray(self.act_bias[...])

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

  def inference_fn(self):
    """A pure, jitted `(observation, key) -> (action, extras)` pinned to this
    policy's *current* weights. Rebuild it after training (the rollout loop
    does, once per iteration); it deliberately does not see later updates.
    One compile per batch shape.

    The key is an argument rather than drawn from the module's own `nnx.Rngs`:
    that keeps the traced function pure, lets the parameter state be captured
    once per rollout instead of round-tripped every step, and leaves the caller
    holding the one piece of state that actually has to advance."""
    # Acting uses running statistics, not the batch's -- a rollout step is a
    # batch of `num_envs`, and a BatchNorm/BatchRenorm critic normalized by that
    # would price actions differently than it was trained to. `use_running_average`
    # is a static attribute, so it has to be set before the split for the
    # compiled function to see it; `.train()` restores the module afterwards.
    self.eval()
    graphdef, state = nnx.split(self)
    self.train()

    @partial(jax.jit, static_argnums=(0,))
    def infer(graphdef, state, observation, key):
      return nnx.merge(graphdef, state).inference_step(observation, key=key)

    return partial(infer, graphdef, state)

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

  @staticmethod
  def lr_schedule(warmup_steps: int, total_steps: int):
    """Multiplicative LR factor: linear 0 -> 1 ramp over `warmup_steps`, then a
    quarter-sine decay 1 -> 0 to `total_steps` (flat at 0 past that, in case
    `total_steps` -- an estimate -- undershoots the run's actual grad-step count)."""
    def schedule(step):
      step = jnp.asarray(step, jnp.float32)
      warm = step / max(1, warmup_steps)
      progress = jnp.clip((step - warmup_steps) / max(1, total_steps - warmup_steps), 0.0, 1.0)
      decay = jnp.sin((1.0 - progress) * jnp.pi / 2)
      return jnp.where(step < warmup_steps, warm, decay) if warmup_steps > 0 else decay
    return schedule

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
    """Returns (action, extras) where extras is a dict of per-step named arrays
    (e.g. entropy bounds) to log/persist."""
    ...

  # ------------------------------------------------------------------
  # Evaluation
  # ------------------------------------------------------------------

  def evaluate(self, env: gym.Env, runs=3, videos: int = None, render_env: gym.Env = None,
               **inference_kwargs):
    """`runs` episodes total: `runs - videos` unrendered on `env` plus `videos`
    rendered on `render_env` (defaulting to `env`), all counting towards the
    reward statistics. `videos=None` renders every run.

    Both may be a single gym.Env or a VectorEnv, and being able to hand this a
    VectorEnv is the point: a single env spends ~90% of eval blocked on batch-1
    device round-trips rather than on the physics. The rendered episodes stay on
    their own env because env.render() costs ~140x the env step and a VectorEnv
    can only render all of its sub-envs at once."""
    n_videos = runs if videos is None else min(videos, runs)
    run_data  = self._run_episodes(env, runs - n_videos, render=False, **inference_kwargs)
    run_data += self._run_episodes(render_env if render_env is not None else env,
                                   n_videos, render=True, **inference_kwargs)
    return run_data

  def _run_episodes(self, env: gym.Env, n: int, render: bool, **inference_kwargs) -> list[dict]:
    """`n` complete episodes, in parallel waves of `num_envs` if `env` is a
    VectorEnv. A slot that finishes before the rest of its wave is reset and
    stops recording -- autoreset_mode=DISABLED asserts rather than letting a
    finished slot be stepped again -- so a wave costs its longest episode."""
    if n <= 0:
      return []

    is_vec = isinstance(env, gym.vector.VectorEnv)
    n_envs = env.num_envs if is_vec else 1

    def reset(**kwargs):
      obs, _ = env.reset(**kwargs)
      return obs if is_vec else obs[None, ...]

    act_scale_np, act_bias_np = self.action_bounds_numpy()
    act = self.inference_fn()
    key = jax.random.PRNGKey(0)

    out = []
    while len(out) < n:
      wave = min(n_envs, n - len(out))
      obs  = reset()
      data = [dict(frames=[], rewards=[], terminated=False) for _ in range(n_envs)]
      live = np.zeros(n_envs, dtype=bool)
      live[:wave] = True

      while live.any():
        key, subkey = jax.random.split(key)
        action, _ = act(jnp.asarray(obs, dtype=jnp.float32), subkey)
        action_np  = np.asarray(action)
        env_action = action_np if act_scale_np is None else action_np * act_scale_np + act_bias_np
        # .copy(): np.asarray() on a JAX array can alias its buffer, and MuJoCo
        # writes through the action array it is handed.
        env_action = np.ascontiguousarray(env_action, dtype=np.float32).copy()
        obs, reward, term, trunc, _ = env.step(env_action if is_vec else env_action[0])

        if not is_vec:
          obs, reward, term, trunc = obs[None, ...], np.array([reward]), np.array([term]), np.array([trunc])

        frames = (env.call("render") if is_vec else [env.render()]) if render else None

        done = term | trunc
        for i in np.flatnonzero(live):
          data[i]["rewards"].append(reward[i])
          if render:
            data[i]["frames"].append(frames[i])
          if done[i]:
            data[i]["terminated"] = bool(term[i])
            live[i] = False

        if live.any() and done.any():
          obs = reset(options={"reset_mask": done}) if is_vec else reset()

      out += data[:wave]

    return out


def register_policy(cls: type):
  name = cls.__name__
  if name in Policy._classes:
    raise ValueError(f"Policy {name} already registered")
  Policy._classes[name] = cls
  return cls
