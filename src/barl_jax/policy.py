import json
import os
import time
from contextlib import contextmanager
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

from barl_jax.action_chunking import ActionChunkExecutor, normalize_action_chunk
from barl_jax.configs import ConfigMixin
from barl_jax.env_stats import EnvStats
from barl_jax.hparam import Config, HParam
from barl_jax.metrics import Recorder, armed


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


def sanitize_action(action, xp=jnp):
  """Force a canonical action into the contract every consumer already assumes:
  finite, and inside [-1, 1] per dimension.

  A failsafe, not a fix. A NaN action reaching `env.step()` is unrecoverable --
  MuJoCo either raises or silently poisons the physics state, and the run dies
  thousands of steps later somewhere unrelated -- so the cost of clamping is
  bounded and the cost of not clamping is not. NaN maps to 0 (the neutral
  action) rather than to a bound, since a bound is a *decision* and 0 is the
  absence of one; infinities saturate to the bound they were heading for.

  `xp` selects the array module: the rollout loop scales host-side in numpy and
  must not be pulled back onto the device to do this. Both modules implement
  `nan_to_num`/`clip` identically here."""
  return xp.clip(xp.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)


def action_bounds_numpy(action_space) -> tuple:
  """Host-side `(scale, bias)` derived from the env's action space."""
  if isinstance(action_space, gym.spaces.Discrete):
    return None, None
  low = np.asarray(action_space.low, dtype=np.float32)
  high = np.asarray(action_space.high, dtype=np.float32)
  return (high - low) / 2, (high + low) / 2


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


_GRAD_STEP = None


@contextmanager
def gating(step):
  """Publishes the gradient-step index for `update_due` for the duration.

  It has to arrive from outside the policy state, because the trainer vmaps the
  step over seeds: a predicate read from `policy.train_steps` is *batched*, and
  `jax.vmap` lowers a `lax.cond` on a batched predicate to a `select` -- both
  branches executed, the loser discarded. Every seed takes the same gradient
  step at the same time, so the counter is genuinely shared and passing it in
  with `in_axes=None` keeps the branch a branch."""
  global _GRAD_STEP
  previous = _GRAD_STEP
  _GRAD_STEP = step
  try:
    yield
  finally:
    _GRAD_STEP = previous


def update_due(policy, every: int):
  """Whether the optimizer key gated at `every` actually updates this gradient
  step -- the same predicate `build_optimizer`'s `conditionally_mask` uses, so a
  loss guarded by this is skipped exactly when its update would be discarded.

  Returns a python `True` (not a traced one) when the step is ungated or when a
  metric tier is armed -- see `gated_loss` for why that distinction matters."""
  if every <= 1 or armed():
    return True
  step = policy.train_steps[...] if _GRAD_STEP is None else _GRAD_STEP
  return (step + 1) % every == 0


def gated_loss(policy, every: int, fn, *modules):
  """`fn(*modules) -> scalar`, evaluated only on the gradient steps where the
  optimizer key gated at `every` actually updates -- zero on the rest, which is
  parameter-identical as long as `fn` carries no gradient into an ungated key
  (both callers price through `frozen`).

  The python `if` is load-bearing: `lax.cond` traces *both* branches whatever
  the predicate, so a metric a gated loss records would be missing from the
  skip branch and the two pytrees would not match. On a logged step the guard is
  python-`True`, `fn` runs unbranched and every metric is real; on an unlogged
  one nothing is recorded at all, so both branches agree."""
  due = update_due(policy, every)
  if due is True:
    return fn(*modules)
  return nnx.cond(due, fn, lambda *_: jnp.zeros(()), *modules)


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


class HParamsOwner:
  """Collects class-attribute declarations (HParam and Config) and resolves
  them to instance attributes against a ConfigMixin of the YAML-override layer.

  `_hparams` is every searchable `HParam` in the MRO (own declarations win), and
  `_configs` every structural `Config` -- together `_decls`, which is what gets
  resolved. `resolve_hparams` reads each against `config`: the YAML value if
  present, the declaration's default otherwise -- which is what moves the
  default out of YAML and into the class that uses the value. Because the
  default lives in the declaration, the resolved config a run logs can
  attribute every value to a layer, and no value is ever silent.

  Two nnx constraints shape the storage:
  - a value resolved onto `self` before `Component.__init__` has run is fine
    for a scalar, but a subclass building a nested module first passes
    `env_stats` so `obs_dim`/`action_dim` exist too;
  - an unhashable value (a dict `norm_kwargs`, `time_kwargs`, `ladder_kwargs`)
    cannot live on the module as a plain attribute -- it would make the
    GraphDef unhashable -- so it is held in a single identity-hashable
    `_Static` and read back through `hparam(name)`.

  `nnx.Module` is not involved here: this mixin is pure Python and only reads
  the class hierarchy at `__init_subclass__` time."""

  _hparams: dict[str, HParam] = {}
  _configs: dict[str, "Config"] = {}

  def __init_subclass__(cls, **kw):
    # Every class in the MRO contributes its own declarations, including plain
    # geometry mixins (CriticOptim, CriticLossConfig, Polyak, ...) that are not
    # themselves
    # HParamsOwner -- a mixin *is* the config, and inheritance is how a class
    # composes it. A later (more-derived) declaration wins over an earlier one.
    super().__init_subclass__(**kw)
    hparams, configs = {}, {}
    for base in reversed(cls.__mro__):
      for name, value in base.__dict__.items():
        if isinstance(value, HParam):
          hparams[name] = value
        elif isinstance(value, Config):
          configs[name] = value
    cls._hparams = hparams
    cls._configs = configs

  @property
  def _decls(self) -> dict[str, Config]:
    """Every declaration, hparams overriding configs of the same name."""
    return {**type(self)._configs, **type(self)._hparams}

  def resolve_hparams(self, config: ConfigMixin, names: tuple = None,
                      env_stats: EnvStats = None) -> None:
    """Resolve every (or `names`-subset of) declared HParam/Config. Idempotent:
    a name already resolved (pre-super, by a subclass that needed a value to
    build a nested module) is re-set to the same value. A style-tagged Config
    (a `net_style`/`time_emb` selector) additionally materializes its named
    style's whole surface as instance attributes, so the component reads
    `self.activation`/`self.norm`/... exactly as if it had declared them."""
    if env_stats is not None:
      self._derive_env(env_stats)
    decls = self._decls if names is None else {n: self._decls[n] for n in names}
    owner = type(self).__name__
    for name, decl in decls.items():
      value = config.resolve(name, decl, owner=owner)
      if decl.style is not None:
        # A style-selector Config: materialize the chosen style's whole
        # surface so the component reads self.activation/self.norm/... as if
        # it had declared them.
        for surface_key, surface_value in decl.style[value].items():
          # The resolver has already layered slot/profile/run overrides over the
          # selected style. Consume that effective value rather than rebuilding
          # the style default here, or the logged config and built module diverge.
          self._absorb(surface_key, config.get(surface_key, type(surface_value)))
      self._absorb(name, value)

  def _absorb(self, name: str, value) -> None:
    """Store a resolved value: unhashable (dict/list/set) values are kept out of
    the GraphDef in a single identity-hashable `_Static`; the rest as plain
    instance attributes."""
    if isinstance(value, (dict, list, set)):
      if not hasattr(self, "_hparams_s"):
        self._hparams_s = _Static({})
      self._hparams_s.value[name] = value
    else:
      setattr(self, name, value)

  def hparam(self, name: str):
    """A resolved value that could not live on the module (a dict, list or set
    declaration) -- see `resolve_hparams`."""
    return self._hparams_s.value[name]

  def _derive_env(self, env_stats: EnvStats) -> None:
    self.obs_dim    = env_stats.get("obs_dim")
    self.action_dim = env_stats.get("act_dim")


class Component(HParamsOwner):
  """Mixin for anything Policy.build_component() constructs. Three jobs: consume
  the run's EnvStats and resolved config, which every component takes as its
  first two positional arguments; give the owning Policy a uniform lifecycle
  to fan out to -- it registers each slot it builds and forwards both hooks
  below, so no Policy has to hand-roll a critic's Polyak update from its own
  train_step; and hold a `metrics` sink already namespaced to this component's
  slot, so writing `self.metrics.dist("critic", pred)` lands at `critic/q/*`
  without the component ever naming itself.

  Unlike the torch original, the EnvStats itself is *not* retained -- only the
  two scalars every component wants. Anything else a component needs from it
  (HLGaussCritic's return range) is read in __init__ and kept as a plain float.

  `config` (a `barl_jax.configs.ConfigMixin`) is likewise not retained: a
  subclass reads whatever it needs via `config.get("key", type)` once, in its
  own __init__, and keeps the result as a plain attribute -- exactly the
  pattern a keyword-only kwarg + `self.x = x` used to be, just sourced from
  the resolved config instead of from a named parameter. Threading `config`
  through every level's `super().__init__(env_stats, config, ...)` (the same
  way `env_stats` already is) is what lets a subclass read config values
  *before* calling super() -- e.g. to build a `q_function` -- with no ordering
  constraint a stored `self.config` would have imposed."""

  def __init__(self, env_stats: EnvStats = None, config: ConfigMixin = None, **kwargs):
    # Neither env_stats nor config is forwarded: nnx.Module's __init__ is
    # object's, taking no args. A component that needs nothing beyond what
    # Component itself reads (e.g. TanhSquash) just never calls config.get().
    super().__init__()
    env_stats = env_stats if env_stats is not None else EnvStats()
    self.obs_dim    = env_stats.get("obs_dim")
    self.action_dim = env_stats.get("act_dim")
    # Both are overwritten by build_component with the real slot; the defaults
    # only keep a component constructed outside a Policy usable.
    self.slot       = type(self).__name__.lower()
    # Every declared Config/HParam class attribute becomes an instance
    # attribute: the YAML layer's value if present, the declaration's default
    # otherwise. A subclass that needed values before this ran (to build a
    # nested module first) calls resolve_hparams itself and this re-sets the
    # same values.
    if config is not None:
      self.resolve_hparams(config, env_stats=env_stats)

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

  def create_optimizer(self) -> dict:
    """Optimizer key -> optax transform. The keys must match the keys this
    component's owning Policy returns from train_step."""
    return {}

  def derived(self) -> dict:
    """Values fixed at construction from something other than this component's
    own config -- an EnvStats entry, or a quantity the owning Policy computed and
    passed in (a critic's support, AMDP's erf_k). They are as load-bearing as any
    resolved kwarg and are not recoverable from config/ alone, so a Policy
    collects them into the run's logged config under `model/derived.*` rather
    than leaving them to be inferred."""
    return {}


class Actor(Component, nnx.Module, ABC):
  """A network that can produce actions and train itself against a critic.

  The shared contract that lets `RLPolicy`/`MaxEntRLPolicy` back every
  actor-critic algorithm here: `generate_action` is the one entry point every
  Policy calls for a stop-gradiented bootstrap draw, `loss` is this actor's own
  training objective, and `extra_losses` is how an actor with more than one
  objective (SDAC's exploration-noise term, trained on its own optimizer key)
  states the rest without a second calling convention. `loss` *returns* its
  scalar rather than mutating a `losses` dict, because the owning Policy runs
  it under `gated_loss` (see `gated_loss`) -- an `nnx.cond` branch has to hand
  back a value on both sides, it cannot mutate a dict on one and not the other.

  `critic` reaches an actor only as an explicit keyword at call time, never
  stored: freezing it (`frozen(critic)`) is a per-call decision, and a stored
  reference would make the actor's subtree non-independent for anything that
  splits or clones it. Whatever draws randomness takes an explicit `key` --
  there is no global RNG."""

  @abstractmethod
  def generate_action(self, obs, *, critic: "Critic" = None, key=None) -> tuple:
    """(action, entropy): a canonical [-1, 1] action, one row per obs row, and
    the differential entropy of whatever distribution produced it (shaped
    `(B, 1)`, or a scalar for an actor whose "entropy" is a property of a
    parameter rather than of the draw -- SDAC's `noise_entropy`)."""

  @abstractmethod
  def loss(self, obs, critic: "Critic", alpha, *, key=None):
    """This actor's own scalar training objective at `obs`, priced against a
    frozen `critic`. Never mutates a `losses` dict -- see the class docstring."""

  def extra_losses(self, obs, critic: "Critic", alpha, *, key=None) -> dict:
    """Any further optimizer-keyed losses this actor trains, beyond `loss`'s
    `"actor"` key -- e.g. SDAC's `"explore"`. Every key here must be one
    `create_optimizer()` also declares, and none of them are gated by
    `optimizer_cadence` (that only ever names `"actor"`/`"entropy_alpha"`), so
    an actor that wants a delayed extra loss has to gate it itself."""
    return {}

  def sample(self, obs, *, critic: "Critic" = None, key=None):
    """The played action alone -- the default just discards the entropy
    `generate_action` also returns. `Policy.inference_step` and the rollout
    loop call this, never `generate_action` directly."""
    return self.generate_action(obs, critic=critic, key=key)[0]


class Critic(Component, nnx.Module, ABC):
  """A network that can price (s,a) -- pure architecture; Bellman targets
  and the regression loss itself live in each Policy's train_step."""

  @abstractmethod
  def value(self, s, a):
    """The only accessor an algorithm needs: a scalar price for (s, a). How that
    scalar is arrived at -- ensemble reduction, quantile mean, categorical
    expectation, pessimism -- is the implementation's business, never the
    caller's."""
    ...


@partial(jax.jit, static_argnums=(0,))
def _jitted_infer(graphdef, state, observation, key):
  """The compiled body behind `Policy.inference_fn`. Module-level so its jit
  cache survives across calls: repeated `nnx.split` of the same policy yields an
  equal-and-equally-hashing GraphDef, so a rollout or an eval wave rebinding its
  weights hits the cache instead of recompiling."""
  action, extras = nnx.merge(graphdef, state).inference_step(observation, key=key)
  return action, extras


class Policy(HParamsOwner, nnx.Module, ABC):

  _classes = dict()

  # True for strictly on-policy algorithms (e.g. PPO) -- the training script
  # clears the replay buffer after each rollout's training pass when set.
  on_policy = False

  # The component slots this Policy builds, by attribute name -- the contract a
  # model YAML's `components:` must match. An omitted slot already fails at
  # build_component(); naming an *extra* one would resolve silently into the
  # snapshot and the wandb config with nothing behind it, so lint_config.py and
  # model_config.validate() check both directions against this.
  SLOTS: tuple[str, ...] = ()

  # Parameters the Policy owns directly rather than through a slot, mapped to
  # the optimizer key that trains them: `{"log_std": "explore"}`. Without an
  # entry a stray parameter is a hard error rather than a silently untrained
  # one (torch's add_param_group had the same job, minus the check).
  policy_params: dict[str, str] = {}
  chunk_size = Config(1)

  # ------------------------------------------------------------------
  # Registry
  # ------------------------------------------------------------------

  @staticmethod
  def make(name: str, env: gym.Env, *args, **kwargs) -> "Policy":
    from barl_jax.model_config import resolve_model_name
    name, profile = resolve_model_name(name)
    if profile is not None:
      kwargs.setdefault("profile", profile)

    cls = Policy._class_for(name)
    return cls(env, *args, model_name=name, **kwargs)

  @staticmethod
  def _class_for(name: str) -> type:
    """A model or class name -> the `Policy` subclass to build.

    Model-name-first: `config/models/<name>.yaml`'s `policy:` key (defaulting to
    `name` itself) is checked before the raw class registry, which is what lets
    `Policy.load` resolve an old checkpoint's `meta["policy_class"]` (e.g.
    `"SDAC"`) to whatever template now implements it (`MaxEntRLPolicy`) without
    a legacy shim class. A bare class name with no model block (there is no
    requirement that one exists) falls through to the registry directly."""
    from barl_jax.model_config import tree
    t = tree()
    if t.policy_cfg(name):
      class_name = t.policy_class_name(name)
    else:
      class_name = name
    if class_name not in Policy._classes:
      raise KeyError(f"{class_name!r} (from model {name!r}) not in registered "
                     f"policies: {list(Policy._classes.keys())}")
    return Policy._classes[class_name]

  @staticmethod
  def available() -> list[str]:
    """Every model name a `model:` string can resolve to -- not the raw class
    registry, which is an implementation detail once one class can back several
    models."""
    from barl_jax.model_config import tree
    return tree().available_models()

  # ------------------------------------------------------------------
  # Construction
  # ------------------------------------------------------------------

  def __init__(self, env: gym.Env, profile: str = None, env_stats: EnvStats = None,
               rngs: nnx.Rngs = None, seed: int = 0, model_name: str = None, **kwargs):
    from barl_jax.model_config import get_overrides, get_snapshot, snapshot_for

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
    # from -- see save()/load().
    self._overrides_s  = _Static(get_overrides())
    self._snapshot_s   = _Static(snapshot_for(self.model_name, profile))
    self._replaying    = get_snapshot() is not None

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

    # Gradient steps taken, for algorithms whose update cadence is `step % k`.
    self.train_steps = Buffer(jnp.zeros((), jnp.int32))

    # The Policy's own declared params (Config/HParam) are resolved here against
    # the resolved `params:` layer (model YAML, profile and run
    # overrides). Subclasses read them off `self` (self.policy_delay, ...) -- the
    # same contract components get from Component.__init__.
    params_config = ConfigMixin("", self.resolve_params(),
                                config_path=f"{self.model_name}.params")
    self.resolve_hparams(params_config)
    self._env_stats = _Static(stats.derive(
      chunk_size=self.chunk_size,
      chunked_act_dim=self.chunk_size * self.action_dim))
    # A replay may carry keys an older implementation consumed but today's no
    # longer does. Recorded values remain useful provenance and must not make an
    # otherwise loadable checkpoint fail; strict leftovers apply to live config.
    if not self._replaying:
      params_config.assert_all_consumed(owner=f"{self.model_name}.params [{type(self).__name__}]")

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

  @property
  def _snapshot(self) -> dict:
    return self._snapshot_s.value



  def resolve_component(self, slot: str) -> tuple[type, dict]:
    """Resolves a named slot's class + kwargs from config/models.yaml.

    Policy constructor kwargs are deliberately not broadcast into every slot:
    slot overrides use the normal `components.<slot>.<key>` path, so a policy
    parameter cannot accidentally retune every component with the same name."""
    from barl_jax.model_config import resolve_slot
    cls, kwargs = resolve_slot(self.model_name, slot, self._profile)
    return cls, kwargs

  def resolve_params(self) -> dict:
    """Resolves this Policy's own top-level tunable kwargs (read directly off
    `self` in train_step, not part of any actor/critic slot) from
    config/models.yaml's `params:` block -- same precedence as
    resolve_component(), minus the components.yaml layer."""
    from barl_jax.model_config import resolve_params
    params = resolve_params(self.model_name, self._profile)
    return {**params, **self._raw_kwargs}

  def build_component(self, slot: str, gate_cls: type, *args,
                      env_stats: EnvStats = None, config_overrides: dict = None,
                      **explicit_kwargs):
    """Resolves + constructs a named slot (actor/critic/squash/...), gated to
    `gate_cls`: the resolved class must be `gate_cls` or a subclass. `env_stats`
    overrides the run's own for a slot another component wraps and feeds a
    transformed input -- pass an `EnvStats.derive()` of the run's, never an
    unrelated one.

    `explicit_kwargs` are for values that were never going to live in config/
    because they're computed at run time and the component takes them as real
    constructor arguments (a critic's `support`, an entropy dual's `target`) --
    everything config *does* supply reaches the component through
    `config.get()`, never through this. `config_overrides` is the rarer case
    of a value that *is* a config-shaped tunable (so config/ may still declare
    a baseline for it) but that this call site pins regardless -- NCLQL's
    `sigmas` slot's `steps`, forced to `self.noise_levels` rather than
    whatever config/models/nclql.yaml says, is the one real example."""
    cls, kwargs = self.resolve_component(slot)
    if config_overrides:
      kwargs.update(config_overrides)
    assert issubclass(cls, gate_cls), (
      f"config/models.yaml: {self.model_name}'s {slot!r} slot names {cls.__name__}, "
      f"which is not a subclass of {gate_cls.__name__} -- {self.model_name} can only "
      f"work with {gate_cls.__name__} (or a subclass of it) there."
    )
    stats = env_stats if env_stats is not None else self.env_stats
    config = ConfigMixin("", kwargs, config_path=f"{self.model_name}.{slot}")
    explicit_kwargs.setdefault("rngs", self.rngs)
    component = cls(stats, config, *args, **explicit_kwargs)
    if not self._replaying:
      config.assert_all_consumed(owner=f"{self.model_name}.{slot} [{cls.__name__}]")
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
    self.train_steps[...] += 1
    for component in self.components.values():
      component.after_step()

  # ------------------------------------------------------------------
  # Optimizers
  # ------------------------------------------------------------------

  def create_optimizer(self) -> dict:
    """Optimizer key -> optax transform, unioned over every slot. A Policy with
    a tunable of its own (an exploration std owned by no component) overrides
    this and adds its key."""
    out = {}
    for component in self.components.values():
      for key, transform in component.create_optimizer().items():
        if key in out:
          raise KeyError(f"{self.model_name}: optimizer key {key!r} is owned by more than one "
                         "component slot")
        out[key] = transform
    return out

  def optimizer_cadence(self) -> dict:
    """Optimizer key -> grad steps between its updates; absent means every step.

    TD3's delayed policy update, which dime_am carries as `policy_delay: 3`
    (`am_diff.py`: the critic steps every gradient step, `update_actor` and
    `update_temperature` only when `(n_updates + i + 1) % policy_delay == 0`).
    The point is to let the critic settle before the actor chases it, so it is a
    property of the algorithm and belongs to the Policy rather than to a slot."""
    return {}

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
      f"{self.model_name}: parameter {'.'.join(map(str, path))} belongs to no "
      f"component slot, so no optimizer key owns it. Either build it through "
      f"build_component(), or declare it in `policy_params`."
    )

  def build_optimizer(self, *, lr_schedule):
    """One optax.multi_transform over the whole param tree. multi_transform
    applies each transform to its own partition, so each component's own clip
    and Adam see only that component's gradients -- exactly what the torch
    loop's clip_grad_norm_(params_of_key_k, ...) did, in one backward pass
    instead of one per key.

    The clip itself lives in the component (networks/optim.adam, spent from
    config/presets/optim.yaml), not here: `global_norm` and `element` are
    different operations rather than two strengths of one, and which of them a
    net wants is a property of that net's gradients."""
    cadence = self.optimizer_cadence()

    transforms = {}
    for key, tx in self.create_optimizer().items():
      every = int(cadence.get(key, 1))
      if every > 1:
        # Masks the *update* and leaves the inner state alone, so a skipped step
        # does not decay Adam's moments -- which zeroing the gradient instead
        # would not give: Adam with g=0 still moves on accumulated momentum.
        # Wrapped around `tx` alone rather than the whole chain, so the LR
        # schedule outside it still advances once per grad step.
        tx = optax.conditionally_mask(tx, lambda step, k=every: (step + 1) % k == 0)
      transforms[key] = (tx if lr_schedule is None else
                         optax.chain(tx, optax.scale_by_schedule(lr_schedule)))
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
      # The model name is what resolves back to a class through _class_for
      # (config-first, so an old sidecar resolves through whatever template now
      # implements it); policy_class is kept as diagnostics -- the only record
      # of which template actually produced these weights.
      "model":        self.model_name,
      "policy_class": type(self).__name__,
      "profile":      self._profile,
      "kwargs":       self._raw_kwargs,
      "overrides":    self._overrides,
      # The whole config this policy was built from, frozen. Without it a
      # checkpoint silently reconstructs a *different* architecture the moment
      # config/ is edited underneath it -- the weights would still load, since
      # only the shapes have to agree.
      "snapshot":     self._snapshot,
    }
    with ocp.StandardCheckpointer() as ckpt:
      ckpt.save(os.path.join(path, "state"), nnx.state(self), force=True)
    with open(os.path.join(path, "meta.json"), "w") as f:
      json.dump(meta, f, indent=2)

  @staticmethod
  def load(name: str, path: str, env: gym.Env, env_stats: EnvStats = None, **kwargs) -> "Policy":
    """`name` is a model name where the sidecar has one (`meta["model"]`) and
    falls back to the raw class name for a sidecar saved before that key
    existed -- `_class_for` resolves either through the same model-first path
    `Policy.make` uses, so an old `"SDAC"` sidecar loads into whatever template
    now implements it."""
    from barl_jax import model_config
    path = os.path.abspath(path)
    with open(os.path.join(path, "meta.json")) as f:
      meta = json.load(f)

    cls = Policy._class_for(name)

    # The architecture has to be back in place before the components are
    # constructed, and it has to be *this checkpoint's*, not whatever config/
    # says now -- the weights match one and not the other. Both the frozen
    # snapshot and the run's dotted overrides are restored, and both are undone
    # afterwards so loading a checkpoint does not silently redefine the config
    # for the rest of the process (which it used to).
    previous = (model_config.get_snapshot(), model_config.get_overrides())
    try:
      model_config.set_snapshot(meta.get("snapshot"))
      model_config.set_overrides(meta.get("overrides"))
      # env_stats is a function of the env, so it is rebuilt rather than persisted.
      policy = cls(env, profile=meta.get("profile"), env_stats=env_stats,
                   model_name=name, **{**meta.get("kwargs", {}), **kwargs})
    finally:
      model_config.set_snapshot(previous[0])
      model_config.set_overrides(previous[1])

    with ocp.StandardCheckpointer() as ckpt:
      restored = ckpt.restore(os.path.join(path, "state"), nnx.state(policy))
    nnx.update(policy, restored)
    return policy

  @staticmethod
  def load_any(path: str, env: gym.Env, env_stats: EnvStats = None, **kwargs) -> "Policy":
    with open(os.path.join(os.path.abspath(path), "meta.json")) as f:
      meta = json.load(f)
    return Policy.load(meta.get("model") or meta["policy_class"], path, env, env_stats, **kwargs)

  # ------------------------------------------------------------------
  # Observation normalisation
  # ------------------------------------------------------------------

  def normalize_observations(self, obs):
    # `obs_std` is all-ones today (the running statistics are never written), so
    # this is the identity -- but the division is the hazard: if a running-std
    # wiring ever lands on a zero-variance channel it must clamp rather than
    # blow up every normalize_obs caller.
    return (obs - self.obs_mean[...]) / jnp.maximum(self.obs_std[...], 1e-6)

  # ------------------------------------------------------------------
  # Action scaling
  # ------------------------------------------------------------------

  # Every continuous actor trains against a canonical per-dimension [-1, 1]
  # output, independent of the env's real bounds; scale/unscale_action are
  # the one-place affine remap, used only at env.step() call sites.

  def scale_action(self, raw_action):
    """Canonical [-1, 1]-per-dimension actor output -> this env's actual action bounds.

    Sanitizes first, so the affine map cannot carry a NaN or an out-of-box value
    into the env: outside [-1, 1] it would land outside the env's own bounds too,
    which gymnasium does not check for MuJoCo."""
    if self.action_discrete:
      return raw_action
    return sanitize_action(raw_action) * self.act_scale[...] + self.act_bias[...]

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

  def inference_fn(self):
    """A pure, jitted `(observation, key) -> (action, extras)` pinned to this
    policy's *current* weights. Rebuild it after training (the rollout loop
    does, once per iteration); it deliberately does not see later updates.
    One compile per batch shape.

    The key is an argument rather than drawn from the module's own `nnx.Rngs`:
    that keeps the traced function pure, lets the parameter state be captured
    once per rollout instead of round-tripped every step, and leaves the caller
    holding the one piece of state that actually has to advance.

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

    return partial(_jitted_infer, graphdef, state)

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

  DECAYS = {
    # Quarter-sine 1 -> 0: flat for the first part of the run, steepest at the end.
    "sine":     lambda progress: jnp.sin((1.0 - progress) * jnp.pi / 2),
    "linear":   lambda progress: 1.0 - progress,
    "cosine":   lambda progress: 0.5 * (1.0 + jnp.cos(progress * jnp.pi)),
    "constant": lambda progress: jnp.ones_like(progress),
  }

  @staticmethod
  def lr_schedule(warmup_steps: int, total_steps: int, decay: str):
    """Multiplicative LR factor: linear 0 -> 1 ramp over `warmup_steps`, then
    `decay` from 1 -> 0 by `total_steps` (flat past that, in case `total_steps` --
    an estimate -- undershoots the run's actual grad-step count). `decay` is the
    run config's `lr_decay`; it used to be hardcoded to "sine" here, which meant
    the schedule's shape was the one part of the optimizer no run recorded."""
    if decay not in Policy.DECAYS:
      raise KeyError(f"lr_decay {decay!r} is not one of {', '.join(Policy.DECAYS)}")
    decay_fn = Policy.DECAYS[decay]

    def schedule(step):
      step = jnp.asarray(step, jnp.float32)
      warm = step / max(1, warmup_steps)
      progress = jnp.clip((step - warmup_steps) / max(1, total_steps - warmup_steps), 0.0, 1.0)
      decayed = decay_fn(progress)
      return jnp.where(step < warmup_steps, warm, decayed) if warmup_steps > 0 else decayed
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
    """Returns `(action_chunk, extras)` where `action_chunk` always has shape
    `(batch, chunk_size, act_dim)` and `extras` is a dict of named arrays to
    log/persist alongside the currently active chunk."""
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
    return run_episodes(self, env, n, render, **inference_kwargs)


def episode_success(info, i: int, is_vec: bool, terminated: bool) -> bool:
  """Did the episode that just ended on slot `i` succeed?

  Where the env says so itself, that answer wins: ManiSkill reports
  `info["success"]` and sets `terminated` *from* it, so the survival convention
  below would read its success rate exactly backwards. Every other family here
  reports no such key, and for them an episode that never terminated is one the
  agent did not fail out of -- a MuJoCo humanoid terminates by falling over.

  Gymnasium's vector envs batch each sub-env info key into an array plus an
  `_<key>` presence mask (a key only some sub-envs reported); a plain env just
  carries the key."""
  value = info.get("success")
  if value is None:
    return not terminated
  if not is_vec:
    return bool(np.asarray(value).reshape(-1)[0])
  present = info.get("_success")
  if present is not None and not bool(np.asarray(present)[i]):
    return not terminated
  return bool(np.asarray(value)[i])


def run_episodes(actor, env: gym.Env, n: int, render: bool, **inference_kwargs) -> list[dict]:
  """`n` complete episodes, in parallel waves of `num_envs` if `env` is a
  VectorEnv. A slot that finishes before the rest of its wave is reset and
  stops recording -- autoreset_mode=DISABLED asserts rather than letting a
  finished slot be stepped again -- so a wave costs its longest episode.

  `actor` is anything exposing `inference_fn()`, usually a Policy or a
  StackedPolicy -- the latter is how the trainer evaluates every seed in one
  vmapped pass instead of one loop per seed."""
  if n <= 0:
    return []

  is_vec = isinstance(env, gym.vector.VectorEnv)
  n_envs = env.num_envs if is_vec else 1

  def reset(**kwargs):
    obs, _ = env.reset(**kwargs)
    return obs if is_vec else obs[None, ...]

  act = actor.inference_fn()
  key = jax.random.PRNGKey(0)
  chunk_size = getattr(actor, "chunk_size", 1)
  action_space = env.single_action_space if is_vec else env.action_space
  act_scale_np, act_bias_np = action_bounds_numpy(action_space)
  act_dim = getattr(actor, "action_dim",
                    int(action_space.n) if isinstance(action_space, gym.spaces.Discrete)
                    else int(np.prod(action_space.shape)))

  out = []
  while len(out) < n:
    wave = min(n_envs, n - len(out))
    obs  = reset()
    chunks = ActionChunkExecutor(act, chunk_size=chunk_size, act_dim=act_dim, n_envs=n_envs)
    data = [dict(frames=[], rewards=[], terminated=False, success=False,
                 act_nonfinite=0, n_act=0, chunk_size=chunk_size,
                 chunk_refills=0, chunk_waste_frac=0.0)
            for _ in range(n_envs)]
    live = np.zeros(n_envs, dtype=bool)
    live[:wave] = True

    while live.any():
      key, subkey = jax.random.split(key)
      action_np, _ = chunks.current(obs, subkey, to_device=lambda x: jnp.asarray(x, dtype=jnp.float32),
                                    live=live)
      # Same failsafe the rollout applies (rollout.py). Without it a NaN action
      # reaches mj_step, which only warns and poisons qpos -- the eval returns
      # quietly wrong numbers and the next render() aborts the process inside
      # mjr_render with no python traceback (sweep 2026-08-27_13-23-15).
      nonfinite = ~np.isfinite(action_np)
      action_np = sanitize_action(action_np, xp=np)
      env_action = action_np if act_scale_np is None else action_np * act_scale_np + act_bias_np
      env_action = np.ascontiguousarray(env_action, dtype=np.float32).copy()
      obs, reward, term, trunc, info = env.step(env_action if is_vec else env_action[0])

      if not is_vec:
        obs, reward, term, trunc = obs[None, ...], np.array([reward]), np.array([term]), np.array([trunc])

      frames = (env.call("render") if is_vec else [env.render()]) if render else None

      done = term | trunc
      chunks.advance(done)
      slot_refills = chunks.slot_refills()
      slot_wasted = chunks.slot_wasted()
      for i in np.flatnonzero(live):
        data[i]["rewards"].append(reward[i])
        data[i]["act_nonfinite"] += int(nonfinite[i].sum())
        data[i]["n_act"] += int(nonfinite[i].size)
        if render:
          data[i]["frames"].append(frames[i])
        if done[i]:
          data[i]["terminated"] = bool(term[i])
          data[i]["success"] = episode_success(info, i, is_vec, bool(term[i]))
          data[i]["chunk_refills"] = int(slot_refills[i])
          executed = max(1, data[i]["chunk_refills"] * chunk_size)
          data[i]["chunk_waste_frac"] = float(slot_wasted[i]) / executed
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
