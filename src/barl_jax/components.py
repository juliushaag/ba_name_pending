"""Component contracts and NNX state utilities."""
from abc import ABC, abstractmethod

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from barl_jax.config.toolkit import ConfigMixin
from barl_jax.env import EnvStats
from barl_jax.hparam import Config, HParam
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
          configs.pop(name, None)
        elif isinstance(value, Config):
          configs[name] = value
          hparams.pop(name, None)
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
    (a `mlp_preset`/`time_emb` selector) additionally materializes its named
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
    self.chunk_size = int(env_stats.get("chunk_size") or 1)
    self.plan_dim   = self.chunk_size * (self.action_dim or 0)


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
  scalars every component wants (`obs_dim`/`action_dim`, plus the chunk plan's
  `chunk_size`/`plan_dim`). Anything else a component needs from it
  (HLGaussCritic's return range) is read in __init__ and kept as a plain float.

  `config` (a `barl_jax.config.toolkit.ConfigMixin`) is likewise not retained: a
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
    # The plan width the actor emits / the critic prices: chunk_size primitives
    # flattened. Single actions at chunk_size 1, so every existing shape is the
    # flat special case and no call site reshapes.
    self.chunk_size = int(env_stats.get("chunk_size") or 1)
    self.plan_dim   = self.chunk_size * (self.action_dim or 0)
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


class ActionSampler(Component, nnx.Module, ABC):
  """Canonical action generation, independent of a training objective."""

  @abstractmethod
  def generate_action(self, obs, *, critic: "Critic" = None, key=None) -> tuple:
    """(action, entropy): a canonical [-1, 1] action, one row per obs row, and
    the differential entropy of whatever distribution produced it (shaped
    `(B, 1)`, or a scalar for an actor whose "entropy" is a property of a
    parameter rather than of the draw -- SDAC's `noise_entropy`)."""

  def sample(self, obs, *, critic: "Critic" = None, key=None):
    """The played action alone -- the default just discards the entropy
    `generate_action` also returns. `Policy.inference_step` and the rollout
    loop call this, never `generate_action` directly."""
    return self.generate_action(obs, critic=critic, key=key)[0]


class Actor(ActionSampler):
  """A sampler with an off-policy actor objective."""

  @abstractmethod
  def loss(self, obs, critic: "Critic", alpha, *, key=None):
    """This actor's own training objective at `obs`, priced against a frozen
    `critic` -- a dict with one scalar per optimizer key it trains (just
    `"actor"` for every actor but SDAC's, which also prices its exploration
    noise on `"explore"`). Returns rather than mutating a `losses` dict -- see
    the class docstring. Every key here must be one `create_optimizer()` also
    declares."""



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


