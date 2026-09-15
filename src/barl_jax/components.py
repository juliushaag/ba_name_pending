"""Component contracts and NNX state utilities."""
from abc import ABC, abstractmethod

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from barl_jax.env import EnvStats
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
  view = jax.lax.stop_gradient(module)
  view.eval()
  return view


class _Static:
  __slots__ = ("value",)

  def __init__(self, value):
    self.value = value

  def __repr__(self):
    return f"_Static({self.value!r})"


class Component:
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

  `config` (a materialized typed dataclass instance) *is* retained, as
  `self.config` -- a subclass reads whatever it needs off it directly
  (`self.config.log_std_min`), in `__init__` or any other method, rather than
  re-declaring every field as its own flat attribute. It is kept behind
  `_Static` (identity-hashed, see `_Static`'s own docstring) rather than a
  bare attribute: a config's nested fields are ordinary mutable containers
  (`MLPConfig.norm_kwargs`, `DDPMActorConfig.ladder_kwargs`, ...), and nnx
  would otherwise need the whole frozen dataclass to hash by value to make it
  a valid static leaf of the split GraphDef `jax.jit` keys its cache on.
  `config` is also a plain settable property: `self.config = replace(self.config,
  gamma=0.5)` is how a test perturbs one field without reconstructing the
  whole component. `env_stats`/`config` are still threaded through every
  level's `super().__init__(env_stats, config, ...)`, which is what lets a
  subclass read the `config` *parameter* (not yet `self.config`) before
  calling super() -- e.g. to build a `q_function`."""

  config_type = None

  def __init__(self, env_stats: EnvStats = None, config=None, **kwargs):
    # Neither env_stats nor config is forwarded: nnx.Module's __init__ is
    # object's, taking no args. A component that needs nothing beyond what
    # Component itself reads (e.g. TanhSquash) just never reads self.config.
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
    self._config_s  = _Static(config)

  @property
  def config(self):
    return self._config_s.value

  @config.setter
  def config(self, value):
    self._config_s.value = value

  def static(self, name: str):
    """Read an explicitly stored derived (non-config) value from NNX static
    state -- values computed from config (`time_kwargs` unpacked from
    `config.time`) rather than a config field itself, which belongs on
    `self.config` instead."""
    return self._static_config.value[name]

  @property
  def metrics(self) -> Recorder:
    """Diagnostics sink. A facade bound to this module, rebuilt per access and
    holding no state -- the logged key comes from where this component sits in
    the module tree, so it never names itself."""
    return Recorder(self)

  def chunk_sum(self, x):
    """An elementwise per-dimension quantity at plan resolution, `(..., plan_dim)`,
    reduced within each chunk position's `action_dim` block: `(..., plan_dim) ->
    (..., chunk_size)`. At chunk_size 1 this is the old full-plan `.sum(axis=-1)`
    unchanged -- a per-chunk-position alpha and a single whole-plan alpha are the
    same code path, just multiplied against a differently-shaped alpha."""
    return x.reshape(*x.shape[:-1], self.chunk_size, self.action_dim).sum(axis=-1)

  def chunk_broadcast(self, alpha):
    """A per-chunk-position alpha, shape `()` or `(chunk_size,)`, widened to
    `(plan_dim,)` by repeating each position's value across its `action_dim`
    block -- the inverse reshape of `chunk_sum`, for a term that lives at plan
    resolution (AMDP's score divisor) rather than being summed down to it."""
    alpha = jnp.asarray(alpha)
    return alpha if alpha.ndim == 0 else jnp.repeat(alpha, self.action_dim)

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

  critic_objective = "td"

  @abstractmethod
  def generate_action(self, obs, *, critic: "Critic" = None, key=None) -> tuple:
    """(action, entropy): a canonical [-1, 1] action, one row per obs row, and
    the differential entropy of whatever distribution produced it, decomposed
    per chunk position -- shaped `(B, chunk_size)`, or `(chunk_size,)` for an
    actor whose "entropy" is a property of a parameter rather than of the draw
    (SDAC's `noise_entropy`). `chunk_size` 1 makes both the same shape (and
    value) the old `(B, 1)`/scalar convention was, so every existing consumer
    is unaffected; a caller wanting one number sums the chunk axis itself
    (`target_entropy_bonus`/`PathwiseActorMixin.loss` do exactly that, weighted
    by whatever shape `alpha` is). A sampler without an entropy estimate
    returns None for entropy."""

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


class PathwiseActorMixin:
  """The pathwise max-entropy actor objective SAC's `StochasticActor` and
  DIME's `DISActor` share once each supplies its own `generate_action`:
  `-(Q + stop_gradient(alpha)*entropy)`, `Q` priced through a *frozen* critic
  so the actor's gradient improves the action, never the critic. The two
  samplers stay separate -- SAC's density-derived entropy and DIME's
  path-density estimate are not interchangeable distributions -- only the
  objective built on top of "an (action, entropy) pair and a critic" is
  identical between them.

  Mix in ahead of `Actor` in the MRO (`class X(PathwiseActorMixin, Actor,
  ...)`) so this concrete `loss` satisfies `Actor.loss`'s abstract method.
  Not for AMDP (score-regression) or SDAC (weighted-denoising/exploration),
  whose objectives are not this one with a different sampler -- leave those
  in their own actors."""

  def loss(self, obs, critic: "Critic", alpha, *, key=None):
    action, entropy = self.generate_action(obs, key=key)
    q = frozen(critic).value(obs, action.reshape(action.shape[0], -1))
    # entropy is (B, chunk_size); alpha is either a scalar (one alpha for the
    # whole chunk) or (chunk_size,) (one per chunk position) -- either way
    # `alpha * entropy` broadcasts, and summing the chunk axis restores the
    # single (B, 1) bonus the old whole-plan entropy already was.
    bonus = (jax.lax.stop_gradient(alpha) * entropy).sum(axis=-1, keepdims=True)
    return {"actor": -(q + bonus).mean()}


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


