from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from barl_jax.components import Actor, Buffer, Critic, frozen
from barl_jax.config.schemas import ComponentsConfig, ModelConfig, register_model
from barl_jax.critics.categorical import C51CriticConfig
from barl_jax.networks import gaussian
from barl_jax.networks.base import NetworkConfig
from barl_jax.networks.mlp import MLPConfig
from barl_jax.networks.regulator import AutotuneRegulatorConfig
from barl_jax.optimization import ActorOptim
from barl_jax.policies.rl import MaxEntPolicyConfig, MaxEntRLPolicy
from barl_jax.registration import get_component, register_component, register_policy


@dataclass(frozen=True)
class StochasticActorConfig:
  class_name: str = "StochasticActor"
  latent_dim: int = 256
  layers: int = 2
  mlp: NetworkConfig = field(default_factory=lambda: MLPConfig(
    activation="silu", norm="layer", norm_kwargs={"epsilon": 1e-5}, dtype="bfloat16"))
  squash_cls: str = "TanhSquash"
  log_std_min: float = -20.0
  log_std_max: float = 2.0
  log_prob_clip: float = 0.99
  lr: float = 3.0e-4
  adam_b1: float = 0.9
  grad_clip: float = 1.0
  grad_clip_mode: str = "element"


@register_component
class StochasticActor(ActorOptim, Actor):
  """Diagonal-Normal MLP actor shared by SAC/PPO/IQL. `dist()` stays an
  unconstrained Normal; sample()/mode() squash it (the `squash_cls` Config
  selects which -- any no-arg Squash works generically here).

  The shallow Gaussian policy has just a state-independent `log_std` plus the
  std bounds. The MLP uses SiLU and LayerNorm on hidden blocks, with no input
  or stem normalization; its settings are grouped under `mlp` below.

  `log_std` is a state-independent parameter rather than a network head --
  barl's deliberate deviation from canonical SAC, kept as-is."""

  kind = "actor"
  config_type = StochasticActorConfig

  def __init__(self, env_stats, config: StochasticActorConfig, *, rngs: nnx.Rngs, **kwargs):
    super().__init__(env_stats, config, **kwargs)

    self.log_std = nnx.Param(jnp.zeros((self.plan_dim,), jnp.float32))
    self.squash  = get_component(config.squash_cls)(env_stats, None)

    low, high = env_stats.get("act_low"), env_stats.get("act_high")
    scale = 1.0 if low is None else (np.asarray(high) - np.asarray(low)) / 2
    scale = np.tile(np.asarray(scale, dtype=np.float32).reshape(-1),
                    self.chunk_size).reshape(-1)
    self.log_act_scale = Buffer(jnp.log(jnp.asarray(scale, jnp.float32)))

    latent_dim, layers = config.latent_dim, config.layers

    self.net = config.mlp.build(self.obs_dim, self.plan_dim, latent_dim, layers, rngs=rngs)
    self.rngs = rngs

  def _plan(self, flat):
    return flat.reshape(flat.shape[0], self.chunk_size, self.action_dim)

  def dist(self, s):
    """(mean, std) of the pre-squash Normal, over the flat plan."""
    mean = self.net(s).astype(jnp.float32)
    std  = jnp.exp(jnp.clip(self.log_std[...], self.config.log_std_min, self.config.log_std_max))
    return mean, std

  def log_prob(self, dist, action):
    """`action` is the squashed (canonical [-1, 1]) value; inverts the squash
    and applies the change-of-variables correction."""
    mean, std = dist
    flat = action.reshape(action.shape[0], -1)
    clip = self.config.log_prob_clip
    x = self.squash.inverse(jnp.clip(flat, -clip, clip))
    lp = gaussian.log_prob(x, mean, std) - self.squash.log_abs_det_jacobian(x, flat)
    return lp.sum(axis=-1)

  def rsample(self, s, key=None):
    """Reparameterized (action, pre-squash x, dist) -- differentiable through s."""
    mean, std = self.dist(s)
    key = self.rngs.noise() if key is None else key
    x = mean + std * jax.random.normal(key, mean.shape, mean.dtype)
    return self.squash(x), x, (mean, std)

  def generate_action(self, s, *, critic: Critic = None, key=None):
    """Reparameterized (action, entropy), differentiable through `s`. The squash
    Jacobian correction needs the extra `act_scale` term since log_prob is the
    density of an affine-transformed variable, not of `action` itself.
    `critic` is unused -- SAC's actor needs none to sample -- and accepted only
    to satisfy the shared `Actor.generate_action` contract."""
    action_flat, x, (mean, std) = self.rsample(s, key)

    log_prob = (gaussian.log_prob(x, mean, std)
                - self.squash.log_abs_det_jacobian(x, action_flat)
                - self.log_act_scale[...])
    # Per-chunk-position entropy, (B, chunk_size): summing within each
    # position's action_dim block rather than over the whole flattened plan --
    # the same shape (and at chunk_size 1 the same value) as the old (B, 1)
    # joint entropy, so a scalar alpha reads the chunk-summed total unchanged.
    entropy = -self.chunk_sum(log_prob)
    return self._plan(action_flat), entropy

  def sample(self, s, *, critic: Critic = None, key=None):
    action_flat, _, _ = self.rsample(s, key)
    return self._plan(action_flat)

  def mode(self, s):
    """Deterministic (mean) action -- IQL trains via AWR but deploys this."""
    mean, _ = self.dist(s)
    return self._plan(self.squash(mean))

  def loss(self, obs, critic: "Critic", alpha, *, key=None):
    action, entropy = self.generate_action(obs, key=key)
    q = frozen(critic).value(obs, action.reshape(action.shape[0], -1))
    # entropy is (B, chunk_size); alpha is a scalar or (chunk_size,), so
    # `alpha * entropy` broadcasts and the chunk sum gives one (B, 1) bonus.
    bonus = (jax.lax.stop_gradient(alpha) * entropy).sum(axis=-1, keepdims=True)
    return {"actor": -(q + bonus).mean()}

@register_policy
class SAC(MaxEntRLPolicy):
  """Binds StochasticActor into the shared max-entropy actor-critic template --
  the actor-policy integration for this algorithm. `actor_cls` tightens
  `build_component`'s gate from the generic `ActionSampler` ABC to this
  concrete class, so a `components.actor.class` mismatch fails at construction
  with a clear error instead of resolving to whatever the config named."""

  actor_cls = StochasticActor


@register_model
@dataclass(frozen=True)
class SACModelConfig(ModelConfig):
  policy_class: str = "SAC"
  params: MaxEntPolicyConfig = field(default_factory=MaxEntPolicyConfig)
  components: ComponentsConfig = field(default_factory=lambda: ComponentsConfig(
    actor=StochasticActorConfig(), critic=C51CriticConfig(),
    entropy=AutotuneRegulatorConfig()))
