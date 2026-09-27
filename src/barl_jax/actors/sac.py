from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from barl_jax.components import Actor, Buffer, Critic
from barl_jax.config.schemas import ComponentsConfig, ModelConfig, register_model
from barl_jax.critics.categorical import C51CriticConfig
from barl_jax.networks import gaussian
from barl_jax.networks.base import NetworkConfig
from barl_jax.networks.mlp import MLPConfig
from barl_jax.networks.regulator import AutotuneRegulatorConfig
from barl_jax.optimization import ActorOptim, ActorOptimConfig
from barl_jax.policies.rl import MaxEntPolicyConfig, MaxEntRLPolicy
from barl_jax.registration import get_component, register_component, register_policy


@dataclass(frozen=True)
class StochasticActorConfig(ActorOptimConfig):
  class_name: str = "StochasticActor"
  latent_dim: int = 512
  layers: int = 3
  mlp: NetworkConfig = field(default_factory=lambda: MLPConfig(
    activation="silu", norm="layer", norm_kwargs={"epsilon": 1e-5}, dtype="bfloat16"))
  squash_cls: str = "TanhSquash"
  log_std_min: float = -20.0
  log_std_max: float = 2.0


@register_component
class StochasticActor(ActorOptim, Actor):
  """Diagonal-Normal MLP actor used by SAC. `dist()` stays an
  unconstrained Normal; sample() squashes it (the `squash_cls` Config
  selects which -- any no-arg Squash works generically here).

  The shallow Gaussian policy has just a state-independent `log_std` plus the
  std bounds. The MLP uses SiLU and LayerNorm on hidden blocks, with no input
  or stem normalization; its settings are grouped under `mlp` below.

  `log_std` is a state-independent parameter rather than a network head --
  barl's deliberate deviation from canonical SAC, kept as-is."""

  kind = "actor"
  config_type = StochasticActorConfig

  def __init__(self, env_stats, config: StochasticActorConfig, *, rngs: nnx.Rngs, **kwargs):
    super().__init__(env_stats, config, rngs=rngs, **kwargs)

    self.log_std = nnx.Param(jnp.zeros((self.plan_dim,), jnp.float32))
    self.squash  = get_component(config.squash_cls)(env_stats, None)

    low, high = env_stats.get("act_low"), env_stats.get("act_high")
    scale = 1.0 if low is None else (np.asarray(high) - np.asarray(low)) / 2
    scale = np.tile(np.asarray(scale, dtype=np.float32).reshape(-1),
                    self.chunk_size).reshape(-1)
    self.log_act_scale = Buffer(jnp.log(jnp.asarray(scale, jnp.float32)))

    latent_dim, layers = config.latent_dim, config.layers

    self.net = config.mlp.build(self.obs_dim, self.plan_dim, latent_dim, layers, rngs=rngs)

  def dist(self, s):
    """(mean, std) of the pre-squash Normal, over the flat plan."""
    mean = self.net(s).astype(jnp.float32)
    std  = jnp.exp(jnp.clip(self.log_std[...], self.config.log_std_min, self.config.log_std_max))
    return mean, std

  def rsample(self, s, key=None):
    """Reparameterized (action, pre-squash x, dist) -- differentiable through s."""
    mean, std = self.dist(s)
    key = self.next_key(key)
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
    return self.to_plan(action_flat), entropy

  def sample(self, s, *, critic: Critic = None, key=None):
    action_flat, _, _ = self.rsample(s, key)
    return self.to_plan(action_flat)


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
