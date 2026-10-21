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
from barl_jax.networks.regulator import AutotuneRegulatorConfig
from barl_jax.networks.simba import SimbaConfig
from barl_jax.optimization import ActorOptim, ActorOptimConfig
from barl_jax.policies.rl import MaxEntPolicyConfig, MaxEntRLPolicy
from barl_jax.registration import get_component, register_component, register_policy


@dataclass(frozen=True)
class StochasticActorConfig(ActorOptimConfig):
  class_name: str = "StochasticActor"
  latent_dim: int = 128
  layers: int = 3
  mlp: NetworkConfig = field(default_factory=lambda: SimbaConfig())
  squash_cls: str = "TanhSquash"
  log_std_min: float = -20.0
  log_std_max: float = 2.0


@register_component
class StochasticActor(ActorOptim, Actor):

  kind = "actor"
  config_type = StochasticActorConfig

  def __init__(self, env_stats, config: StochasticActorConfig, *, rngs: nnx.Rngs, **kwargs):
    super().__init__(env_stats, config, rngs=rngs, **kwargs)

    self.squash  = get_component(config.squash_cls)(env_stats, None)

    low, high = env_stats.get("act_low"), env_stats.get("act_high")
    scale = 1.0 if low is None else (np.asarray(high) - np.asarray(low)) / 2
    scale = np.tile(np.asarray(scale, dtype=np.float32).reshape(-1),
                    self.chunk_size).reshape(-1)
    self.log_act_scale = Buffer(jnp.log(jnp.asarray(scale, jnp.float32)))

    latent_dim, layers = config.latent_dim, config.layers

    self.net = config.mlp.build(self.obs_dim, 2 * self.plan_dim, latent_dim, layers, rngs=rngs)

  def dist(self, s):
    out = self.net((s,)).astype(jnp.float32)
    mean, log_std = jnp.split(out, 2, axis=-1)
    std = jnp.exp(jnp.clip(log_std, self.config.log_std_min, self.config.log_std_max))
    return mean, std

  def rsample(self, s, key=None):
    mean, std = self.dist(s)
    key = self.next_key(key)
    x = mean + std * jax.random.normal(key, mean.shape, mean.dtype)
    return self.squash(x), x, (mean, std)

  def generate_action(self, s, *, critic: Critic = None, key=None):
    action_flat, x, (mean, std) = self.rsample(s, key)

    log_prob = (gaussian.log_prob(x, mean, std)
                - self.squash.log_abs_det_jacobian(x, action_flat)
                - self.log_act_scale[...])

    entropy = -self.chunk_sum(log_prob)
    return self.to_plan(action_flat), entropy

  def sample(self, s, *, critic: Critic = None, key=None):
    action_flat, _, _ = self.rsample(s, key)
    return self.to_plan(action_flat)

  
  def loss(self, obs, critic: "Critic", alpha, *, key=None):
  
    action, entropy = self.generate_action(obs, key=key)
    q = frozen(critic).value(obs, action.reshape(action.shape[0], -1))
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
  params: MaxEntPolicyConfig = field(default_factory=lambda: MaxEntPolicyConfig(policy_delay=3))
  components: ComponentsConfig = field(default_factory=lambda: ComponentsConfig(
    actor=StochasticActorConfig(), critic=C51CriticConfig(),
    entropy=AutotuneRegulatorConfig()))
