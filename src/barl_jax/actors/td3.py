from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.components import Actor, Critic, frozen
from barl_jax.config.schemas import ComponentsConfig, ModelConfig, register_model
from barl_jax.critics.base import TargetConfig
from barl_jax.critics.direct import EnsembleCriticConfig
from barl_jax.networks.base import NetworkConfig
from barl_jax.networks.mlp import MLPConfig
from barl_jax.optimization import ActorOptim, ActorOptimConfig
from barl_jax.policies.rl import PolicyConfig, RLPolicy
from barl_jax.registration import register_component, register_policy


@dataclass(frozen=True)
class DeterministicActorConfig(ActorOptimConfig):
  class_name: str = "DeterministicActor"
  latent_dim: int = 512
  layers: int = 3
  mlp: NetworkConfig = field(default_factory=lambda: MLPConfig(
    activation="relu", norm="none", dtype="float32"))
  explore_noise: float = 0.1
  target_noise: float = 0.2
  target_noise_clip: float = 0.5


@register_component
class DeterministicActor(ActorOptim, Actor):
  """TD3's tanh-MLP policy (Fujimoto et al., 2018). `sample` is the played
  action, mean plus Gaussian exploration noise; `generate_action` feeds the
  Bellman bootstrap, mean plus clipped target-policy smoothing noise; `loss`
  ascends Q at the noiseless mean. There is no entropy, so `generate_action`
  reports None."""

  kind = "actor"
  config_type = DeterministicActorConfig

  def __init__(self, env_stats, config: DeterministicActorConfig, *, rngs: nnx.Rngs, **kwargs):
    super().__init__(env_stats, config, rngs=rngs, **kwargs)
    self.net = config.mlp.build(self.obs_dim, self.plan_dim, config.latent_dim, config.layers,
                                rngs=rngs)

  def mean(self, s):
    return jnp.tanh(self.net(s).astype(jnp.float32))

  def noisy(self, s, key, std, clip=None):
    mean = self.mean(s)
    noise = std * jax.random.normal(self.next_key(key), mean.shape, mean.dtype)
    if clip is not None:
      noise = jnp.clip(noise, -clip, clip)
    return self.to_plan(jnp.clip(mean + noise, -1.0, 1.0))

  def generate_action(self, s, *, critic: Critic = None, key=None):
    cfg = self.config
    return self.noisy(s, key, cfg.target_noise, cfg.target_noise_clip), None

  def sample(self, s, *, critic: Critic = None, key=None):
    return self.noisy(s, key, self.config.explore_noise)

  def loss(self, obs, critic: Critic, alpha, *, key=None):
    q = frozen(critic).value(obs, self.mean(obs))
    return {"actor": -q.mean()}


@register_policy
class TD3(RLPolicy):
  """DeterministicActor on the shared actor-critic template with clipped double
  Q and delayed actor updates. The bootstrap action comes from the online actor
  (no target policy network), as in every other algorithm here."""

  actor_cls = DeterministicActor


@register_model
@dataclass(frozen=True)
class TD3ModelConfig(ModelConfig):
  policy_class: str = "TD3"
  params: PolicyConfig = field(default_factory=lambda: PolicyConfig(policy_delay=2))
  components: ComponentsConfig = field(default_factory=lambda: ComponentsConfig(
    actor=DeterministicActorConfig(),
    critic=EnsembleCriticConfig(
      latent_dim=256, layers=1, target=TargetConfig(kind="polyak", joint_forward=False),
      mlp=MLPConfig(activation="relu", norm="none", dtype="float32")),
    entropy=None))
