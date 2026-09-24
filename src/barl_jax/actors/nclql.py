"""Annealed Langevin action sampling from a noise-conditioned critic."""
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.components import ActionSampler, frozen
from barl_jax.config.schemas import ComponentsConfig, ModelConfig, register_model
from barl_jax.critics.categorical import C51CriticConfig
from barl_jax.critics.presets import nclql
from barl_jax.metrics import DEBUG
from barl_jax.policies.rl import PolicyConfig, RLPolicy
from barl_jax.registration import register_component, register_policy


@dataclass(frozen=True)
class LangevinActorConfig:
  class_name: str = "LangevinActor"
  steps_per_level: int = 2
  step_lr: float = 2.0e-3
  temperature: float = 500.0
  normalize_score: bool = True
  clip_actions: bool = True
  action_limit: float = 1.0
  init_scale: float = 1.0


@register_component
class LangevinActor(ActionSampler):
  """No policy network or optimizer: climb the live critic's action gradients.

  The critic owns the noise ladder used by both sampling and smoothing loss.
  Score norms are diagnostics, never an entropy estimate.
  """

  config_type = LangevinActorConfig
  kind = "sampler"
  critic_objective = "noise_td"

  def __init__(self, env_stats, config: LangevinActorConfig, *, rngs, **kwargs):
    super().__init__(env_stats, config, **kwargs)
    self.rngs = rngs

  def critic_score(self, critic, obs, act, level):
    """grad_a E[Z(s, a, level)], averaged over heads -- `expectation()` and not
    `value()`, since here the gradient field *is* the policy and a pessimism
    term would change the distribution the chain samples from.

    Uses nnx.grad on the frozen critic to avoid trace-level issues when called
    from within a vmapped context (multi-seed rollout)."""
    critic = frozen(critic)

    def q_fn(critic, a):
      return critic.expectation(obs, a, level=level).sum()

    return nnx.grad(q_fn, argnums=1)(critic, act)

  def generate_action(self, obs, *, critic=None, key=None):
    if critic is None:
      raise ValueError("LangevinActor requires a noise-conditioned critic")
    key = self.rngs.noise() if key is None else key
    sigmas = critic.noise.sigmas[...]
    num_levels = int(sigmas.size)
    key, init_key = jax.random.split(key)
    a = self.config.init_scale * (2.0 * jax.random.uniform(
      init_key, (obs.shape[0], self.chunk_size * self.action_dim), jnp.float32) - 1.0)

    score_norm = jnp.zeros(())
    keys = jax.random.split(key, num_levels * self.config.steps_per_level)

    # Relative spacing normalized to 1 at the coarsest level, so `step_lr` is that
    # level's absolute step independent of the ladder's sigma units. Unnormalized,
    # the default ladder puts the coarsest drift at ~250 against action_limit 1.
    alpha_ratio = (sigmas / sigmas[-1]) ** 2
    alpha_ratio = alpha_ratio / alpha_ratio[0]
    for i in range(num_levels):
      alpha = self.config.step_lr * alpha_ratio[i]
      for t in range(self.config.steps_per_level):
        g = self.critic_score(critic, obs, a, i)
        score_norm = jnp.linalg.norm(g, axis=-1).mean()

        if self.config.normalize_score:
          g = g / (jnp.linalg.norm(g, axis=-1, keepdims=True) + 1e-8)
        drift = 0.5 * alpha * self.config.temperature * g

        noise = jax.random.normal(keys[i * self.config.steps_per_level + t], a.shape, a.dtype)
        a = a + drift + jnp.sqrt(alpha) * noise

        self.metrics.scalar(f"chain/drift_norm_l{i}",
                            jnp.linalg.norm(drift, axis=-1).mean(), tier=DEBUG)
        self.metrics.scalar(f"chain/noise_scale_l{i}", jnp.sqrt(alpha), tier=DEBUG)

        if self.config.clip_actions:
          clipped = (jnp.abs(a) > self.config.action_limit).astype(jnp.float32).mean()
          self.metrics.scalar(f"chain/clipped_frac_l{i}", clipped, tier=DEBUG)
          a = jnp.clip(a, -self.config.action_limit, self.config.action_limit)

    self.metrics.scalar("score_norm", score_norm)
    return a.reshape(a.shape[0], self.chunk_size, self.action_dim), None


@register_policy
class NCLQL(RLPolicy):
  """Binds LangevinActor into the shared actor-critic template -- the
  actor-policy integration for this algorithm. No entropy dual: the sampler's
  "policy" is the noise-conditioned critic's own gradient field, so this pins
  `RLPolicy` directly rather than `MaxEntRLPolicy`."""

  actor_cls = LangevinActor


@register_model
@dataclass(frozen=True)
class NCLQLModelConfig(ModelConfig):
  policy_class: str = "NCLQL"
  params: PolicyConfig = field(default_factory=PolicyConfig)
  components: ComponentsConfig = field(default_factory=lambda: ComponentsConfig(
    actor=LangevinActorConfig(),
    critic=nclql(C51CriticConfig()), entropy=None))

