"""Annealed Langevin action sampling from a noise-conditioned critic."""
import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.components import ActionSampler, frozen
from barl_jax.hparam import Config, HParam
from barl_jax.metrics import DEBUG
from barl_jax.registration import register_component


@register_component
class LangevinActor(ActionSampler):
  """No policy network or optimizer: climb the live critic's action gradients.

  The critic owns the noise ladder used by both sampling and smoothing loss.
  Score norms are diagnostics, never an entropy estimate.
  """

  POLICY = {
    "policy": "RLPolicy",
    "components": {
      "actor": {
        "class": "LangevinActor",
        "steps_per_level": 2,
        "step_size": 0.002,
        "temperature": 500.0,
        "normalize_score": True,
        "clip_actions": True,
        "action_limit": 1.0,
        "init_scale": 1.0
      },
      "critic": {
        "class": "NCLQLC51Critic",
        "tier": "xlarge",
        "reward_scale": 1.0,
        "noise_schedule": {
          "class": "ExponentialSchedule",
          "start": 0.1,
          "end": 0.001,
          "steps": 10
        }
      }
    }
  }
  kind = "sampler"
  critic_objective = "noise_td"
  steps_per_level = Config(2)
  step_size = HParam(2.0e-3)
  temperature = HParam(500.0)
  normalize_score = Config(True)
  clip_actions = Config(True)
  action_limit = Config(1.0)
  init_scale = Config(1.0)

  def __init__(self, env_stats, config, *, rngs, **kwargs):
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
      return critic.expectation(obs, critic.augment(a, level)).sum()

    return nnx.grad(q_fn, argnums=1)(critic, act)

  def generate_action(self, obs, *, critic=None, key=None):
    if critic is None:
      raise ValueError("LangevinActor requires a noise-conditioned critic")
    key = self.rngs.noise() if key is None else key
    sigmas = critic.sigmas[...]
    num_levels = int(sigmas.size)
    key, init_key = jax.random.split(key)
    a = self.init_scale * (2.0 * jax.random.uniform(
      init_key, (obs.shape[0], self.chunk_size * self.action_dim), jnp.float32) - 1.0)

    score_norm = jnp.zeros(())
    keys = jax.random.split(key, num_levels * self.steps_per_level)

    # Relative spacing only -- normalized to 1.0 at the coarsest level so `step_size`
    # sets the actor's absolute step scale directly, independent of whatever
    # absolute sigma units the critic's noise_schedule happens to use. Before this
    # normalization, alpha_ratio[0] was (sigma_0/sigma_last)**2, which spans four
    # orders of magnitude for the default ladder (0.1 -> 0.001) and drove the
    # coarsest drift to ~250 against an action_limit of 1.0 -- a chain that spends
    # its first levels slamming into the clip instead of annealing.
    alpha_ratio = (sigmas / sigmas[-1]) ** 2
    alpha_ratio = alpha_ratio / alpha_ratio[0]
    for i in range(num_levels):
      alpha = self.step_size * alpha_ratio[i]
      for t in range(self.steps_per_level):
        g = self.critic_score(critic, obs, a, i)
        score_norm = jnp.linalg.norm(g, axis=-1).mean()

        if self.normalize_score:
          g = g / (jnp.linalg.norm(g, axis=-1, keepdims=True) + 1e-8)
        drift = 0.5 * alpha * self.temperature * g

        noise = jax.random.normal(keys[i * self.steps_per_level + t], a.shape, a.dtype)
        a = a + drift + jnp.sqrt(alpha) * noise

        # `normalize_score` fixes |g| at 1, so the drift is 0.5*alpha*temperature
        # outright while the noise is sqrt(alpha) -- whether either is comparable
        # to `action_limit` is the difference between a chain that anneals and one
        # that spends its first levels slamming into the clip.
        self.metrics.scalar(f"chain/drift_norm_l{i}",
                            jnp.linalg.norm(drift, axis=-1).mean(), tier=DEBUG)
        self.metrics.scalar(f"chain/noise_scale_l{i}", jnp.sqrt(alpha), tier=DEBUG)

        if self.clip_actions:
          clipped = (jnp.abs(a) > self.action_limit).astype(jnp.float32).mean()
          self.metrics.scalar(f"chain/clipped_frac_l{i}", clipped, tier=DEBUG)
          a = jnp.clip(a, -self.action_limit, self.action_limit)

    self.metrics.scalar("score_norm", score_norm)
    return a.reshape(a.shape[0], self.chunk_size, self.action_dim), None

