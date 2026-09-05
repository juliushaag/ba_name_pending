import gymnasium as gym
import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.components import Buffer, frozen
from barl_jax.critics.nclql import (
  NoiseConditionedCritic,
)
from barl_jax.hparam import Config, HParam
from barl_jax.metrics import DEBUG
from barl_jax.networks.schedule import Schedule
from barl_jax.objectives import noise_td_loss
from barl_jax.policy import Policy, register_policy


@register_policy
class NCLQL(Policy):
  """Noise-Conditioned Langevin Q-Learning (Ki et al. 2026).

  Reference implementation: https://github.com/ku-dmlab/NCLQL/tree/main

  Actor-free. There is no policy network, no entropy estimate and no entropy
  dual -- the whole actor + AutotuneRegulator apparatus disappears, because
  grad_a log pi_soft(a|s) = grad_a Q(s,a) means the target score is already
  available from policy evaluation. Learning reduces to fitting one
  noise-conditioned critic; the policy is annealed Langevin dynamics on its
  action-gradient field.

  What is left here is exactly that sampler. Fitting the critic -- the TD loss
  at the clean level and the smoothing loss under it -- belongs to whatever
  `NoiseConditionedCritic` fills the `critic` slot, and `train_step` hands it
  the ladder, a key and the bootstrap action the chain produced. Nothing on
  either side assumes the heads predict a scalar (the chain climbs
  `expectation()`, the losses go through `regression_loss`), so `NCLQLCritic`
  (the reference's scalar ensemble), `NCLQLC51Critic` and `NCLQLHLGaussCritic`
  are interchangeable by `class:` alone.

  Index convention follows the paper -- level 0 == sigma_1 (coarsest),
  level L-1 == sigma_L (clean); annealing runs 0 -> L-1."""

  SLOTS = ("critic", "sigmas")
  SLOT_OBJECTIVES = {"critic": "noise_td"}

  # The annealed-Langevin sampler's knobs, each with its reference's default as
  # the class default; a model YAML's `params:` overrides the ones that differ.
  noise_levels    = Config(10)
  steps_per_level = Config(2)

  # The reference's `step_lr`: the level's step is step_size*(sigma_i/sigma_L)^2,
  # the drift is half of that times `temperature`, and the injected noise is its
  # square root. Tied to the ladder, so retune whenever `sigmas` moves.
  step_size       = HParam(1.0e-4)
  # The paper's `w`, a temperature on Q. It scales the drift whether or not the
  # score is normalized -- normalization replaces the score's *direction*
  # vector, not the step length.
  temperature     = HParam(500.0)
  normalize_score = Config(True)
  clip_actions    = Config(True)
  reward_scale    = HParam(5.0)
  action_limit    = Config(1.0)
  init_scale      = Config(1.0)

  def __init__(self, env: gym.Env, **kwargs):
    super().__init__(env, **kwargs)

    assert isinstance(env.action_space, gym.spaces.Box)

    self.sigmas_sched: Schedule = self.build_component(
      "sigmas", Schedule, config_overrides={"steps": self.noise_levels})

    sigmas = self.sigmas_sched.ladder()
    self.sigmas = Buffer(sigmas)
    self.alpha_ratio = Buffer((sigmas / sigmas[-1]) ** 2)

    stats = self.env_stats
    if stats.get("return_min") is not None:
      stats = stats.derive(return_min=stats["return_min"] * self.reward_scale,
                           return_max=stats["return_max"] * self.reward_scale)

    self.critic: NoiseConditionedCritic = self.build_component(
      "critic", NoiseConditionedCritic,
      env_stats=stats.derive(act_dim=stats["chunked_act_dim"]))

  @property
  def num_levels(self) -> int:
    return int(self.sigmas[...].size)

  def critic_score(self, obs, act, level):
    """grad_a E[Z(s, a, level)], averaged over heads -- `expectation()` and not
    `value()`, since here the gradient field *is* the policy and a pessimism
    term would change the distribution the chain samples from.

    Uses nnx.grad on the frozen critic to avoid trace-level issues when called
    from within a vmapped context (multi-seed rollout)."""
    critic = frozen(self.critic)

    def q_fn(critic, a):
      return critic.expectation(obs, critic.augment(a, level)).sum()

    return nnx.grad(q_fn, argnums=1)(critic, act)

  def sample(self, obs, key):
    key, init_key = jax.random.split(key)
    a = self.init_scale * (2.0 * jax.random.uniform(
      init_key, (obs.shape[0], self.chunk_size * self.action_dim), jnp.float32) - 1.0)

    score_norm = jnp.zeros(())
    keys = jax.random.split(key, self.num_levels * self.steps_per_level)

    alpha_ratio = self.alpha_ratio[...]
    for i in range(self.num_levels):
      alpha = self.step_size * alpha_ratio[i]
      for t in range(self.steps_per_level):
        g = self.critic_score(obs, a, i)
        score_norm = jnp.linalg.norm(g, axis=-1).mean()

        if self.normalize_score:
          g = g / (jnp.linalg.norm(g, axis=-1, keepdims=True) + 1e-8)
        drift = 0.5 * alpha * self.temperature * g

        noise = jax.random.normal(keys[i * self.steps_per_level + t], a.shape, a.dtype)
        a = a + drift + jnp.sqrt(alpha) * noise

        # Per level, because the ladder spans four orders of magnitude in `alpha`
        # and the two terms scale differently under it: `normalize_score` fixes
        # |g| at 1, so the drift is 0.5*alpha*temperature outright while the
        # noise is sqrt(alpha). Whether either is comparable to `action_limit`
        # is the difference between a chain that anneals and one that spends its
        # first levels slamming into the clip.
        self.metrics.scalar(f"chain/drift_norm_l{i}",
                            jnp.linalg.norm(drift, axis=-1).mean(), tier=DEBUG)
        self.metrics.scalar(f"chain/noise_scale_l{i}", jnp.sqrt(alpha), tier=DEBUG)

        if self.clip_actions:
          clipped = (jnp.abs(a) > self.action_limit).astype(jnp.float32).mean()
          self.metrics.scalar(f"chain/clipped_frac_l{i}", clipped, tier=DEBUG)
          a = jnp.clip(a, -self.action_limit, self.action_limit)

    return a, score_norm

  @property
  def discount(self):
    return self.critic.gamma

  def train_step(self, batch):
    obs      = self.normalize_observations(batch["obs"])
    next_obs = self.normalize_observations(batch["next_obs"])
    action   = batch["act"]
    term     = batch["term"]
    reward   = batch["rew"] * self.reward_scale
    horizon  = batch.get("horizon")

    losses = {}
    keys = jax.random.split(self.rngs.noise(), 2)

    next_action, score_norm = self.sample(next_obs, keys[0])
    next_action = jax.lax.stop_gradient(next_action)

    noise_td_loss(self.critic, obs, action, next_obs, next_action, reward, term,
                     self.sigmas[...], keys[1], losses, horizon=horizon)

    self.metrics.vec("action", next_action.reshape(
      next_action.shape[0], self.chunk_size, self.action_dim), limit=1.0)
    self.metrics.scalar("score_norm", score_norm)

    return losses

  def inference_step(self, observation, step_info=None, key=None):
    obs = self.normalize_observations(observation)
    key = self.rngs.noise() if key is None else key
    action, _ = self.sample(obs, key)
    return self.action_chunk(
      action.reshape(action.shape[0], self.chunk_size, self.action_dim)), {}
