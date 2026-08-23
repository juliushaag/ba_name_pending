import gymnasium as gym
import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.networks.critic import C51Critic, EnsembleCritic, HLGaussCritic
from barl_jax.networks.flow import TimeEmbedding

from barl_jax.networks.registry import register_component
from barl_jax.policy import Buffer, Policy, frozen, register_policy
from barl_jax.networks.schedule import Schedule


class NoiseConditionedCritic:
  """Mixin: makes any `QCriticBase` noise-conditioned, by widening the action it
  is built over with a learned embedding of the level index.

  Level conditioning is orthogonal to how a critic represents a return, so it
  lives here rather than in a critic class of its own: the inner critic is built
  against an `EnvStats` whose `act_dim` already includes `time_dim`, and
  `augment()` is what turns `(a, level)` into the action it expects. Everything
  underneath -- the ensemble reduction, the Bellman target, the regression loss
  -- is the base critic's, unchanged, which is what lets NC-LQL swap a scalar
  ensemble for a categorical one with a one-word `class:` edit.

  `noise_conditioned_loss()` is the whole objective that fits such a critic, and
  it is here rather than in a Policy's train_step for the same reason a target
  net's Polyak update is in `QCriticBase`: TD at the clean level plus smoothing
  of the noisier ones is what *this architecture* has to be trained by, whatever
  algorithm is sampling from it. What stays with the Policy is the part that is
  genuinely algorithmic -- the annealed Langevin chain.

  The embedding is inside the critic, not on the Policy, so a Polyak target
  copies it too: the bootstrap is conditioned on a level, and an embedding that
  tracked the live weights would leak straight back into the target's value."""

  def __init__(self, env_stats, *args, smoothing_coef: float, time_dim: int,
               time_embed: str, time_kwargs: dict, rngs: nnx.Rngs, **kwargs):

    self.time_dim       = int(time_dim)
    self.smoothing_coef = float(smoothing_coef)
    # `time_kwargs` used to stop here: the embedding was built with none of them,
    # so its frequency ladder sat at TimeEmbedding's own Python defaults.
    self.time_embed = TimeEmbedding(time_dim=self.time_dim, kind=time_embed,
                                    rngs=rngs, **time_kwargs)

    super().__init__(env_stats.derive(act_dim=env_stats["act_dim"] + self.time_dim),
                     *args, rngs=rngs, **kwargs)

    # The derived stats are for the *network's* input width; `action_dim` stays
    # the env's, as it means everywhere else.
    self.action_dim = env_stats["act_dim"]

  def target_modules(self) -> dict:
    return {**super().target_modules(), "time_embed": self.time_embed}

  def augment(self, act, level, target: bool = False):
    """`(a, level)` -> the action the inner critic was built over. `level` is
    either one index for the whole batch or one per sample."""
    if not isinstance(level, jax.Array):
      level = jnp.full((act.shape[0],), level, jnp.int32)

    embed = self.time_embed_target if target and self.has_target else self.time_embed
    return jnp.concatenate([act, embed(level.astype(jnp.float32)[:, None])], axis=-1)

  # ---- objective ----------------------------------------------------------

  def noise_conditioned_loss(self, obs, action, next_obs, next_action, reward, term,
                             sigmas, key, losses: dict, shift=0.0) -> None:
    """`td_loss` at the clean level, plus a smoothing loss pulling every noisier
    level onto it -- one `losses["q"]`, since both fit the same heads.

    `sigmas` is passed in rather than owned here because it is the same ladder
    the owning Policy's sampler anneals over; index L-1 is the clean level, by
    that Policy's convention. `next_action` likewise comes from the caller: how
    the bootstrap action is chosen is the algorithm's business, and here it is
    a Langevin chain run on the live critic -- only the value it is scored with
    comes from the target."""
    clean_level = sigmas.shape[0] - 1

    level_key, noise_key = jax.random.split(key)
    levels = jax.random.randint(level_key, (action.shape[0],), 0, clean_level)
    noisy  = action + sigmas[levels][:, None] * jax.random.normal(
      noise_key, action.shape, action.dtype)

    pred_clean, pred_noisy = self.raw_batches(
      (obs, self.augment(action, clean_level)),
      (obs, self.augment(noisy, levels)),
    )

    next_raw  = self.target_raw(next_obs, self.augment(next_action, clean_level, target=True))
    td_target = self.bellman_target(reward, term, self.bootstrap_from_raw(next_raw), shift)

    td_loss = self.regression_loss(pred_clean, td_target)

    # In whatever the critic's target representation is -- a detached scalar, or
    # a detached distribution over atoms.
    smoothing_loss = self.regression_loss(pred_noisy, self.target_from_raw(pred_clean))

    losses["q"] = td_loss + self.smoothing_coef * smoothing_loss

    # Once, and against the clean pass: recording both would have the second
    # overwrite the first's `critic/q/*`.
    self.record(pred_clean, td_target)
    self.metrics.scalar("td_loss", td_loss)
    self.metrics.scalar("smoothing_loss", smoothing_loss)


@register_component
class NCLQLCritic(NoiseConditionedCritic, EnsembleCritic):
  """The reference's critic (ku-dmlab/NCLQL): two scalar heads with a Polyak
  target, `min` over the heads for the bootstrap and `mean` for the Langevin
  score.

  Both reductions are `EnsembleCritic`'s, not special-cased: at `n_models: 2`
  the population std is |q1 - q2|/2, so `rho: 1.0` makes `reduce_value` exactly
  the reference's `min(target_q1, target_q2)`, while `expectation()` -- what the
  chain climbs -- averages the heads by construction. The target net is the one
  genuine departure from the rest of this codebase, and it is NC-LQL's: with no
  actor, the bootstrap action is whatever annealed Langevin lands on in the
  critic's own gradient field, so nothing else keeps it on-distribution."""


@register_component
class NCLQLC51Critic(NoiseConditionedCritic, C51Critic):
  """NC-LQL over a categorical return distribution instead of a scalar.

  The Langevin chain climbs E[Z] either way, so the algorithm is untouched; what
  changes is that the spread the bootstrap's `min` is pessimistic over becomes
  the return's own rather than the heads' disagreement. Note that the support is
  scaled by `reward_scale` for you -- see NCLQL.__init__."""


@register_component
class NCLQLHLGaussCritic(NoiseConditionedCritic, HLGaussCritic):
  """NC-LQL fitting its scalar Bellman target as a Gaussian-smoothed histogram
  (Farebrother et al., 2024) rather than by MSE -- the classification-as-
  regression half of C51's story without the distributional bootstrap."""


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


  def __init__(self, env: gym.Env, **kwargs):
    super().__init__(env, **kwargs)

    assert isinstance(env.action_space, gym.spaces.Box)

    params = self.resolve_params()

    self.steps_per_level = params["steps_per_level"]
    self.noise_levels    = float(params["noise_levels"])
    self.step_size       = float(params["step_size"])
    self.temperature     = float(params["temperature"])
    self.normalize_score = params["normalize_score"]
    self.reward_scale    = float(params["reward_scale"])
    self.clip_actions    = params["clip_actions"]
    self.action_limit    = params["action_limit"]
    self.init_scale      = params["init_scale"]


    self.sigmas_sched: Schedule = self.build_component("sigmas", Schedule, steps=self.noise_levels)

    sigmas = self.sigmas_sched.ladder()
    self.sigmas = Buffer(sigmas)
    self.alpha_ratio = Buffer((sigmas / sigmas[-1]) ** 2)

    stats = self.env_stats
    if stats.get("return_min") is not None:
      stats = stats.derive(return_min=stats["return_min"] * self.reward_scale,
                           return_max=stats["return_max"] * self.reward_scale)

    self.critic: NoiseConditionedCritic = self.build_component(
      "critic", NoiseConditionedCritic, env_stats=stats)

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
      init_key, (obs.shape[0], self.action_dim), jnp.float32) - 1.0)

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

        if self.clip_actions:
          a = jnp.clip(a, -self.action_limit, self.action_limit)

    return a, score_norm

  def train_step(self, batch):
    obs      = self.normalize_obs(batch["obs"])
    next_obs = self.normalize_obs(batch["next_obs"])
    action   = batch["act"]
    term     = batch["term"]
    reward   = batch["rew"] * self.reward_scale

    losses = {}
    keys = jax.random.split(self.rngs.noise(), 2)

    next_action, score_norm = self.sample(next_obs, keys[0])
    next_action = jax.lax.stop_gradient(next_action)

    self.critic.noise_conditioned_loss(obs, action, next_obs, next_action, reward, term,
                                       self.sigmas[...], keys[1], losses)

    self.metrics.vec("action", next_action, limit=1.0)
    self.metrics.scalar("score_norm", score_norm)

    return losses

  def inference_step(self, observation, step_info=None, key=None):
    obs = self.normalize_obs(observation)
    key = self.rngs.noise() if key is None else key
    action, _ = self.sample(obs, key)
    return action, {}
