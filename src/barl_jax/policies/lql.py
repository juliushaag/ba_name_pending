import gymnasium as gym
import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.networks.critic import QCriticBase
from barl_jax.policy import Buffer, Policy, register_policy
from barl_jax.schedule import Schedule


@register_policy
class NCLQL(Policy):
  """Noise-Conditioned Langevin Q-Learning (Ki et al. 2026).

  Actor-free. There is no policy network, no entropy estimate and no entropy
  dual -- the whole actor + AutotuneRegulator apparatus disappears, because
  grad_a log pi_soft(a|s) = grad_a Q(s,a) means the target score is already
  available from policy evaluation. Learning reduces to fitting one
  noise-conditioned critic; the policy is annealed Langevin dynamics on its
  action-gradient field.

  The `critic` slot is an ordinary QCriticBase: level conditioning is folded
  into its action input by `augment_action`, so it never learns that levels
  exist and any ensemble/quantile/categorical critic can fill the slot. Index
  convention follows the paper -- level 0 == sigma_1 (coarsest), level L-1 ==
  sigma_L (clean); annealing runs 0 -> L-1."""

  # The level embedding is trained by the critic's optimizer -- it is part of
  # the critic's input, but the Policy owns it because it is what makes the
  # critic noise-conditioned.
  policy_params = {"level_embed": "q"}

  def __init__(self, env: gym.Env, **kwargs):
    super().__init__(env, **kwargs)

    assert isinstance(env.action_space, gym.spaces.Box)

    params = self.resolve_params()
    embed_dim = params["level_embed_dim"]

    self.steps_per_level = params["steps_per_level"]
    self.step_size       = float(params["step_size"])
    self.temperature     = float(params["temperature"])
    self.normalize_score = params["normalize_score"]
    self.drift_scale     = float(params["drift_scale"])
    self.clip_actions    = params["clip_actions"]
    self.action_limit    = params["action_limit"]
    self.init_scale      = params["init_scale"]

    # The ladder sigma_1 > ... > sigma_L. The paper reports only its two
    # endpoints (Table 6), so the spacing is the `sigmas` slot's `class:` --
    # ExponentialSchedule is the geometric Song & Ermon (2019) default.
    self.sigmas_sched: Schedule = self.build_component("sigmas", Schedule)

    sigmas = self.sigmas_sched.ladder()
    self.sigmas = Buffer(sigmas)
    # alpha_i / eps = sigma_i^2 / sigma_L^2   (Definition 2)
    self.alpha_ratio = Buffer((sigmas / sigmas[-1]) ** 2)

    self.level_embed = nnx.Embed(int(sigmas.size), embed_dim,
                                 embedding_init=nnx.initializers.normal(0.02),
                                 rngs=self.rngs)

    self.critic: QCriticBase = self.build_component(
      "critic", QCriticBase,
      env_stats=self.env_stats.derive(act_dim=self.action_dim + embed_dim))

  @property
  def num_levels(self) -> int:
    return int(self.sigmas[...].size)

  @property
  def clean_level(self) -> int:
    return self.num_levels - 1

  def augment_action(self, act, level):
    if not isinstance(level, jax.Array):
      level = jnp.full((act.shape[0],), level, jnp.int32)
    return jnp.concatenate([act, self.level_embed(level)], axis=-1)

  def critic_score(self, obs, act, level):
    """expectation(), not value(): the Langevin chain *is* the policy here, so a
    pessimistic bound would sample from a different distribution than exp(wQ)/Z
    rather than merely shrink a value estimate.

    nnx.grad rather than jax.grad, and the critic passed explicitly rather than
    captured: an ensemble critic runs nnx.vmap internally, and a raw jax.grad
    over a closure holding the module makes that vmap re-extract it at a
    different trace level ("Cannot extract graph node from different trace
    level")."""
    def q(critic, a):
      return critic.expectation(obs, self.augment_action(a, level)).sum()
    return nnx.grad(q, argnums=1)(self.critic, act)

  def sample(self, obs, key):
    """Annealed Langevin dynamics (Definition 2 / Algorithm 6) -- the policy:

        alpha_i = eps * sigma_i^2 / sigma_L^2
        a_{i,t} = a_{i,t-1} + drift_i(a_{i,t-1}) + sqrt(alpha_i) * z_t

    with hand-off a_{i+1,0} <- a_{i,T} across levels, so coarse levels take
    large, heavily-perturbed jumps across a smoothed landscape and the final
    level is a near-deterministic refinement whose stationary law is exp(wQ)/Z.

    The two drift modes are not the same algorithm. `normalize_score` off is the
    paper exactly -- (alpha_i/2) * w * grad_a~ Q -- and works because alpha_i's
    sigma_i^2 cancels the score's ~1/sigma_i^2 magnitude, leaving an O(1) step
    down the whole ladder. Normalizing to a unit vector bounds an untrustworthy
    critic gradient but destroys exactly that cancellation, so the normalized
    drift instead gets its own level-appropriate length, `drift_scale * sigma_i`.
    `w` is a temperature on Q and means nothing once the score is normalized, so
    it applies only to the exact path."""
    key, init_key = jax.random.split(key)
    a = self.init_scale * (2.0 * jax.random.uniform(
      init_key, (obs.shape[0], self.action_dim), jnp.float32) - 1.0)

    score_norm = jnp.zeros(())
    keys = jax.random.split(key, self.num_levels * self.steps_per_level)

    sigmas, alpha_ratio = self.sigmas[...], self.alpha_ratio[...]
    for i in range(self.num_levels):
      alpha = self.step_size * alpha_ratio[i]
      for t in range(self.steps_per_level):
        g = self.critic_score(obs, a, i)
        score_norm = jnp.linalg.norm(g, axis=-1).mean()

        if self.normalize_score:
          drift = self.drift_scale * sigmas[i] * g / (
            jnp.linalg.norm(g, axis=-1, keepdims=True) + 1e-8)
        else:
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
    reward   = batch["rew"]
    term     = batch["term"]

    losses = {}
    keys = jax.random.split(self.rngs.noise(), 3)

    # The Langevin chain differentiates the critic w.r.t. the *action*; its
    # result is a target, so nothing in it trains the critic.
    next_action, score_norm = self.sample(next_obs, keys[0])
    next_action = jax.lax.stop_gradient(next_action)

    clean = self.clean_level

    # -- bootstrap loss. The level conditioning rides in the action, so the
    # critic sees an ordinary (s, a) pair and the target/CrossQ choice is its
    # own business as everywhere else.
    self.critic.td_loss(obs, self.augment_action(action, clean),
                        next_obs, self.augment_action(next_action, clean),
                        reward, term, losses)
    td_loss = losses.pop("q")

    # -- smoothing loss: regress the noised level onto the clean value, which is
    # what makes the critic noise-conditioned. Reusing critic.loss() twice and
    # popping its key is what keeps this representation-agnostic.
    idx   = jax.random.randint(keys[1], (obs.shape[0],), 0, clean)
    noisy = action + self.sigmas[...][idx][:, None] * jax.random.normal(
      keys[2], action.shape, action.dtype)

    smoothing_target = jax.lax.stop_gradient(self.critic.terminal_target(
      self.critic.expectation(obs, self.augment_action(action, clean))))

    self.critic.loss(obs, self.augment_action(noisy, idx), smoothing_target, losses)
    smoothing_loss = losses.pop("q")

    losses["q"] = td_loss + smoothing_loss

    self.metrics.scalar("td_loss", td_loss)
    self.metrics.scalar("smoothing_loss", smoothing_loss)
    self.metrics.vec("action", next_action, limit=1.0)
    self.metrics.scalar("score_norm", score_norm)

    return losses

  def inference_step(self, observation, step_info=None, key=None):
    obs = self.normalize_obs(observation)
    key = self.rngs.noise() if key is None else key
    action, _ = self.sample(obs, key)
    return action, {}
