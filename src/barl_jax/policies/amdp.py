import gymnasium as gym
import jax

from barl_jax.networks.actor import ScoreContinuousActor
from barl_jax.networks.critic import QCriticBase
from barl_jax.networks.flow import sinusoidal_emb
from barl_jax.networks.regulator import AutotuneRegulator
from barl_jax.networks.squash import Squash
from barl_jax.policy import Policy, frozen, register_policy


@register_policy
class AMDP(Policy):
  """AMDP (Thilges et al. 2026): direct-score regression against a critic's
  pessimistic action-gradient, over a ScoreContinuousActor's forward-SDE vector
  field, with an entropy dual.

  Ported from the torch implementation as it actually runs, which differs from
  its own docstring there in two ways worth knowing: there is **no trust-region
  dual** (the `actor_old` EMA was stepped every train_step but read by no loss,
  and config/models.yaml's `trust_region` slot was never built -- it was
  dropped in 8129930), and AMDP is the only policy that does **not** normalize
  observations. Both are preserved here rather than quietly changed."""

  def __init__(self, env: gym.Env, **kwargs):
    super().__init__(env, **kwargs)

    assert isinstance(env.action_space, gym.spaces.Box)

    params = self.resolve_params()
    self.entropy_scale = params["entropy_scale"]

    self.critic: QCriticBase = self.build_component("critic", QCriticBase)
    self.actor: ScoreContinuousActor = self.build_component("actor", ScoreContinuousActor)

    self.entropy: AutotuneRegulator = self.build_component(
      "entropy", AutotuneRegulator, target=self.action_dim * self.entropy_scale)

    # erf_k is fixed by the actor's terminal variance -- the squash's
    # log-Jacobian is chosen to cancel out of the score regression below.
    self.squash = self.build_component("squash", Squash,
                                       erf_k=(2 * self.actor.terminal_var) ** -0.5)

  def train_step(self, batch):
    obs      = batch["obs"]
    next_obs = batch["next_obs"]
    action   = batch["act"]
    reward   = batch["rew"]
    term     = batch["term"]

    alpha  = self.entropy.value()
    losses = {}

    # Both terminal samples are targets: the SDE rollout is never
    # differentiated through, only the single-step vector field below is.
    next_X, entropy_penalty = jax.lax.stop_gradient(self.actor.generate_action(next_obs))
    X, _ = jax.lax.stop_gradient(self.actor.generate_action(obs))

    entropy_bonus = jax.lax.stop_gradient(alpha * entropy_penalty)
    next_action   = self.squash(next_X)

    self.critic.td_loss(obs, action, next_obs, next_action, reward, term,
                        losses, shift=entropy_bonus)

    # --- actor: regress the SDE's vector field onto the critic's action score
    X_t, tau = self.actor.sample_noised(X)
    time_embed = sinusoidal_emb(tau.reshape(X.shape[0], 1), self.actor.time_dim)
    u_tilde = self.actor.diffusion_act(obs, X_t, time_embed)

    # The score is a gradient w.r.t. the *action*; the critic must not be
    # trained through the actor loss.
    critic = frozen(self.critic)

    def q_values(x):
      return critic.value(obs, self.squash(x)).sum()

    q_new, x_score = jax.value_and_grad(q_values)(X)

    sigma_tau = self.actor.sigma_fn(tau)[:, None]

    u      = sigma_tau * u_tilde
    target = jax.lax.stop_gradient(sigma_tau * (x_score / alpha))
    amdp_loss = (0.5 * ((u - target) ** 2).sum(axis=-1) / sigma_tau.squeeze(-1)).mean()

    self.entropy.loss(entropy_penalty, losses)

    losses["actor"] = amdp_loss

    self.metrics.scalar("score_loss", amdp_loss)
    # Free: value_and_grad returns it alongside the score the actor regresses on.
    self.critic.metrics.dist("q_at_sample", q_new)
    self.metrics.dist("score_target", target)
    self.actor.metrics.vec("u", u)
    self.actor.metrics.vec("action", next_action, limit=1.0)
    self.actor.metrics.dist("entropy", entropy_penalty)
    self.critic.metrics.vec("score", x_score)

    return losses

  def inference_step(self, observation, step_info=None, key=None):
    X, _ = self.actor.generate_action(observation, key)
    return self.squash(X), {}
