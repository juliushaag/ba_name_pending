import gymnasium as gym
import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.networks.actor import ScoreContinuousActor
from barl_jax.networks.critic import QCriticBase
from barl_jax.networks.regulator import AutotuneRegulator
from barl_jax.networks.squash import Squash
from barl_jax.policy import Policy, frozen, register_policy


def clip_score(score, max_norm: float, percentile: float):
  """Per-sample norm clip on the critic's action gradient, the quantity the
  adjoint target is built from. The threshold is the tighter of `max_norm` and
  this batch's `percentile` of the per-sample norms, so a batch that is
  uniformly explosive is still pulled down rather than passing because every
  sample is equally bad. Returns the clipped score and the fraction clipped."""
  norms      = jnp.linalg.norm(score, axis=-1, keepdims=True)
  threshold  = jnp.minimum(jnp.percentile(norms, percentile), max_norm)
  scale      = jnp.where(norms > threshold, threshold / (norms + 1e-6), 1.0)
  return score * scale, jnp.mean(scale < 1.0)


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
    self.entropy_scale          = params["entropy_scale"]
    self.q_score_max_norm       = params["q_score_max_norm"]
    self.q_score_max_percentile = params["q_score_max_percentile"]
    self.policy_delay           = params["policy_delay"]

    self.critic: QCriticBase = self.build_component("critic", QCriticBase)
    self.actor: ScoreContinuousActor = self.build_component("actor", ScoreContinuousActor)

    self.entropy: AutotuneRegulator = self.build_component(
      "entropy", AutotuneRegulator, target=self.action_dim * self.entropy_scale)

    # erf_k is fixed by the actor's terminal variance -- the squash's
    # log-Jacobian is chosen to cancel out of the score regression below.
    self.squash = self.build_component("squash", Squash,
                                       erf_k=(2 * self.actor.terminal_var) ** -0.5)


  def optimizer_cadence(self) -> dict:
    """dime_am's `policy_delay`: the critic steps every gradient step, the actor
    and the entropy dual only every `policy_delay`-th -- both gated by the same
    branch there, so both are gated by the same key here."""
    return {"actor": self.policy_delay, "entropy_alpha": self.policy_delay}

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
    # The actor's own module, not a fresh sinusoidal_emb: a `fourier`/`learned`
    # embedding carries parameters, and a second copy would not be the one the
    # sampler used.
    time_embed = self.actor.time_emb(tau.reshape(X.shape[0], 1))
    u_tilde = self.actor.diffusion_act(obs, X_t, time_embed)

    # The score is a gradient w.r.t. the *action*; the critic must not be
    # trained through the actor loss.
    #
    # nnx.value_and_grad rather than jax's, and the critic passed explicitly
    # rather than captured: an ensemble critic runs nnx.vmap internally, and a
    # raw jax.value_and_grad over a closure holding the module makes that vmap
    # re-extract it at a different trace level ("Cannot extract graph node from
    # different trace level"). Same reason as NCLQL.critic_score.
    def q_values(critic, x):
      return critic.value(obs, self.squash(x)).sum()

    q_new, x_score = nnx.value_and_grad(q_values, argnums=1)(frozen(self.critic), X)
    x_score, clipped_frac = clip_score(x_score, self.q_score_max_norm,
                                       self.q_score_max_percentile)

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
    self.metrics.scalar("score_clipped_frac", clipped_frac)

    return losses

  def inference_step(self, observation, step_info=None, key=None):
    X, _ = self.actor.generate_action(observation, key)
    return self.squash(X), {}
