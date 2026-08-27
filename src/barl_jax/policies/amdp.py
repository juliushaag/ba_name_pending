import math

import gymnasium as gym
import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.networks import gaussian
from barl_jax.networks.diffusion import TimeConditionedField, controlled_sde
from barl_jax.networks.critic import QCriticBase
from barl_jax.networks.optim import adam
from barl_jax.networks.registry import get_component, register_component
from barl_jax.networks.regulator import AutotuneRegulator
from barl_jax.policy import (Actor, Buffer, Critic, Policy, frozen, gated_loss,
                             register_policy)


def clip_score(score, max_norm: float, percentile: float):
  norms      = jnp.linalg.norm(score, axis=-1, keepdims=True)
  threshold  = jnp.minimum(jnp.percentile(norms, percentile), max_norm)
  scale      = jnp.where(norms > threshold, threshold / (norms + 1e-6), 1.0)
  return score * scale, jnp.mean(scale < 1.0)


@register_component
class ScoreContinuousActor(Actor):

  def __init__(self, env_stats, *, latent_dim: int, layers: int, max_steps: int,
               squash_cls: str,
               sigma_min: float, sigma_max: float, tau_min: float, tau_max: float,
               out_clip: float,
               lr: float, adam_b1: float,
               grad_clip: float, grad_clip_mode: str,
               time_dim: int, time_embed: str, time_kwargs: dict,
               activation: str, norm: str,
               norm_input: bool, norm_stem: bool, norm_after_act: bool,
               dropout: float, final_activation: bool,
               residual: bool, norm_kwargs: dict, dtype: str,
               rngs: nnx.Rngs, **kwargs):

    super().__init__(env_stats, **kwargs)

    self.sigma_min, self.sigma_max = float(sigma_min), float(sigma_max)
    self.tau_min,   self.tau_max   = float(tau_min),   float(tau_max)
    self.max_steps, self.time_dim  = int(max_steps), int(time_dim)
    self.out_clip  = float(out_clip)
    self.lr        = float(lr)
    self.adam_b1   = float(adam_b1)
    self.grad_clip = float(grad_clip)
    self.grad_clip_mode = str(grad_clip_mode)
    self._log_r    = math.log(sigma_max / sigma_min)
    self._int_base = float(self._raw_int(jnp.asarray(tau_min, jnp.float32)))

    self.field = TimeConditionedField(
      self.obs_dim, self.action_dim, latent_dim=latent_dim, layers=layers,
      time_dim=self.time_dim, time_embed=time_embed, time_kwargs=time_kwargs,
      activation=activation, norm=norm, norm_input=norm_input, norm_stem=norm_stem,
      norm_after_act=norm_after_act, dropout=dropout, final_activation=final_activation,
      residual=residual, norm_kwargs=norm_kwargs, dtype=dtype, rngs=rngs)

    self.field.zero_init_head()   # u ~ 0 at the beginning

    taus   = jnp.linspace(self.tau_min, self.tau_max, self.max_steps + 1, dtype=jnp.float32)
    sig_sq = self.sigma_sq_int_fn(taus)                        # exact, rebased to tau_min
    self.taus     = Buffer(taus)
    self.step_var = Buffer(jnp.maximum(sig_sq[1:] - sig_sq[:-1], 0.0))
    self.rngs     = rngs

    self._terminal_var = float(self.step_var[...].sum())

    # The squash belongs to the actor because `erf_k` is a function of the noise
    # schedule two lines up -- a Policy holding the squash had to reach back into
    # the actor to build it. Any Squash works here; `erf_k` is absorbed by the
    # ones that do not need it.
    self.squash = get_component(squash_cls)(
      env_stats, erf_k=(2 * self._terminal_var) ** -0.5)

  def sigma_fn(self, tau):
    r = self.sigma_max / self.sigma_min
    return self.sigma_min * r ** (1.0 - tau) * math.sqrt(2.0 * self._log_r)

  def _raw_int(self, tau):
    r = self.sigma_max / self.sigma_min
    return self.sigma_max ** 2 * (1.0 - r ** (-2.0 * tau))

  def sigma_sq_int_fn(self, tau):
    """int_{tau_min}^{tau} sigma^2(t) dt -- matches what the sampler injects."""
    return self._raw_int(tau) - self._int_base

  def sample_noised(self, X, key=None):
    key = self.rngs.noise() if key is None else key
    k_tau, k_eps = jax.random.split(key)

    tau = self.tau_min + jax.random.uniform(k_tau, (X.shape[0],), jnp.float32) * (self.tau_max - self.tau_min)
    sig_sq_tau = self.sigma_sq_int_fn(tau)[:, None]
    sig_sq_one = self.sigma_sq_int_fn(jnp.full((X.shape[0],), self.tau_max, jnp.float32))[:, None]

    var = jnp.maximum(sig_sq_tau - (sig_sq_tau ** 2 / sig_sq_one), 0.0)
    X_t = X * (sig_sq_tau / sig_sq_one) + jnp.sqrt(var) * jax.random.normal(k_eps, X.shape, X.dtype)

    return X_t, tau

  @property
  def terminal_var(self) -> float:
    """Var(X_1) at zero control. erf_k MUST be derived from this."""
    return self._terminal_var

  def squash_correction(self, X):
    """H(action) = H(X) + E[log|dy/dx|]: the squash lifts the path weight onto
    the action's density, whatever class fills it."""
    return self.squash.log_abs_det_jacobian(X).sum(axis=-1, keepdims=True)

  def derived(self) -> dict:
    return self.squash.derived()

  def prior_surprisal(self, X):
    return gaussian.surprisal(X, self._terminal_var, self.action_dim)

  def control(self, obs, x, tau):
    """The control net's output, clipped the way dime_am's does it
    (ctrl_net.py `outer_clip`). At 1e4 it never binds on a healthy run; it is
    what keeps one blown-up step from writing inf into the whole chain."""
    u = self.field(obs, x, self.scaled_time(tau))
    return jnp.clip(u, -self.out_clip, self.out_clip)

  def scaled_time(self, tau):
    """The level the control net conditions on. The reference scales it into
    [-1, 1] before its fourier coder (ctrl_net.py), so the fixed frequency ladder
    reads the same span whatever tau_min/tau_max say."""
    return 2.0 * tau - 1.0

  def generate_action(self, obs, squash=True, key=None):
    
    B = obs.shape[0]
    key = self.rngs.noise() if key is None else key

    def euler_step(X, i, step_key):
      tau     = jnp.broadcast_to(self.taus[...][i], (B, 1))
      u_tilde = self.control(obs, X, tau)
      v       = self.step_var[...][i]
      eps     = jax.random.normal(step_key, u_tilde.shape, u_tilde.dtype)

      log_w = -0.5 * v * (u_tilde ** 2).sum(axis=-1, keepdims=True) \
              - jnp.sqrt(v) * (eps * u_tilde).sum(axis=-1, keepdims=True)
      return X + v * u_tilde + jnp.sqrt(v) * eps, log_w

    X = jnp.zeros((B, self.action_dim), jnp.float32)
    X, log_w = controlled_sde(euler_step, X, self.max_steps, key,
                              aux=jnp.zeros((B, 1), jnp.float32))

    log_w = log_w + self.prior_surprisal(X) 

    if squash:
      log_w = log_w + self.squash_correction(X)
      X = self.squash(X)

    return X, log_w

  def actor_loss(self, obs, critic: Critic, alpha,
                 *, max_norm: float, percentile: float, key):
    """Returns the loss rather than writing it into a `losses` dict, because the
    owning Policy runs it under `gated_loss` -- a branch has to hand its value
    back, it cannot mutate a dict on one side and not the other."""
    rollout_key, noise_key = jax.random.split(key)

    X, _ = jax.lax.stop_gradient(self.generate_action(obs, squash=False, key=rollout_key))

    X_t, tau = self.sample_noised(X, noise_key)
    u_tilde  = self.control(obs, X_t, tau[:, None])

    def q_values(critic, x):
      return critic.value(obs, self.squash(x)).sum()

    q_new, x_score = nnx.value_and_grad(q_values, argnums=1)(frozen(critic), X)
    x_score, clipped_frac = clip_score(x_score, max_norm, percentile)

    sigma_tau = self.sigma_fn(tau)[:, None]

    u      = sigma_tau * u_tilde
    target = jax.lax.stop_gradient(sigma_tau * (x_score / alpha))
    loss   = (0.5 * ((u - target) ** 2).sum(axis=-1) / sigma_tau.squeeze(-1)).mean()

    self.metrics.scalar("score_loss", loss)
    self.metrics.dist("score_target", target)
    self.metrics.vec("u", u)
    self.metrics.scalar("score_clipped_frac", clipped_frac)
    critic.metrics.scalar("q_at_sample", q_new / obs.shape[0])
    critic.metrics.vec("score", x_score)
    return loss

  def create_optimizer(self) -> dict:
    return {"actor": adam(self.lr, self.adam_b1, self.grad_clip, self.grad_clip_mode)}


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

    # Every draw this step descends from one key, split here: the two sampler
    # rollouts must not share a stream, and a step's randomness should be
    # readable off train_step rather than hidden in each component's `rngs`.
    keys = jax.random.split(self.rngs.noise(), 2)

    next_action, entropy = jax.lax.stop_gradient(
      self.actor.generate_action(next_obs, key=keys[0]))
    entropy_bonus = jax.lax.stop_gradient(alpha * entropy)

    self.critic.td_loss(obs, action, next_obs, next_action, reward, term,
                        losses, shift=entropy_bonus)

    # Skipped outright on the two steps in three `policy_delay` would mask the
    # update on anyway: the loss prices through `frozen` and its target is
    # stop_gradient'd, so nothing here reaches the critic's parameters and the
    # skip is parameter-identical. What it drops is the second 16-step chain and
    # a value_and_grad through the 2048-wide ensemble.
    losses["actor"] = gated_loss(
      self, self.policy_delay,
      lambda actor, critic: actor.actor_loss(
        obs, critic, alpha, max_norm=self.q_score_max_norm,
        percentile=self.q_score_max_percentile, key=keys[1]),
      self.actor, self.critic)

    self.entropy.loss(entropy, losses)

    self.actor.metrics.vec("action", next_action, limit=1.0)
    self.actor.metrics.dist("entropy", entropy)

    return losses

  def inference_step(self, observation, step_info=None, key=None):
    act, _ = self.actor.generate_action(observation, key=key)
    return act, {}
