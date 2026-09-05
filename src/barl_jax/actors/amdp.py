import math

import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.components import Actor, Buffer, Critic, frozen
from barl_jax.config.toolkit import ConfigMixin
from barl_jax.hparam import Config, HParam
from barl_jax.networks import gaussian
from barl_jax.networks.diffusion import TimeConditionedField, controlled_sde
from barl_jax.networks.flow import TIME_STYLES
from barl_jax.networks.mlp import MLP_PRESETS
from barl_jax.networks.optim import ActorOptim, adam
from barl_jax.networks.registry import get_component, register_component


def clip_score(score, max_norm: float, percentile: float):
  norms      = jnp.linalg.norm(score, axis=-1, keepdims=True)
  threshold  = jnp.minimum(jnp.percentile(norms, percentile), max_norm)
  scale      = jnp.where(norms > threshold, threshold / (norms + 1e-6), 1.0)
  return score * scale, jnp.mean(scale < 1.0)


@register_component
class ScoreContinuousActor(Actor, ActorOptim):
  """AMDP's controlled-SDE actor. The net is `silu_fp32` (the chain accumulates
  a log-weight) and the ctrl time coder by default; both are `Config` selectors
  you can switch with `mlp_preset`/`time_emb` -- configuration, not inheritance.
  Only the SDE and score-clipping knobs are this class's own HParams."""

  kind = "actor"
  # fp32 not bf16: the 16-step chain accumulates a log-weight, and `eps` is
  # drawn at the control net's dtype -- in bf16 the noise driving the SDE
  # carries ~3 significant digits. AMDP went non-finite in bf16 on both DMC
  # envs (sweep 2026-08-27_13-23-15).
  mlp_preset = Config("silu_fp32", style=MLP_PRESETS)
  time_emb  = Config("ctrl", style=TIME_STYLES)
  squash_cls = Config("ErfSquash")   # `erf_k` is derived from the noise schedule below
  sigma_min    = HParam(0.01)
  sigma_max    = HParam(2.0)
  tau_min      = HParam(0.0)
  tau_max      = HParam(1.0)
  max_steps    = Config(16)
  out_clip     = Config(10000.0)   # ctrl_net.py's `outer_clip`
  # The critic's action-gradient can spike on an outlier sample; this is the
  # score-clipping intrinsic to how *this* actor's loss uses that gradient,
  # not an algorithm-level tunable.
  q_score_max_norm       = HParam(5.0)
  q_score_max_percentile = HParam(95.0)

  def __init__(self, env_stats, config: ConfigMixin, *, rngs: nnx.Rngs, **kwargs):
    self.resolve_hparams(config, env_stats=env_stats)

    self._log_r    = math.log(self.sigma_max / self.sigma_min)
    self._int_base = float(self._raw_int(jnp.asarray(self.tau_min, jnp.float32)))

    # latent_dim/layers are tier-ladder-owned.
    latent_dim, layers = config.get("latent_dim", int), config.get("layers", int)

    self.field = TimeConditionedField(
      self.obs_dim, self.plan_dim, latent_dim=latent_dim, layers=layers,
      time_dim=self.time_dim, time_embed=self.time_embed, time_kwargs=self.hparam("time_kwargs"),
      activation=self.activation, norm=self.norm, norm_input=self.norm_input, norm_stem=self.norm_stem,
      norm_after_act=self.norm_after_act, dropout=self.dropout, final_activation=self.final_activation,
      residual=self.residual, norm_kwargs=self.hparam("norm_kwargs"), dtype=self.dtype, rngs=rngs)

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
    self.squash = get_component(self.squash_cls)(
      env_stats, ConfigMixin(), erf_k=(2 * self._terminal_var) ** -0.5)

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
    return gaussian.surprisal(X, self._terminal_var, self.plan_dim)

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

  def generate_action(self, obs, *, critic: Critic = None, squash=True, key=None):
    """`critic` is unused -- AMDP's actor needs none to sample -- and accepted
    only to satisfy the shared `Actor.generate_action` contract."""
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

    X = jnp.zeros((B, self.plan_dim), jnp.float32)
    X, log_w = controlled_sde(euler_step, X, self.max_steps, key,
                              aux=jnp.zeros((B, 1), jnp.float32))

    log_w = log_w + self.prior_surprisal(X)

    if squash:
      log_w = log_w + self.squash_correction(X)
      X = self.squash(X)

    return X.reshape(B, self.chunk_size, self.action_dim), log_w

  def loss(self, obs, critic: Critic, alpha, *, key=None):
    """The direct-score regression against the critic's action-gradient."""
    key = self.rngs.noise() if key is None else key
    rollout_key, noise_key = jax.random.split(key)

    X, _ = jax.lax.stop_gradient(self.generate_action(obs, squash=False, key=rollout_key))
    X = X.reshape(X.shape[0], -1)

    X_t, tau = self.sample_noised(X, noise_key)
    u_tilde  = self.control(obs, X_t, tau[:, None])

    def q_values(critic, x):
      return critic.value(obs, self.squash(x)).sum()

    x_score = nnx.grad(q_values, argnums=1)(frozen(critic), X)
    x_score, clipped_frac = clip_score(x_score, self.q_score_max_norm, self.q_score_max_percentile)

    sigma_tau = self.sigma_fn(tau)[:, None]

    u      = sigma_tau * u_tilde
    target = jax.lax.stop_gradient(sigma_tau * (x_score / alpha))
    loss   = (0.5 * ((u - target) ** 2).sum(axis=-1) / sigma_tau.squeeze(-1)).mean()

    self.metrics.scalar("score_loss", loss)
    self.metrics.dist("score_target", target)
    self.metrics.vec("u", u)
    self.metrics.scalar("score_clipped_frac", clipped_frac)
    critic.metrics.vec("score", x_score)
    return {"actor": loss}

  def create_optimizer(self) -> dict:
    return {"actor": adam(self.lr, self.adam_b1, self.grad_clip, self.grad_clip_mode)}
