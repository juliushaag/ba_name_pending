import math
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.networks.flow import TimeEmbedding
from barl_jax.networks.mlp import MLP
from barl_jax.networks.optim import adam
from barl_jax.networks.registry import get_component, register_component
from barl_jax.policy import Actor, Buffer, Critic


def safe_norm(x, axis=-1, keepdims=True, eps=1e-8):
  """L2 norm with well-defined gradients at zero."""
  return jnp.sqrt(jnp.sum(x**2, axis=axis, keepdims=keepdims) + eps)

_LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)


def normal_log_prob(x, mean, std):
  """Elementwise log N(x; mean, std). Hand-rolled rather than pulled from
  distrax/TFP: distrax's Transformed+Tanh has a long-standing NaN log_prob
  issue that lands exactly on SAC, and neither library is otherwise needed."""
  return -0.5 * ((x - mean) / std) ** 2 - jnp.log(std) - _LOG_SQRT_2PI


@register_component
class StochasticActor(Actor):
  """Diagonal-Normal MLP actor shared by SAC/PPO/IQL. `dist()` stays an
  unconstrained Normal; sample()/mode() squash it (`squash_cls`, resolved by
  name from config/models.yaml -- any no-arg Squash works generically here).

  `log_std` is a state-independent parameter rather than a network head --
  barl's deliberate deviation from canonical SAC, kept as-is."""

  def __init__(self, env_stats, *, latent_dim: int, layers: int, squash_cls: str,
               lr: float, adam_b1: float, grad_clip: float, grad_clip_mode: str,
               log_std_min: float, log_std_max: float, log_prob_clip: float,
               activation: str, norm: str,
               norm_input: bool, norm_stem: bool, norm_after_act: bool,
               dropout: float, final_activation: bool,
               residual: bool, norm_kwargs: dict, dtype: str,
               rngs: nnx.Rngs, **kwargs):
    super().__init__(env_stats, **kwargs)

    self.lr, self.adam_b1 = float(lr), float(adam_b1)
    self.grad_clip, self.grad_clip_mode = float(grad_clip), str(grad_clip_mode)
    self.log_std_min, self.log_std_max = float(log_std_min), float(log_std_max)
    self.log_prob_clip = float(log_prob_clip)
    self.log_std = nnx.Param(jnp.zeros((self.action_dim,), jnp.float32))
    self.squash  = get_component(squash_cls)(env_stats)

    self.net = MLP(self.obs_dim, self.action_dim, latent_dim, layers,
                   activation=activation, norm=norm, norm_input=norm_input,
                         norm_stem=norm_stem, norm_after_act=norm_after_act,
                   dropout=dropout, final_activation=final_activation,
                   residual=residual, dtype=dtype, rngs=rngs, **(norm_kwargs or {}))
    self.rngs = rngs

  def dist(self, s):
    """(mean, std) of the pre-squash Normal."""
    mean = self.net(s).astype(jnp.float32)
    std  = jnp.exp(jnp.clip(self.log_std[...], self.log_std_min, self.log_std_max))
    return mean, std

  def log_prob(self, dist, action):
    """`action` is the squashed (canonical [-1, 1]) value; inverts the squash
    and applies the change-of-variables correction."""
    mean, std = dist
    x = self.squash.inverse(jnp.clip(action, -self.log_prob_clip, self.log_prob_clip))
    lp = normal_log_prob(x, mean, std) - self.squash.log_abs_det_jacobian(x, action)
    return lp.sum(axis=-1)

  def rsample(self, s, key=None):
    """Reparameterized (action, pre-squash x, dist) -- differentiable through s."""
    mean, std = self.dist(s)
    key = self.rngs.noise() if key is None else key
    x = mean + std * jax.random.normal(key, mean.shape, mean.dtype)
    return self.squash(x), x, (mean, std)

  def sample(self, s, key=None, critic: Critic = None):
    action, _, _ = self.rsample(s, key)
    return action

  def mode(self, s):
    """Deterministic (mean) action -- IQL trains via AWR but deploys this."""
    mean, _ = self.dist(s)
    return self.squash(mean)

  def create_optimizer(self) -> dict:
    return {"actor": adam(self.lr, self.adam_b1, self.grad_clip, self.grad_clip_mode)}


@register_component
class ScoreContinuousActor(Actor):
  """AMDP continuous-time control (Thilges et al. 2026). The network predicts
  u_tilde with u = sigma(tau) * u_tilde, so each Euler step reduces to
  X += v_i * u_tilde + sqrt(v_i) * eps  with  v_i = int_{tau_i}^{tau_{i+1}} sigma^2.
  Raw, unsquashed output -- the owning Policy applies the erf squash."""

  def __init__(self, env_stats, *, latent_dim: int, layers: int, max_steps: int,
               sigma_min: float, sigma_max: float, tau_min: float, tau_max: float,
               bound_c: float, lr: float, adam_b1: float,
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
    self.bound_c   = float(bound_c)
    self.lr        = float(lr)
    self.adam_b1   = float(adam_b1)
    self.grad_clip = float(grad_clip)
    self.grad_clip_mode = str(grad_clip_mode)
    self._log_r    = math.log(sigma_max / sigma_min)
    self._int_base = float(self._raw_int(jnp.asarray(tau_min, jnp.float32)))

    self.time_emb = TimeEmbedding(self.time_dim, kind=time_embed, rngs=rngs, **time_kwargs)

    self.diffusion_act = MLP(self.obs_dim + self.action_dim + self.time_dim,
                             self.action_dim, latent_dim, layers,
                             activation=activation, norm=norm, norm_input=norm_input,
                             norm_stem=norm_stem, norm_after_act=norm_after_act,
                             dropout=dropout, final_activation=final_activation,
                             residual=residual, dtype=dtype, rngs=rngs, **(norm_kwargs or {}))

    # u ~ 0 at the beginning.
    self.diffusion_act.head.kernel[...] = jnp.zeros_like(self.diffusion_act.head.kernel[...])
    self.diffusion_act.head.bias[...]   = jnp.zeros_like(self.diffusion_act.head.bias[...])

    taus   = jnp.linspace(self.tau_min, self.tau_max, self.max_steps + 1, dtype=jnp.float32)
    sig_sq = self.sigma_sq_int_fn(taus)                        # exact, rebased to tau_min
    self.taus     = Buffer(taus)
    self.step_var = Buffer(jnp.maximum(sig_sq[1:] - sig_sq[:-1], 0.0))
    self.sigma    = Buffer(self.sigma_fn(taus[:-1]))
    self.rngs     = rngs

  def sigma_fn(self, tau):
    """dime_am's `get_geometric_scheduler`: sigma(t) = sigma_min * r^(1-t) *
    sqrt(2 log r), which *decreases* over the path. barl previously used the
    increasing sigma_min * r^t with no sqrt(2 log r) factor -- a different
    process, not a reparameterization of this one."""
    r = self.sigma_max / self.sigma_min
    return self.sigma_min * r ** (1.0 - tau) * math.sqrt(2.0 * self._log_r)

  def _raw_int(self, tau):
    """int_0^tau sigma^2 = sigma_max^2 (1 - r^(-2 tau)), the reference's
    `sigma_t_0(t)^2`, in closed form."""
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
    return float(self.step_var[...].sum())

  def bound_control(self, u_tilde, X, sigma_i):
    """||u|| <= C(1 + ||X||) with u = sigma * u_tilde."""
    max_norm = self.bound_c * (1.0 + safe_norm(X)) / sigma_i
    n = safe_norm(u_tilde)
    return u_tilde * jnp.minimum(max_norm / n, 1.0)

  def generate_action(self, obs, key=None):
    """Euler-Maruyama over the forward SDE. Left as an unrolled Python loop
    rather than a lax.scan: max_steps is 16, the unrolled graph compiles fine,
    and it keeps the step algebra readable and identical to the torch version."""
    B = obs.shape[0]
    key = self.rngs.noise() if key is None else key
    keys = jax.random.split(key, self.max_steps)

    X        = jnp.zeros((B, self.action_dim), jnp.float32)
    path_int = jnp.zeros((B, 1), jnp.float32)

    for i in range(self.max_steps):
      emb     = self.time_emb(jnp.broadcast_to(self.taus[...][i], (B, 1)))
      u_tilde = self.bound_control(self.diffusion_act(obs, X, emb), X, self.sigma[...][i])
      v       = self.step_var[...][i]

      X        = X + v * u_tilde + jnp.sqrt(v) * jax.random.normal(keys[i], u_tilde.shape, u_tilde.dtype)
      path_int = path_int + (v * (u_tilde ** 2).sum(axis=-1))[:, None]

    return X, math.log(2) * self.action_dim - 0.5 * path_int

  def sample(self, s, key=None, critic: Critic = None):
    """Raw, unsquashed -- not a valid action on its own; exists to satisfy the
    Actor contract for standalone use."""
    X, _ = self.generate_action(s, key)
    return X

  def create_optimizer(self) -> dict:
    return {"actor": adam(self.lr, self.adam_b1, self.grad_clip, self.grad_clip_mode)}


class DiscreteDiffusionActorBase(Actor, ABC):
  """Shared skeleton for fixed-step (DDPM-style) diffusion actors.

  Subclasses supply the noise process -- its marginal coefficients, how a raw
  network output becomes a score, what the prior is, and one reverse step --
  and get the forward/DSM helpers and the reverse loop from here. `DDPMActor`
  below covers every variance-preserving ladder; an actor whose process is part
  of its own paper rather than a shared architecture lives next to that
  algorithm instead (OUDiffusionActor in policies/dime.py).

  Every method that draws noise takes an explicit `key`. The reverse loop is
  differentiated through by DIME, so the draws have to be ordinary values in
  the graph rather than a side effect on the module."""

  def __init__(self, env_stats, *, latent_dim: int, layers: int, num_steps: int,
               time_dim: int, time_embed: str, time_kwargs: dict,
               lr: float, adam_b1: float, grad_clip: float, grad_clip_mode: str,
               activation: str, norm: str,
               norm_input: bool, norm_stem: bool, norm_after_act: bool,
               dropout: float, final_activation: bool,
               residual: bool, norm_kwargs: dict, dtype: str,
               rngs: nnx.Rngs, **kwargs):
    super().__init__(env_stats, **kwargs)

    self.num_steps = int(num_steps)
    self.time_dim  = int(time_dim)
    self.lr        = float(lr)
    self.adam_b1   = float(adam_b1)
    self.grad_clip = float(grad_clip)
    self.grad_clip_mode = str(grad_clip_mode)

    self.time_emb = TimeEmbedding(self.time_dim, kind=time_embed, rngs=rngs, **time_kwargs)

    self.diffusion_act = MLP(self.obs_dim + self.action_dim + time_dim, self.action_dim,
                             latent_dim, layers, activation=activation, norm=norm,
                             norm_input=norm_input, norm_stem=norm_stem,
                             norm_after_act=norm_after_act, dropout=dropout,
                             final_activation=final_activation, residual=residual,
                             dtype=dtype, rngs=rngs, **(norm_kwargs or {}))
    self.tau_table = Buffer(jnp.arange(num_steps + 1, dtype=jnp.float32) / num_steps)
    self.rngs = rngs

  # ---- hooks -------------------------------------------------------------
  @abstractmethod
  def marginal_coef(self, n):
    """(gamma, nu): the mean coefficient and variance of q(x_n | x_0)."""
    ...

  @abstractmethod
  def to_score(self, raw, x, n):
    ...

  @abstractmethod
  def prior_sample(self, batch: int, key):
    ...

  @abstractmethod
  def reverse_step(self, obs, x, n, key):
    """-> (x_prev, aux_increment)."""
    ...

  def init_aux(self, x):
    return jnp.zeros((x.shape[0], 1), x.dtype)

  # ---- network -----------------------------------------------------------
  def _tau(self, n, batch: int):
    tau = self.tau_table[...][n]
    tau = jnp.broadcast_to(tau, (batch,)) if jnp.ndim(tau) == 0 else tau
    return tau[:, None]

  def predict_raw(self, obs, x, n):
    emb = self.time_emb(self._tau(n, x.shape[0]))
    return self.diffusion_act(obs, x, emb)

  def predict_score(self, obs, x, n):
    return self.to_score(self.predict_raw(obs, x, n).astype(jnp.float32), x, n)

  # ---- derived forward API (RSM / any DSM loss) ---------------------------
  def diffuse(self, x0, n, key):
    g, nu = self.marginal_coef(n)
    return g * x0 + jnp.sqrt(nu) * jax.random.normal(key, x0.shape, x0.dtype)

  def sample_phi0(self, x_n, n, num_candidates: int, key):
    # Posterior of x_0 given x_n under the same N(0, I) prior `prior_sample`
    # draws from. A flat prior instead spreads the candidates by sqrt(nu)/g,
    # which is 31 at the top of a VP ladder ending at alpha_bar 1e-3 -- every
    # candidate then squashes to the same +-1 corner and the weights that carry
    # the whole learning signal go exactly uniform.
    g, nu = self.marginal_coef(n)
    denom = g ** 2 + nu
    eps   = jax.random.normal(key, (num_candidates, *x_n.shape), x_n.dtype)
    return (g / denom * x_n)[None] + jnp.sqrt(nu / denom)[None] * eps

  def score_target(self, x0, x_n, n):
    g, nu = self.marginal_coef(n)
    return -(x_n - g * x0) / nu

  def dsm_weight(self, n):
    return self.marginal_coef(n)[1]

  # ---- derived forward API (eps parameterization) -------------------------
  # The duals of the four above. The DDPM literature states its losses in eps
  # space where this codebase states them in score space; since every schedule
  # here defines `to_score` as -raw/sqrt(nu), `predict_raw` already *is*
  # eps_theta, so these are renamings and one division rather than a second
  # parameterization to keep in sync.

  def predict_eps(self, obs, x, n):
    return self.predict_raw(obs, x, n).astype(jnp.float32)

  def eps_from_score(self, score, n):
    return -score * jnp.sqrt(self.marginal_coef(n)[1])

  def x0_from_eps(self, x_n, eps, n):
    g, nu = self.marginal_coef(n)
    return (x_n - jnp.sqrt(nu) * eps) / g

  def eps_regression(self, obs, x_n, n, eps_target, weights=None):
    """Weighted eps-space regression against a supplied target. `weights`
    broadcasts over the batch and over any leading candidate axis; None is the
    plain DDPM loss. Reduction is sum-over-action-dims then mean, matching the
    score-space losses here -- note the references mean over both, which is this
    divided by `action_dim`."""
    squared_error = (self.predict_eps(obs, x_n, n) - eps_target) ** 2
    if weights is not None:
      squared_error = weights * squared_error
    return squared_error.sum(axis=-1).mean()

  def eps_loss(self, obs, x0, n, key, weights=None):
    """Denoising regression onto the noise that produced x_n. Draws its own eps
    rather than calling `diffuse`, which discards the draw the target needs."""
    eps = jax.random.normal(key, x0.shape, x0.dtype)
    g, nu = self.marginal_coef(n)
    return self.eps_regression(obs, g * x0 + jnp.sqrt(nu) * eps, n, eps, weights)

  # ---- shared reverse loop -----------------------------------------------
  def generate(self, obs, x=None, n_stop: int = 0, key=None):
    key = self.rngs.noise() if key is None else key
    key, prior_key = jax.random.split(key)
    if x is None:
      x = self.prior_sample(obs.shape[0], prior_key)

    aux = self.init_aux(x)
    keys = jax.random.split(key, max(1, self.num_steps - n_stop))
    for i, n in enumerate(range(self.num_steps, n_stop, -1)):
      x, inc = self.reverse_step(obs, x, n, keys[i])
      aux = aux + inc
    return x, aux

  def sample(self, s, key=None, critic: Critic = None):
    return self.generate(s, key=key)[0]

  def create_optimizer(self) -> dict:
    return {"actor": adam(self.lr, self.adam_b1, self.grad_clip, self.grad_clip_mode)}


def _from_betas(betas):
  """A beta ladder as the alpha_bar it implies, with the clean end prepended."""
  return jnp.concatenate([jnp.ones((1,), jnp.float32), jnp.cumprod(1.0 - betas)])


def alpha_bar_ladder(schedule: str, num_steps: int, *, scale: float, cosine_s: float,
                     ab_start: float, ab_end: float):
  """alpha_bar over n = 0..num_steps, index 0 being the clean end.

  alpha_bar is the parameterization rather than beta because it is what both
  families here are actually defined by: `vp_cosine` prescribes alpha_bar
  directly (and deliberately does *not* start at exactly 1), while the beta
  schedules prescribe increments. Everything downstream reads alpha_bar and
  derives alpha_t = ab_t / ab_prev, so the two meet immediately."""
  if schedule == "vp_cosine":
    # A cosine curve rescaled into [ab_end, ab_start] -- not the Nichol-Dhariwal
    # cosine beta schedule, which this would otherwise be confused with.
    t = jnp.linspace(0.0, 1.0, num_steps + 1, dtype=jnp.float32)
    f = jnp.cos((t + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2
    return ab_end + (ab_start - ab_end) * (f / f[0])

  if schedule == "linear":
    return _from_betas(scale * jnp.linspace(1e-4, 0.999, num_steps, dtype=jnp.float32))

  if schedule == "cosine":
    t  = jnp.linspace(0.0, 1.0, num_steps + 1, dtype=jnp.float32)
    f  = jnp.cos((t + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2
    ab = f / f[0]
    return _from_betas(jnp.clip(scale * (1.0 - ab[1:] / ab[:-1]), 0.0, 0.999))

  if schedule == "vp":
    b_min, b_max = 0.1, 10.0
    t = jnp.arange(1, num_steps + 1, dtype=jnp.float32)
    alpha = jnp.exp(-b_min / num_steps
                    - 0.5 * (b_max - b_min) * (2 * t - 1) / num_steps ** 2)
    return _from_betas(jnp.clip(scale * (1.0 - alpha), 0.0, 0.999))

  raise ValueError(
    f"beta_schedule must be vp_cosine/linear/cosine/vp, got {schedule!r}")


@register_component
class DDPMActor(DiscreteDiffusionActorBase):
  """Fixed-step DDPM over an alpha_bar ladder, with the standard ancestral
  sampler.

  One class covers every variance-preserving discrete diffusion actor here,
  because they differ only in three things and none of them is the update rule:
  the ladder (`beta_schedule`), the prior's scale, and whether the reconstructed
  x0 is clipped back into the canonical action box each reverse step. The
  posterior mean itself is shared -- see `reverse_step` for why the clip is the
  only reason its two branches are written differently.

  `VPDiffusionActor` below is this class with the settings SDAC was tuned on."""

  def __init__(self, env_stats, *, latent_dim: int, layers: int, num_steps: int,
               beta_schedule: str, beta_scale: float,
               cosine_s: float, ab_start: float, ab_end: float,
               clip_denoised: bool, prior_scale: float,
               rngs: nnx.Rngs, **kwargs):
    super().__init__(env_stats, latent_dim=latent_dim, layers=layers, num_steps=num_steps,
                     rngs=rngs, **kwargs)

    self.alpha_bar = Buffer(alpha_bar_ladder(
      beta_schedule, num_steps,
      scale=beta_scale, cosine_s=cosine_s, ab_start=ab_start, ab_end=ab_end))

    self.clip_denoised = bool(clip_denoised)
    self.prior_scale   = float(prior_scale)

  def marginal_coef(self, n):
    ab = self.alpha_bar[...][n][..., None]
    return jnp.sqrt(ab), 1.0 - ab

  def to_score(self, raw, x, n):
    return -raw / jnp.sqrt(self.marginal_coef(n)[1])      # raw == eps_theta

  def prior_sample(self, batch: int, key):
    noise = jax.random.normal(key, (batch, self.action_dim), jnp.float32)
    return self.prior_scale * noise if self.prior_scale != 1.0 else noise

  def reverse_step(self, obs, x, n, key):
    ab_t, ab_prev = self.alpha_bar[...][n], self.alpha_bar[...][n - 1]
    a_t    = ab_t / ab_prev
    beta_t = 1.0 - a_t

    if self.clip_denoised:
      # x0 form, because the clip is a projection onto the action box that has
      # no expression in terms of the score alone.
      x0   = jnp.clip(self.x0_from_eps(x, self.predict_eps(obs, x, n), n), -1.0, 1.0)
      mean = ((jnp.sqrt(ab_prev) * beta_t) * x0
              + (jnp.sqrt(a_t) * (1.0 - ab_prev)) * x) / (1.0 - ab_t)
    else:
      # Algebraically the same posterior mean in eps form, in fewer ops. Kept
      # verbatim rather than folded into the branch above: it is the expression
      # SDAC's published numbers were produced with, and in float32 "equal" and
      # "identical" are not the same claim.
      mean = (x + beta_t * self.predict_score(obs, x, n)) / jnp.sqrt(a_t)

    if n > 1:
      # sigma_t^2 -- Eq. 8 omits the sqrt. `n` is a Python int here (the reverse
      # loop is unrolled), so this branch is resolved at trace time.
      var  = (1.0 - ab_prev) / (1.0 - ab_t) * beta_t
      mean = mean + jnp.sqrt(var) * jax.random.normal(key, x.shape, x.dtype)

    return mean, 0.0


@register_component
class VPDiffusionActor(DDPMActor):
  """Variance-preserving cosine ladder + DDPM ancestral sampler, unit prior, no
  x0 clip. Fixed schedule (Ma et al. Table 4); the settings SDAC was tuned on,
  named so that config and checkpoints keep referring to them by name."""

  def __init__(self, env_stats, **kwargs):
    # Purely a named alias for DDPMActor at the VP settings: what makes it that
    # is its config/components/actors.yaml entry (beta_schedule: vp_cosine,
    # clip_denoised: false, prior_scale: 1.0), not a hardcoded super() call --
    # so a run's config shows the sampler it actually used.
    super().__init__(env_stats, **kwargs)
