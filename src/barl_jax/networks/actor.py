import math
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from barl_jax.networks.flow import sinusoidal_emb
from barl_jax.networks.mlp import MLP
from barl_jax.networks.registry import get_component, register_component
from barl_jax.policy import Actor, Buffer, Critic

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0

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

  def __init__(self, env_stats, hidden_dim: int, layers: int, squash_cls: str,
               lr: float = 3e-4,
               activation: str = "silu", norm: str = None, norm_stem: bool = False,
               dropout: float = 0.0, final_activation: bool = False,
               residual: bool = False, norm_kwargs: dict = None,
               *, rngs: nnx.Rngs, **kwargs):
    super().__init__(env_stats, **kwargs)

    self.lr = float(lr)
    self.log_std = nnx.Param(jnp.zeros((self.action_dim,), jnp.float32))
    self.squash  = get_component(squash_cls)(env_stats)

    self.net = MLP(self.obs_dim, self.action_dim, hidden_dim, layers,
                   activation=activation, norm=norm, norm_stem=norm_stem,
                   dropout=dropout, final_activation=final_activation,
                   residual=residual, rngs=rngs, **(norm_kwargs or {}))
    self.rngs = rngs

  def dist(self, s):
    """(mean, std) of the pre-squash Normal."""
    mean = self.net(s).astype(jnp.float32)
    std  = jnp.exp(jnp.clip(self.log_std[...], LOG_STD_MIN, LOG_STD_MAX))
    return mean, std

  def log_prob(self, dist, action):
    """`action` is the squashed (canonical [-1, 1]) value; inverts the squash
    and applies the change-of-variables correction."""
    mean, std = dist
    x = self.squash.inverse(jnp.clip(action, -0.99, 0.99))
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
    return {"actor": optax.adam(self.lr)}


@register_component
class ScoreContinuousActor(Actor):
  """AMDP continuous-time control (Thilges et al. 2026). The network predicts
  u_tilde with u = sigma(tau) * u_tilde, so each Euler step reduces to
  X += v_i * u_tilde + sqrt(v_i) * eps  with  v_i = int_{tau_i}^{tau_{i+1}} sigma^2.
  Raw, unsquashed output -- the owning Policy applies the erf squash."""

  def __init__(self, env_stats, latent_dim: int, layers: int, max_steps: int,
               sigma_min: float, sigma_max: float, tau_min: float, tau_max: float,
               bound_c: float = 50.0, lr: float = 3e-4,
               activation: str = "silu", norm: str = "none", norm_stem: bool = False,
               dropout: float = 0.0, final_activation: bool = False,
               residual: bool = False, norm_kwargs: dict = None,
               *, rngs: nnx.Rngs, **kwargs):

    super().__init__(env_stats, **kwargs)

    self.sigma_min, self.sigma_max = float(sigma_min), float(sigma_max)
    self.tau_min,   self.tau_max   = float(tau_min),   float(tau_max)
    self.max_steps, self.time_dim  = int(max_steps), 8
    self.bound_c   = float(bound_c)
    self.lr        = float(lr)
    self._log_r    = math.log(sigma_max / sigma_min)
    self._int_base = float(self._raw_int(jnp.asarray(tau_min, jnp.float32)))

    self.diffusion_act = MLP(self.obs_dim + self.action_dim + self.time_dim,
                             self.action_dim, latent_dim, layers,
                             activation=activation, norm=norm, norm_stem=norm_stem,
                             dropout=dropout, final_activation=final_activation,
                             residual=residual, rngs=rngs, **(norm_kwargs or {}))

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
    return self.sigma_min * (self.sigma_max / self.sigma_min) ** tau

  def _raw_int(self, tau):
    return self.sigma_min ** 2 * ((self.sigma_max / self.sigma_min) ** (2 * tau) - 1) / (2 * self._log_r)

  def sigma_sq_int_fn(self, tau):
    """int_{tau_min}^{tau} sigma^2(t) dt -- matches what the sampler injects."""
    return self._raw_int(tau) - self._int_base

  def sample_noised(self, X, key=None):
    key = self.rngs.noise() if key is None else key
    k_tau, k_eps = jax.random.split(key)

    tau = self.tau_min + jax.random.uniform(k_tau, (X.shape[0],), jnp.float32) * (self.tau_max - self.tau_min)
    sig_sq_tau = self.sigma_sq_int_fn(tau)[:, None]
    sig_sq_one = self.sigma_sq_int_fn(jnp.full((X.shape[0],), self.tau_max, jnp.float32))[:, None]

    # Sample X from the policy, NOT from the batch -- the bridge is toward the
    # policy's own terminal sample.
    var = jnp.maximum(sig_sq_tau - (sig_sq_tau ** 2 / sig_sq_one), 0.0)
    X_t = X * (sig_sq_tau / sig_sq_one) + jnp.sqrt(var) * jax.random.normal(k_eps, X.shape, X.dtype)

    return X_t, tau

  @property
  def terminal_var(self) -> float:
    """Var(X_1) at zero control. erf_k MUST be derived from this."""
    return float(self.step_var[...].sum())

  def bound_control(self, u_tilde, X, sigma_i):
    """||u|| <= C(1 + ||X||) with u = sigma * u_tilde."""
    max_norm = self.bound_c * (1.0 + jnp.linalg.norm(X, axis=-1, keepdims=True)) / sigma_i
    n = jnp.maximum(jnp.linalg.norm(u_tilde, axis=-1, keepdims=True), 1e-8)
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
      emb     = sinusoidal_emb(jnp.broadcast_to(self.taus[...][i], (B, 1)), self.time_dim)
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
    return {"actor": optax.adam(self.lr)}


class DiscreteDiffusionActorBase(Actor, ABC):
  """Shared skeleton for fixed-step (DDPM-style) diffusion actors.

  Subclasses supply the noise process -- its marginal coefficients, how a raw
  network output becomes a score, what the prior is, and one reverse step --
  and get the forward/DSM helpers and the reverse loop from here. The concrete
  actors live next to their algorithms (OUDiffusionActor in policies/dime.py,
  VPDiffusionActor in policies/sdac.py) because their schedules are part of
  those papers, not a shared architecture.

  Every method that draws noise takes an explicit `key`. The reverse loop is
  differentiated through by DIME, so the draws have to be ordinary values in
  the graph rather than a side effect on the module."""

  def __init__(self, env_stats, latent_dim: int, layers: int, num_steps: int,
               time_dim: int = 8, lr: float = 3e-4,
               activation: str = "silu", norm: str = "none", norm_stem: bool = False,
               dropout: float = 0.0, final_activation: bool = False,
               residual: bool = False, norm_kwargs: dict = None,
               *, rngs: nnx.Rngs, **kwargs):
    super().__init__(env_stats, **kwargs)

    self.num_steps = int(num_steps)
    self.time_dim  = int(time_dim)
    self.lr        = float(lr)

    self.diffusion_act = MLP(self.obs_dim + self.action_dim + time_dim, self.action_dim,
                             latent_dim, layers, activation=activation, norm=norm,
                             norm_stem=norm_stem, dropout=dropout,
                             final_activation=final_activation, residual=residual,
                             rngs=rngs, **(norm_kwargs or {}))
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
    emb = sinusoidal_emb(self._tau(n, x.shape[0]), self.time_dim)
    return self.diffusion_act(obs, x, emb)

  def predict_score(self, obs, x, n):
    return self.to_score(self.predict_raw(obs, x, n).astype(jnp.float32), x, n)

  # ---- derived forward API (RSM / any DSM loss) ---------------------------
  def diffuse(self, x0, n, key):
    g, nu = self.marginal_coef(n)
    return g * x0 + jnp.sqrt(nu) * jax.random.normal(key, x0.shape, x0.dtype)

  def sample_phi0(self, x_n, n, num_candidates: int, key):
    g, nu = self.marginal_coef(n)
    eps = jax.random.normal(key, (num_candidates, *x_n.shape), x_n.dtype)
    return (x_n / g)[None] + (jnp.sqrt(nu) / g)[None] * eps

  def score_target(self, x0, x_n, n):
    g, nu = self.marginal_coef(n)
    return -(x_n - g * x0) / nu

  def dsm_weight(self, n):
    return self.marginal_coef(n)[1]

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
    return {"actor": optax.adam(self.lr)}
