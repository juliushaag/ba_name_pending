import math

import torch
from torch import nn
import torch.nn.functional as F

from barl.networks import bfloat16
from barl.networks.mlp import MLP
from barl.networks.flow import sinusoidal_emb
from barl.networks.registry import register_component, get_component

from barl.policy import Actor, Critic

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


@register_component
class DeterministicActor(Actor, nn.Module):
  """TD3-style point-estimate actor (pure BC regression, used by DirectPolicy).
  Output is a plain clamp, not a real squash -- training against the replay
  buffer's already-canonical actions is what keeps it near [-1, 1]."""

  def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, layers: int, **kwargs):
    super().__init__(**kwargs)

    self.action_dim = action_dim
    self.net = MLP(obs_dim, action_dim, hidden_dim, layers)

  def forward(self, s: torch.Tensor) -> torch.Tensor:
    return self.net(s)

  def sample(self, s: torch.Tensor, critic: Critic = None) -> torch.Tensor:
    return self(s).clamp(-1.0, 1.0)

  def create_optimizer(self) -> dict:
    return {"actor": torch.optim.AdamW(self.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=1e-4, fused=True)}


@register_component
class StochasticActor(Actor, nn.Module):
  """Diagonal-Normal MLP actor shared by SAC/PPO/IQL. `dist()` stays an
  unconstrained Normal; sample()/mode() squash it (`squash_cls`, resolved by
  name from config/models.yaml -- any no-arg Squash works generically here)."""

  def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, layers: int, squash_cls: str, **kwargs):
    super().__init__(**kwargs)

    self.action_dim = action_dim

    self.log_std = nn.Parameter(torch.zeros(action_dim))
    self.squash  = get_component(squash_cls)()

    self.net = MLP(obs_dim, action_dim, hidden_dim, layers)

  def dist(self, s: torch.Tensor) -> torch.distributions.Distribution:
    out = self.net(s)
    log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
    return torch.distributions.Normal(out.float(), log_std.exp())

  def log_prob(self, dist: torch.distributions.Distribution, action: torch.Tensor) -> torch.Tensor:
    """`action` is the squashed (canonical [-1, 1]) value; inverts the squash
    and applies the change-of-variables correction."""
    x = self.squash.inverse(action)
    log_prob = dist.log_prob(x) - self.squash.log_abs_det_jacobian(x, action)
    return log_prob.sum(dim=-1)

  def sample(self, s: torch.Tensor, critic: Critic = None) -> torch.Tensor:
    action = self.dist(s).sample()
    return self.squash(action)

  def mode(self, s: torch.Tensor) -> torch.Tensor:
    """Deterministic (mean) action -- IQL trains via AWR but deploys this, not sample()."""
    dist = self.dist(s)
    return self.squash(dist.mean)

  def create_optimizer(self) -> dict:
    return {"actor": torch.optim.Adam(self.parameters(), lr=3e-4)}

@register_component
class ScoreContinuousActor(Actor, nn.Module):
  """AMDP continuous-time control (Thilges et al. 2026). The network predicts
  u_tilde with u = sigma(tau) * u_tilde, so each Euler step reduces to
  X += v_i * u_tilde + sqrt(v_i) * eps  with  v_i = \int_{tau_i}^{tau_{i+1}} sigma^2.
  Raw, unsquashed output -- the owning Policy applies the erf squash."""

  def __init__(self, obs_dim: int, action_dim: int, latent_dim: int, layers: int, max_steps: int,
               sigma_min: float, sigma_max: float, tau_min: float, tau_max: float,
               bound_c: float = 50.0, layernorm=False, **kwargs):
    
    super().__init__(**kwargs)

    self.sigma_min, self.sigma_max = sigma_min, sigma_max
    self.tau_min,   self.tau_max   = tau_min,   tau_max
    self.action_dim, self.max_steps, self.time_dim = action_dim, max_steps, 8
    self.bound_c   = bound_c
    self._log_r    = math.log(sigma_max / sigma_min)
    self._int_base = float(self._raw_int(torch.tensor(float(tau_min))))

    self.diffusion_act = MLP(obs_dim + action_dim + self.time_dim, action_dim, latent_dim, layers, layernorm=layernorm)

    taus   = torch.linspace(tau_min, tau_max, max_steps + 1)
    sig_sq = self.sigma_sq_int_fn(taus)                        # exact, rebased to tau_min
    self.register_buffer("taus",     taus)
    self.register_buffer("step_var", (sig_sq[1:] - sig_sq[:-1]).clamp_min(0.0))
    self.register_buffer("sigma",    self.sigma_fn(taus[:-1]))

  def sigma_fn(self, tau):
    return self.sigma_min * (self.sigma_max / self.sigma_min) ** tau

  def _raw_int(self, tau):
    return self.sigma_min ** 2 * ((self.sigma_max / self.sigma_min) ** (2 * tau) - 1) / (2 * self._log_r)

  def sigma_sq_int_fn(self, tau):
    """\int_{tau_min}^{tau} sigma^2(t) dt -- matches what the sampler injects."""
    return self._raw_int(tau) - self._int_base
  
  @property
  def terminal_var(self) -> float:
    """Var(X_1) at zero control. erf_k MUST be derived from this."""
    return float(self.step_var.sum())

  def bound_control(self, u_tilde, X, sigma_i):
    """||u|| <= C(1 + ||X||) with u = sigma * u_tilde."""
    max_norm = self.bound_c * (1.0 + X.norm(dim=-1, keepdim=True)) / sigma_i
    n = u_tilde.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return u_tilde * (max_norm / n).clamp(max=1.0)
  
  def generate_action(self, obs):
      B = obs.shape[0]
      X        = torch.zeros((B, self.action_dim), device=obs.device)
      path_int = torch.zeros((B, 1),               device=obs.device)

      for i in range(self.max_steps):
        emb     = sinusoidal_emb(self.taus[i].expand(B, 1), self.time_dim)
        u_tilde = self.bound_control(self.diffusion_act(obs, X, emb), X, self.sigma[i])
        v       = self.step_var[i]

        X        = X + v * u_tilde + v.sqrt() * torch.randn_like(u_tilde)
        path_int = path_int + (v * (u_tilde ** 2).sum(-1)).unsqueeze(1)   # = \int ||u||^2 dtau

      return X, math.log(2) * self.action_dim - 0.5 * path_int

  def sample(self, s: torch.Tensor, critic: Critic = None) -> torch.Tensor:
    """Raw, unsquashed -- not a valid action on its own; exists to satisfy
    the Actor contract for standalone use."""
    X, _ = self.generate_action(s)
    return X

  def create_optimizer(self) -> dict:
    return {"actor": torch.optim.Adam(self.diffusion_act.parameters(), lr=3e-4)}
import math

import torch
from torch import nn
import torch.nn.functional as F

from barl.networks.mlp import MLP
from barl.networks.flow import sinusoidal_emb
from barl.networks.registry import register_component

from barl.policy import Actor, Critic


def _inv_softplus(y: float) -> float:
  return math.log(math.expm1(y))


def _inv_sigmoid(y: float) -> float:
  return math.log(y / (1.0 - y))


@register_component
class ScoreDiscreteActor(Actor, nn.Module):
  """Discrete-time diffusion actor for DIME (Celik et al. 2025) """

  EULER_MAX_STEP = 0.35
  MIN_TOTAL_NOISE = 2.0

  def __init__(self, obs_dim: int, action_dim: int, latent_dim: int, layers: int, num_steps: int,
               beta_min: float, beta_max: float, noise_start: float, noise_end: float,
               beta_scale_min: float = 0.7,
               beta_scale_max: float = 1.8,
               learn_beta_scale: bool = True,
               eta_init: float = 1.5811,          # sqrt(2.5), the paper's prior
               learn_eta: bool = False,
               anchor_score: bool = True, **kwargs):
    super().__init__(**kwargs)

    self.num_steps    = num_steps
    self.action_dim   = action_dim
    self.time_dim     = 8
    self.anchor_score = anchor_score

    self.diffusion_act = MLP(obs_dim + action_dim + self.time_dim, action_dim, latent_dim, layers, layernorm=False)
    t    = torch.linspace(0.0, 1.0, num_steps + 1)
    beta = beta_min + 0.5 * (beta_max - beta_min) * (1.0 - torch.cos(math.pi * t))
    self.register_buffer("beta", beta)

    self.register_buffer("logvar_offset", 0.5 * action_dim * (beta[1:].log() - beta[:-1].log()))

    self.beta_scale_min = beta_scale_min
    self.beta_scale_max = min(beta_scale_max, self.EULER_MAX_STEP / float(beta[1:].max()))

    self.total_noise    = float(beta[1:].sum())
    self.max_step       = float(beta[1:].max()) * self.beta_scale_max
    self.logratio_const = float(self.logvar_offset.sum())

    # ---- variance-preserving schedule (SDAC path only) -------------------
    s  = 0.008
    tt = torch.linspace(0, 1, num_steps + 1)
    f  = torch.cos((tt + s) / (1 + s) * math.pi / 2) ** 2
    self.register_buffer("alpha_bars", noise_end + (noise_start - noise_end) * (f / f[0]))

    lo, hi = self.beta_scale_min, self.beta_scale_max
    self._beta_scale_raw = nn.Parameter(
        torch.full((action_dim,), _inv_sigmoid((1.0 - lo) / (hi - lo))),
        requires_grad=learn_beta_scale)
    self._eta_raw = nn.Parameter(
        torch.full((action_dim,), _inv_softplus(eta_init)), requires_grad=learn_eta)

  @property
  def beta_scale(self) -> torch.Tensor:
    lo, hi = self.beta_scale_min, self.beta_scale_max
    return lo + (hi - lo) * torch.sigmoid(self._beta_scale_raw)

  @property
  def eta(self) -> torch.Tensor:
    return F.softplus(self._eta_raw).clamp(0.25, 4.0)

  def gaussian_logprob(self, x, mean, var):
    return (-0.5 * (x - mean) ** 2 / var - 0.5 * torch.log(2 * math.pi * var)).sum(-1, keepdim=True)

  def time_emb(self, t: torch.Tensor) -> torch.Tensor:
    return sinusoidal_emb(t, self.time_dim)

  def time_emb_for(self, t: torch.Tensor, batch_size: int) -> torch.Tensor:
    if t.dim() == 0:
      t = t.expand(batch_size)
    return self.time_emb(t.float().unsqueeze(-1))

  def predict_score(self, obs: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """`t` is expected in [0, 1] (n / num_steps), not the raw step index."""
    return self.diffusion_act(obs, x_t, self.time_emb_for(t, x_t.shape[0]))

  # --- SDAC-only helpers, on the variance-preserving schedule ---------------

  def diffuse(self, x: torch.Tensor, t: torch.Tensor):
    alpha_bar_t = self.alpha_bars[t].unsqueeze(-1)
    return alpha_bar_t.sqrt() * x + (1.0 - alpha_bar_t).sqrt() * torch.randn_like(x)

  def score_target(self, x0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    alpha_bar_t = self.alpha_bars[t].unsqueeze(-1)
    return -(x_t - alpha_bar_t.sqrt() * x0) / (1.0 - alpha_bar_t)

  def sample_phi0(self, x_t: torch.Tensor, t: torch.Tensor, K: int) -> torch.Tensor:
    alpha_bar_t = self.alpha_bars[t].unsqueeze(-1)
    mean = x_t / alpha_bar_t.sqrt()
    var  = (1.0 - alpha_bar_t) / alpha_bar_t
    eps  = torch.randn((K,) + x_t.shape, device=x_t.device)
    return mean.unsqueeze(0) + var.sqrt().unsqueeze(0) * eps

  # --- DIME path -----------------------------------------------------------

  def generate_action(self, obs: torch.Tensor, x: torch.Tensor = None, n_stop: int = 0):
    """Reverse chain a^N -> a^0 with the latent entropy lower bound (Eq. 15).

    Returns (x, ell_latent). CALLER MUST ADD the squash log-Jacobian."""
    B     = obs.shape[0]
    eta2  = self.eta ** 2                                   # (D,)
    scale = self.beta_scale                                 # (D,)

    if x is None:
      x = eta2.sqrt() * torch.randn((B, self.action_dim), device=obs.device)

    ell = -self.gaussian_logprob(x, torch.zeros_like(x), eta2.expand_as(x))

    # x holds a^n; each iteration maps a^n -> a^{n-1}
    for n in range(self.num_steps, n_stop, -1):
      tau = torch.tensor(n / self.num_steps, device=obs.device)

      score = self.predict_score(obs, x, tau).float()   # evaluated at the SOURCE state a^n
      if self.anchor_score:
        score = score - x / eta2

      b_rev = self.beta[n] * scale        # Eq. 12: reverse step sourced at n     (D,)
      b_fwd = self.beta[n - 1] * scale    # Eq. 11: forward step sourced at n-1   (D,)

      var_rev = (2.0 * eta2 * b_rev).expand_as(x)
      var_fwd = (2.0 * eta2 * b_fwd).expand_as(x)

      rev_mean = x + b_rev * x + 2.0 * eta2 * b_rev * score
      eps      = torch.randn_like(x)
      x_new    = rev_mean + var_rev.sqrt() * eps

      fwd_mean = x_new - b_fwd * x_new
      resid    = x - fwd_mean

      step = (0.5 * eps ** 2 - 0.5 * resid ** 2 / var_fwd).sum(-1, keepdim=True)
      ell  = ell + step + self.logvar_offset[n - 1]

      x = x_new

    return x, ell

  def sample(self, s: torch.Tensor, critic: Critic = None):
    """Raw, unsquashed x_0 -- the owning Policy squashes it."""
    act, _ = self.generate_action(s)
    return act

  def create_optimizer(self) -> dict:
    process = [p for p in (self._beta_scale_raw, self._eta_raw) if p.requires_grad]
    return {"actor": torch.optim.Adam(
        list(self.diffusion_act.parameters()) + process, lr=3e-4)}