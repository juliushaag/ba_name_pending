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
  """AMDP-style continuous-time score-matching actor (Thilges et al. 2026),
  trained via direct-score regression by the owning Policy (AMDP). No
  squash -- generate_action()'s raw terminal state isn't a canonical action."""

  def __init__(self, obs_dim: int, action_dim: int, latent_dim: int, layers: int, max_steps: int, 
               sigma_min: float, sigma_max: float, tau_min: float, tau_max: float, layernorm=False, **kwargs):
    super().__init__(**kwargs)

    self.sigma_min, self.sigma_max = sigma_min, sigma_max
    self.tau_min,   self.tau_max   = tau_min,   tau_max

    self.action_dim = action_dim
    self.max_steps   = max_steps
    self.time_dim     = 8

    self.diffusion_act = MLP(obs_dim + action_dim + self.time_dim, action_dim, latent_dim, layers, layernorm=layernorm)

    taus = torch.linspace(self.tau_min, self.tau_max, self.max_steps + 1)
    self.register_buffer("taus", taus)

    sigmas = torch.tensor([self.sigma_fn(taus[i]) for i in range(self.max_steps)])
    self.register_buffer("sigma", sigmas)

  def sigma_fn(self, tau: torch.Tensor) -> torch.Tensor:
    return self.sigma_min * (self.sigma_max / self.sigma_min) ** tau

  def sigma_sq_int_fn(self, tau: torch.Tensor) -> torch.Tensor:
    return self.sigma_min ** 2 * ((self.sigma_max / self.sigma_min) ** (2 * tau) - 1) / (2 * math.log(self.sigma_max / self.sigma_min))

  def bound_control(self, u: torch.Tensor, X: torch.Tensor, C: float = 10.0) -> torch.Tensor:
    """Enforce the admissible-control growth condition ||u|| <= C(1+||X||) (paper Eq. 20)."""
    max_norm = C * (1.0 + X.norm(dim=-1, keepdim=True))
    u_norm   = u.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scale    = (max_norm / u_norm).clamp(max=1.0)
    return u * scale
  
  def generate_action(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (raw unsquashed terminal state X, entropy bonus from the path integral)."""
  
    X        = torch.zeros((obs.shape[0], self.action_dim), device=obs.device)
    path_int = torch.zeros((obs.shape[0], 1), device=obs.device)

    for i in range(self.max_steps):
      tau  = self.taus[i]
      dtau = self.taus[i + 1] - self.taus[i]

      # u = self.bound_control(self.diffusion_act(obs, X, sinusoidal_emb(tau.expand(obs.shape[0], 1), self.time_dim)), X)
      
      u = self.diffusion_act(obs, X, sinusoidal_emb(tau.expand(obs.shape[0], 1), self.time_dim))
      
      drift     = self.sigma[i] * u * dtau
      diffusion = self.sigma[i] * torch.sqrt(dtau) * torch.randn_like(u)

      X        = X + drift + diffusion

      path_int = path_int + ((u ** 2).sum(dim=-1) * dtau).unsqueeze(1)

    return X, math.log(2) * self.action_dim - 0.5 * path_int

  def sample(self, s: torch.Tensor, critic: Critic = None) -> torch.Tensor:
    """Raw, unsquashed -- not a valid action on its own; exists to satisfy
    the Actor contract for standalone use."""
    X, _ = self.generate_action(s)
    return X

  def create_optimizer(self) -> dict:
    return {"actor": torch.optim.Adam(self.diffusion_act.parameters(), lr=3e-4)}

@register_component
class ScoreDiscreteActor(Actor, nn.Module):
  """Discrete-time diffusion actor (Ho et al. 2020 DDPM): a fixed `num_steps`
  variance-preserving forward noising schedule with a score-prediction
  network (learns s_theta(x_t, t) ~= grad_x log p_t(x_t) directly, no
  eps -> score conversion), ancestral reverse sampling x_T ~ N(0, I) -> x_0.
  Counterpart to ScoreContinuousActor; same "raw, unsquashed output" contract."""

  def __init__(self, obs_dim: int, action_dim: int, latent_dim: int, layers: int, num_steps: int,
               noise_start: float, noise_end: float, **kwargs):
    super().__init__(**kwargs)

    self.noise_start, self.noise_end = noise_start, noise_end

    self.num_steps   = num_steps
    self.action_dim = action_dim
    self.time_dim     = 8

    self.diffusion_act = MLP(obs_dim + action_dim + self.time_dim, action_dim, latent_dim, layers, layernorm=False)

    s  = 0.008
    tt = torch.linspace(0, 1, num_steps + 1)
    f  = torch.cos((tt + s) / (1 + s) * math.pi / 2) ** 2
    g  = f / f[0]                                             # cosine-shaped, 1 -> 0

    alpha_bars = noise_end + (noise_start - noise_end) * g     # noise_start -> noise_end

    ratios = alpha_bars[1:] / alpha_bars[:-1]
    trans_betas = (1.0 - ratios).clamp(1e-5, 0.999)
    betas  = torch.cat([trans_betas[:1], trans_betas])         # length num_steps + 1
    alphas = torch.cat([torch.ones(1), ratios])                # kept index-aligned

    print("betas", betas.tolist())

    print("alphas", alphas.tolist())

    self.register_buffer("betas", betas)
    self.register_buffer("alphas", alphas)
    self.register_buffer("alpha_bars", alpha_bars)
    # eta (Eq. 11/12), softplus-parameterized to stay positive; init so softplus(raw) ~= 1.0.
    self._eta_raw = nn.Parameter(torch.full((action_dim,), math.log(math.e - 1.0)))

  @property
  def eta(self) -> torch.Tensor:
    return F.softplus(self._eta_raw)

  def gaussian_logprob(self, x, mean, var):
    # keepdim: entropy bonuses feed critic.loss's `penalty` as (B, 1) -- dropping it
    # would let `target_value() + penalty` silently outer-broadcast to (B, B).
    return (-0.5 * (x - mean) ** 2 / var - 0.5 * torch.log(2 * math.pi * var)).sum(-1, keepdim=True)

  def time_emb(self, t: torch.Tensor) -> torch.Tensor:
    return sinusoidal_emb(t, self.time_dim)
 
  def time_emb_for(self, t: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Normalizes t to shape (batch_size,) whether it arrives as a scalar
    (single diffusion step shared across the batch, as in generate_action)
    or already per-sample (B,), then embeds it."""
    if t.dim() == 0:
      t = t.expand(batch_size)
    return self.time_emb(t.float().unsqueeze(-1))

  def predict_score(self, obs: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """s_theta(x_t; s, t) ~= grad_{x_t} log p_t(x_t | s), predicted directly
    (no eps-prediction intermediary)."""
    t_emb = self.time_emb_for(t, x_t.shape[0])
    return self.diffusion_act(obs, x_t, t_emb)

  def diffuse(self, x: torch.Tensor, t: torch.Tensor):
    alpha_bar_t = self.alpha_bars[t].unsqueeze(-1)  # (B,) -> (B,1), broadcasts against (B, action_dim)
    return alpha_bar_t.sqrt() * x + (1.0 - alpha_bar_t).sqrt() * torch.randn_like(x)

  def score_target(self, x0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    alpha_bar_t = self.alpha_bars[t].unsqueeze(-1)  # (B,1), broadcasts against (..., B, action_dim)
    return -(x_t - alpha_bar_t.sqrt() * x0) / (1.0 - alpha_bar_t)

  def sample_phi0(self, x_t: torch.Tensor, t: torch.Tensor, K: int) -> torch.Tensor:
    """K candidates ~ phi_{0|t}(.|x_t) = N(x_t/sqrt(alpha_bar_t), (1-alpha_bar_t)/alpha_bar_t * I) (Ma et al. 2025, Eq. 15/24) """
    alpha_bar_t = self.alpha_bars[t].unsqueeze(-1)          # (B, 1)
    mean = x_t / alpha_bar_t.sqrt()                          # (B, action_dim)
    var  = (1.0 - alpha_bar_t) / alpha_bar_t                 # (B, 1)
    eps  = torch.randn((K,) + x_t.shape, device=x_t.device)  # (K, B, action_dim)
    return mean.unsqueeze(0) + var.sqrt().unsqueeze(0) * eps # (K, B, action_dim)
  
  def generate_action(self, obs: torch.Tensor, x: torch.Tensor = None, n_stop: int = 0):
    B = obs.shape[0]
    x = torch.randn((B, self.action_dim), device=obs.device) if x is None else x

    ell = -self.gaussian_logprob(x, torch.zeros_like(x), torch.ones_like(x))
    eta = self.eta

    # x holds a^n; each iteration maps a^n -> a^{n-1}
    for n in range(self.num_steps, n_stop, -1):
      n_t   = torch.tensor(n, device=obs.device)
      score = self.predict_score(obs, x, n_t).float()   # grad log pi_n, evaluated at the SOURCE state a^n

      beta_rev = self.betas[n]       # Eq. 12: reverse step sourced at n
      beta_fwd = self.betas[n - 1]   # Eq. 11: forward step sourced at n-1

      var_rev = (2.0 * eta ** 2 * beta_rev).expand_as(x)
      var_fwd = (2.0 * eta ** 2 * beta_fwd).expand_as(x)

      rev_mean = x + beta_rev * x + 2.0 * (eta ** 2) * beta_rev * score
      x_new    = rev_mean + var_rev.sqrt() * torch.randn_like(x)
      logp_rev = self.gaussian_logprob(x_new, rev_mean, var_rev)

      fwd_mean = x_new - beta_fwd * x_new
      logp_fwd = self.gaussian_logprob(x, fwd_mean, var_fwd)

      ell = ell + (logp_fwd - logp_rev)
      x = x_new

    return x, ell
 
  def sample(self, s: torch.Tensor, critic: Critic = None):
    """Raw, unsquashed x_0 -- the owning Policy squashes it."""
    act, _ = self.generate_action(s)
    return act
 
  def create_optimizer(self) -> dict:
    return {"actor": torch.optim.Adam(
        list(self.diffusion_act.parameters()) + [self._eta_raw], lr=3e-4
    )}