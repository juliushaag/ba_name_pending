import math

import torch
from torch import nn
import torch.nn.functional as F

from barl.networks import bfloat16
from barl.networks.mlp import MLP
from barl.networks.flow import sinusoidal_emb
from barl.networks.registry import register_component, get_component

from barl.policy import Actor, Critic, HParam

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

  sigma_min = HParam(0.01, "Sampling-time diffusion coefficient (min)", range=(0.01, 1.0))
  sigma_max = HParam(2.5,  "Sampling-time diffusion coefficient (max)", range=(0.01, 100.0))
  tau_min   = HParam(0.01, "Min tau value", range=(0.01, 0.5))
  tau_max   = HParam(1.0,  "Max tau value", range=(0.5, 1.0))

  def __init__(self, obs_dim: int, action_dim: int,
               latent_dim: int, max_steps: int, **kwargs):
    super().__init__(**kwargs)

    self.action_dim = action_dim
    self.max_steps   = max_steps
    self.time_dim     = 8

    self.diffusion_act = MLP(obs_dim + action_dim + self.time_dim, action_dim, latent_dim, 4, layernorm=False)

    taus = torch.linspace(self.tau_min, self.tau_max, self.max_steps + 1)
    self.register_buffer("taus", taus)

    sigmas = torch.tensor([self.sigma_fn(taus[i]) for i in range(self.max_steps)])
    self.register_buffer("sigma", sigmas)

  def sigma_fn(self, tau: torch.Tensor) -> torch.Tensor:
    return self.sigma_min * (self.sigma_max / self.sigma_min) ** tau

  def sigma_sq_int_fn(self, tau: torch.Tensor) -> torch.Tensor:
    return self.sigma_min ** 2 * ((self.sigma_max / self.sigma_min) ** (2 * tau) - 1) / (2 * math.log(self.sigma_max / self.sigma_min))

  def generate_action(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (raw unsquashed terminal state X, entropy bonus from the path integral)."""
  
    X        = torch.zeros((obs.shape[0], self.action_dim), device=obs.device)
    path_int = torch.zeros((obs.shape[0], 1), device=obs.device)
    for i in range(self.max_steps):
      tau  = self.taus[i]
      dtau = self.taus[i + 1] - self.taus[i]

      net_out = self.diffusion_act(obs, X, sinusoidal_emb(tau.expand(obs.shape[0], 1), self.time_dim))
      u = net_out * self.sigma[i]

      drift     = u * dtau
      diffusion = self.sigma[i] * torch.sqrt(dtau) * torch.randn_like(u)

      X        = X + drift + diffusion
      # Girsanov control cost of drift `u` under base diffusion sigma*dW is 0.5*(u/sigma)^2 = 0.5*net_out^2.
      path_int = path_int + (net_out ** 2).sum(dim=-1, keepdim=True) * dtau

    return X, math.log(2) - 0.5 * path_int

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
  variance-preserving forward noising schedule with an epsilon-prediction
  network, ancestral reverse sampling x_T ~ N(0, I) -> x_0. Counterpart to
  ScoreContinuousActor; same "raw, unsquashed output" contract."""

  beta_min = HParam(0.01, "Sampling-time diffusion coefficient (min)", range=(1e-4, 1e-2))
  beta_max = HParam(0.02,  "Sampling-time diffusion coefficient (max)", range=(0.01, 0.1))
  
  def __init__(self, obs_dim: int, action_dim: int,
               latent_dim: int, num_steps: int, **kwargs):
    super().__init__(**kwargs)


    self.action_dim = action_dim
    self.num_steps   = num_steps
    self.time_dim     = 8
 
    self.diffusion_act = MLP(obs_dim + action_dim + self.time_dim, action_dim, latent_dim, 4, layernorm=False)

    betas      = torch.linspace(self.beta_min, self.beta_max, num_steps)
    alphas     = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
 
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
 
  def predict_noise(self, obs: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    t_emb = self.time_emb(t.float().expand(x_t.shape[0], 1))
    return self.diffusion_act(obs, x_t, t_emb)
 
  def generate_action(self, obs: torch.Tensor, x : torch.Tensor = None):
    """Differentiable Euler-Maruyama reverse sampling x_T ~ N(0, I) -> x_0
    (DIME Eq. 12, not ancestral DDPM) -- NOT wrapped in torch.no_grad(), the
    actor loss needs gradients through the whole chain. Returns (raw
    unsquashed x_0, entropy lower bound from Eq. 15)."""
    B = obs.shape[0]
    x = torch.randn((B, self.action_dim), device=obs.device) if x is None else x
 
    # terminal prior term: -log pi_N(x_T | s), x_T ~ N(0, I)
    ell = -self.gaussian_logprob(x, torch.zeros_like(x), torch.ones_like(x))
 
    eta = self.eta  # (action_dim,), broadcasts against (B, action_dim)
 
    for t in reversed(range(self.num_steps)):
      t_tensor = torch.tensor(t, device=obs.device)
      eps_hat  = self.predict_noise(obs, x, t_tensor).float()
 
      beta_t      = self.betas[t]                                   # plays the role of (beta_n * delta)
      alpha_bar_t = self.alpha_bars[t]
      score       = -eps_hat / torch.sqrt(1.0 - alpha_bar_t)         # eps -> score conversion
 
      var = 2.0 * eta ** 2 * beta_t                                 # (action_dim,) -> broadcasts to (B, action_dim)
      var = var.expand_as(x)
 
      # reverse (denoising) step actually taken, Eq. 12
      rev_mean = x + (beta_t * x + 2.0 * eta ** 2 * beta_t * score)
      noise    = torch.randn_like(x)
      x_new    = rev_mean + torch.sqrt(var) * noise
      logp_rev = self.gaussian_logprob(x_new, rev_mean, var)        # log pi_{n-1|n}(x_new | x)
 
      # matching forward (noising) transition density, Eq. 11, no score term
      fwd_mean = x_new - beta_t * x_new
      logp_fwd = self.gaussian_logprob(x, fwd_mean, var)            # log pi^_{n+1|n}(x | x_new)
 
      ell = ell + (logp_fwd - logp_rev)
      x = x_new
 
    return x, ell  # x_0, ell_pi(x_0, s)
 
  def sample(self, s: torch.Tensor, critic: Critic = None):
    """Raw, unsquashed x_0 -- the owning Policy squashes it."""
    act, _ = self.generate_action(s)
    return act
 
  def create_optimizer(self) -> dict:
    return {"actor": torch.optim.Adam(
        list(self.diffusion_act.parameters()) + [self._eta_raw], lr=3e-4
    )}