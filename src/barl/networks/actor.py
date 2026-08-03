import math

import torch
from torch import nn
import torch.nn.functional as F

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

  def __init__(self, env_stats, hidden_dim: int, layers: int, **kwargs):
    super().__init__(env_stats, **kwargs)

    self.net = MLP(self.obs_dim, self.action_dim, hidden_dim, layers)

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

  def __init__(self, env_stats, hidden_dim: int, layers: int, squash_cls: str, **kwargs):
    super().__init__(env_stats, **kwargs)

    self.log_std = nn.Parameter(torch.zeros(self.action_dim))
    self.squash  = get_component(squash_cls)(env_stats)

    self.net = MLP(self.obs_dim, self.action_dim, hidden_dim, layers)

  def dist(self, s: torch.Tensor) -> torch.distributions.Distribution:
    out = self.net(s)
    log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
    return torch.distributions.Normal(out.float(), log_std.exp())

  def log_prob(self, dist: torch.distributions.Distribution, action: torch.Tensor) -> torch.Tensor:
    """`action` is the squashed (canonical [-1, 1]) value; inverts the squash
    and applies the change-of-variables correction."""
    x = self.squash.inverse(action.clip(-0.99, 0.99))
    log_prob = dist.log_prob(x) - self.squash.log_abs_det_jacobian(x, action)
    return log_prob.sum(dim=-1)

  def sample(self, s: torch.Tensor, critic: Critic = None) -> torch.Tensor:
    dist = self.dist(s)
    # Same draw as dist.sample(), but that routes through torch.normal(loc,
    # scale), which costs ~14 ms regardless of batch size -- and this runs once
    # per env step in both rollout and eval. Matches rsample()'s form.
    action = dist.mean + dist.stddev * torch.randn_like(dist.mean)
    return self.squash(action)

  def mode(self, s: torch.Tensor) -> torch.Tensor:
    """Deterministic (mean) action -- IQL trains via AWR but deploys this, not sample()."""
    dist = self.dist(s)
    return self.squash(dist.mean)

  def create_optimizer(self) -> dict:
    return {"actor": torch.optim.Adam(self.parameters(), lr=3e-4, fused=True)}

@register_component
class ScoreContinuousActor(Actor, nn.Module):
  """AMDP continuous-time control (Thilges et al. 2026). The network predicts
  u_tilde with u = sigma(tau) * u_tilde, so each Euler step reduces to
  X += v_i * u_tilde + sqrt(v_i) * eps  with  v_i = \int_{tau_i}^{tau_{i+1}} sigma^2.
  Raw, unsquashed output -- the owning Policy applies the erf squash."""

  def __init__(self, env_stats, latent_dim: int, layers: int, max_steps: int,
               sigma_min: float, sigma_max: float, tau_min: float, tau_max: float,
               bound_c: float = 50.0, layernorm=False, **kwargs):

    super().__init__(env_stats, **kwargs)

    self.sigma_min, self.sigma_max = sigma_min, sigma_max
    self.tau_min,   self.tau_max   = tau_min,   tau_max
    self.max_steps, self.time_dim  = max_steps, 8
    self.bound_c   = bound_c
    self._log_r    = math.log(sigma_max / sigma_min)
    self._int_base = float(self._raw_int(torch.tensor(float(tau_min))))

    self.diffusion_act = MLP(self.obs_dim + self.action_dim + self.time_dim, self.action_dim,
                             latent_dim, layers, layernorm=layernorm)

    # u approx 0 at the beginning
    nn.init.zeros_(self.diffusion_act.layers[-1].weight)
    nn.init.zeros_(self.diffusion_act.layers[-1].bias)

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

  def sample_noised(self, X) -> torch.Tensor:

    
    tau = self.tau_min + torch.rand(X.shape[0], device=X.device) * (self.tau_max - self.tau_min)
    sig_sq_tau = self.sigma_sq_int_fn(tau).unsqueeze(-1)
    sig_sq_one = self.sigma_sq_int_fn(torch.full((X.shape[0],), self.tau_max, device=X.device)).unsqueeze(-1)

    # !!! Very important to sample X from the policy NOT from the batch !!!
    var = (sig_sq_tau - (sig_sq_tau ** 2 / sig_sq_one)).clamp_min(0.0)
    X_t = X * (sig_sq_tau / sig_sq_one) + var.sqrt() * torch.randn_like(X)

    return X_t, tau
  
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

        path_int = path_int + (v * (u_tilde ** 2).sum(-1)).unsqueeze(1)

      return X, math.log(2) * self.action_dim - 0.5 * path_int

  def sample(self, s: torch.Tensor, critic: Critic = None) -> torch.Tensor:
    """Raw, unsquashed -- not a valid action on its own; exists to satisfy
    the Actor contract for standalone use."""
    X, _ = self.generate_action(s)
    return X

  def create_optimizer(self) -> dict:
    return {"actor": torch.optim.Adam(self.diffusion_act.parameters(), lr=3e-4, fused=True)}

from abc import ABC, abstractmethod

class   DiscreteDiffusionActorBase(Actor, nn.Module, ABC):
  """Shared skeleton for discrete-time diffusion actors."""

  def __init__(self, env_stats, latent_dim, layers, num_steps,
               time_dim=8, lr=3e-4, layernorm=False, **kwargs):
    super().__init__(env_stats, **kwargs)
    self.num_steps, self.time_dim, self.lr = num_steps, time_dim, lr
    self.diffusion_act = MLP(self.obs_dim + self.action_dim + time_dim, self.action_dim,
                             latent_dim, layers, layernorm=layernorm)
    self.register_buffer("tau_table", torch.arange(num_steps + 1, dtype=torch.float32) / num_steps)

  # ---- hooks -------------------------------------------------------------
  @abstractmethod
  def marginal_coef(self, n): # -> (gamma, nu)
    ...

  @abstractmethod
  def to_score(self, raw, x, n):
    ...

  @abstractmethod
  def prior_sample(self, B, dev):
    ...

  @abstractmethod
  def reverse_step(self, obs, x, n): # -> (x_prev, aux_inc)
    ...

  def init_aux(self, x): 
    return torch.zeros(x.shape[0], 1, device=x.device)

  def process_parameters(self): return []

  # ---- network -----------------------------------------------------------
  def _tau(self, n, B, device):
    # Indexing the device-side table, rather than building a tensor from the
    # python step index: `torch.tensor(float(n), device="cuda")` is a pageable
    # H2D copy costing ~12x the MLP forward it feeds, once per reverse step.
    tau = self.tau_table[n]
    return (tau.expand(B) if tau.dim() == 0 else tau).unsqueeze(-1)

  def predict_raw(self, obs, x, n):
    emb = sinusoidal_emb(self._tau(n, x.shape[0], x.device), self.time_dim)
    return self.diffusion_act(obs, x, emb)

  def predict_score(self, obs, x, n):
    return self.to_score(self.predict_raw(obs, x, n).float(), x, n)

  # ---- derived forward API (RSM / any DSM loss) ---------------------------
  def diffuse(self, x0, n):
    g, nu = self.marginal_coef(n)
    return g * x0 + nu.sqrt() * torch.randn_like(x0)

  def sample_phi0(self, x_n, n, K):
    g, nu = self.marginal_coef(n)
    eps = torch.randn((K,) + x_n.shape, device=x_n.device)
    return (x_n / g).unsqueeze(0) + (nu.sqrt() / g).unsqueeze(0) * eps

  def score_target(self, x0, x_n, n):
    g, nu = self.marginal_coef(n)
    return -(x_n - g * x0) / nu

  def dsm_weight(self, n):
    return self.marginal_coef(n)[1]

  # ---- shared reverse loop -----------------------------------------------
  def generate(self, obs, x=None, n_stop=0):
    if x is None: x = self.prior_sample(obs.shape[0], obs.device)
    aux = self.init_aux(x)
    for n in range(self.num_steps, n_stop, -1):
      x, inc = self.reverse_step(obs, x, n)
      aux = aux + inc
    return x, aux

  def sample(self, s, critic=None):
    return self.generate(s)[0]

  def create_optimizer(self):
    params = list(self.diffusion_act.parameters()) + list(self.process_parameters())
    return {"actor": torch.optim.Adam(params, lr=self.lr, fused=True)}