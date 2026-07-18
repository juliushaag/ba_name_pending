import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from barl.policy import Policy, Critic, MaxEntActor, HParam, register_policy
from barl.networks.mlp import MLP
from barl.networks.flow import sinusoidal_emb
from barl.networks.critic import QCritic, q_bellman_loss

import gymnasium as gym


class FlowSDEActor(nn.Module, MaxEntActor):
  """
  AMDP-style actor (Thilges et al. 2026): forward SDE X0=0 -> X1 with an
  erf-squashed output, trained with a direct-score regression against the
  critic's pessimistic action-gradient.

  Carries its own entropy dual (log_alpha, MaxEnt) and trust-region dual
  (log_tr_lambda) regularizing against a hard-copied snapshot of itself,
  refreshed once per epoch via on_epoch_end() rather than Polyak-averaged.

  No update() -- DAMPolicy.train_step runs the AMDP loss, entropy-dual
  loss, and trust-region loss directly, calling critic.score() itself.
  """

  def __init__(self, obs_dim: int, action_dim: int, act_low: torch.Tensor, act_high: torch.Tensor,
               latent_dim: int = 512, max_steps: int = 10, tr_lambda_init: float = 0.5,
               sigma_min: float = 0.01, sigma_max: float = 2.5, tau_min: float = 0.01, tau_max: float = 1.0):
    super().__init__()

    self.action_dim     = action_dim
    self.entropy_target = -1.0 * action_dim
    self.max_steps       = max_steps
    self.time_dim         = 8

    self.sigma_min = sigma_min
    self.sigma_max = sigma_max
    self.tau_min   = tau_min
    self.tau_max   = tau_max

    self.register_buffer("act_low",  act_low.clone())
    self.register_buffer("act_high", act_high.clone())

    self.log_alpha     = nn.Parameter(torch.tensor(math.log(0.1)))
    self.log_tr_lambda = nn.Parameter(torch.tensor(math.log(tr_lambda_init)))

    self.diffusion_act     = MLP(obs_dim + action_dim + self.time_dim, action_dim, latent_dim, 4, layernorm=False)
    self.diffusion_act_old = copy.deepcopy(self.diffusion_act).eval().requires_grad_(False)

    taus = torch.linspace(self.tau_min, self.tau_max, self.max_steps + 1)
    self.register_buffer("taus", taus)

    sigmas = torch.tensor([self.sigma_fn(taus[i]) for i in range(self.max_steps)])
    self.register_buffer("sigma", sigmas)

    # erf-squashing scale (Eq. 14): chosen so the log-Jacobian gradient term of the change of
    # variables exactly cancels, simplifying the AMDP loss (Remark E.3).
    sig_sq_one = self.sigma_sq_int_fn(torch.tensor(self.tau_max))
    self.register_buffer("erf_k", (2 * sig_sq_one) ** -0.5)

  def on_epoch_end(self):
    # .copy_() directly on the tensor, not via .data: .data access is
    # disallowed under vmap (see networks/ema.py's identical fix), needed
    # for training several seeds at once through VmapPolicyGroup.
    for p_t, p_s in zip(self.diffusion_act_old.parameters(), self.diffusion_act.parameters()):
      p_t.copy_(p_s)
    for b_t, b_s in zip(self.diffusion_act_old.buffers(), self.diffusion_act.buffers()):
      b_t.copy_(b_s)

  def sigma_fn(self, tau: torch.Tensor) -> torch.Tensor:
    return self.sigma_min * (self.sigma_max / self.sigma_min) ** tau

  def sigma_sq_int_fn(self, tau: torch.Tensor) -> torch.Tensor:
    return self.sigma_min ** 2 * ((self.sigma_max / self.sigma_min) ** (2 * tau) - 1) / (2 * math.log(self.sigma_max / self.sigma_min))

  @torch.no_grad()
  def generate_action_raw(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    X        = torch.zeros((obs.shape[0], self.action_dim), device=obs.device)
    path_int = torch.zeros((obs.shape[0], 1), device=obs.device)

    with torch.amp.autocast("cuda", torch.bfloat16):
      for i in range(self.max_steps):
        tau  = self.taus[i]
        dtau = self.taus[i + 1] - self.taus[i]

        u = self.diffusion_act(obs, X, sinusoidal_emb(tau.expand(obs.shape[0], 1), self.time_dim))

        drift     = u * dtau
        diffusion = torch.sqrt(dtau) * torch.randn_like(u)

        X        = X + self.sigma[i] * (drift + diffusion)
        path_int = path_int + (u ** 2).sum(dim=-1, keepdim=True) * dtau

    return X, path_int

  def entropy_bonus(self, path_int: torch.Tensor) -> torch.Tensor:
    return -0.5 * path_int + math.log(2)

  def sample(self, s: torch.Tensor, critic: Critic = None) -> torch.Tensor:
    X, _ = self.generate_action_raw(s)
    return torch.erf(self.erf_k * X).clamp(self.act_low, self.act_high)

  def sample_with_entropy(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    X, path_int = self.generate_action_raw(s)
    action = torch.erf(self.erf_k * X).clamp(self.act_low, self.act_high)
    alpha  = self.log_alpha.exp()
    return action, (alpha * self.entropy_bonus(path_int)).detach()

  def create_optimizer(self) -> dict:
    return {
      "policy":    torch.optim.Adam(self.diffusion_act.parameters(), lr=3e-4),
      "alpha":     torch.optim.Adam([self.log_alpha],     lr=3e-4),
      "tr_lambda": torch.optim.Adam([self.log_tr_lambda], lr=3e-4, maximize=True),
    }


@register_policy
class DAMPolicy(Policy):

  gamma          = HParam(0.99,  "Discount factor",                                              range=(0.9, 0.9999))
  rho            = HParam(0.5,   "Pessimism weight on Q std (lower bound = mean - rho * std)",    range=(0.0, 5.0))
  polyak_tau     = HParam(0.005, "EMA decay step size for the Q target (1 - polyak_tau = decay)", range=(0.001, 0.05))
  tr_lambda_init = HParam(0.5,   "Trust region lambda (dual variable init)",  range=(0.01, 5.0))
  tr_epsilon     = HParam(0.1,   "Trust region KL/L2 bound",                  range=(0.01, 1.0))
  sigma_min      = HParam(0.01,  "Sampling-time diffusion coefficient (min)", range=(0.01, 1.0))
  sigma_max      = HParam(2.5,   "Sampling-time diffusion coefficient (max)", range=(0.01, 100.0))
  tau_min        = HParam(0.01,  "Min tau value",                            range=(0.01, 0.5))
  tau_max        = HParam(1.0,   "Max tau value",                            range=(0.5, 1.0))

  def __init__(self, env: gym.Env, latent_dim: int = 512, max_steps: int = 10,
               n_models: int = 10, distributional: bool = False, num_quantiles: int = 25, **kwargs):
    super().__init__(env, **kwargs)

    assert isinstance(env.action_space, gym.spaces.Box)

    self.critic = QCritic(self.observation_dim, self.action_dim, hidden_dim=latent_dim, layers=4,
                           n_models=n_models, distributional=distributional, num_quantiles=num_quantiles,
                           polyak_tau=self.polyak_tau, rho=self.rho, pessimistic_score=True)
    self.actor  = FlowSDEActor(self.observation_dim, self.action_dim, self.act_low, self.act_high,
                                latent_dim=latent_dim, max_steps=max_steps, tr_lambda_init=self.tr_lambda_init,
                                sigma_min=self.sigma_min, sigma_max=self.sigma_max,
                                tau_min=self.tau_min, tau_max=self.tau_max)

  def on_epoch_end(self):
    self.actor.on_epoch_end()
    return super().on_epoch_end()

  def create_optimizer(self):
    return {**self.critic.create_optimizer(), **self.actor.create_optimizer()}

  def train_step(self, batch) -> tuple[dict, dict]:
    obs      = self.normalize_obs(batch["obs"].to(self.device))
    next_obs = self.normalize_obs(batch["next_obs"].to(self.device))
    action   = batch["act"].to(self.device)
    reward   = batch["rew"].to(self.device)
    term     = batch["term"].to(self.device)

    self.critic.step_ema()

    """ Critic: soft-Q Bellman regression, entropy bonus folded into the bootstrap """
    with torch.no_grad():
      next_action, entropy_bonus_val = self.actor.sample_with_entropy(next_obs)
      bootstrap = self.critic.next_bootstrap(next_obs, next_action) + entropy_bonus_val

    q_loss = q_bellman_loss(self.critic, obs, action, reward, term, bootstrap, self.gamma)

    """ Actor: AMDP direct-score regression + entropy dual + trust region """
    alpha     = self.actor.log_alpha.exp()
    tr_lambda = self.actor.log_tr_lambda.exp()

    # Entropy-dual target: an independent SDE draw from the one used for the
    # critic's own soft-Q target above -- both are unbiased path-integral
    # estimates, trading a little variance for not threading a shared
    # intermediate value across the critic/actor loss boundary.
    with torch.no_grad():
      _, next_path_int = self.actor.generate_action_raw(next_obs)
    ent_loss = self.actor.entropy_bonus(next_path_int)

    # NB: original code did action.clamp(0.99, -0.99) (min > max), which
    # collapses every action to the constant -0.99 before erfinv -- fixed here.
    X = torch.erfinv(action.clamp(-0.99, 0.99)) / self.actor.erf_k

    """ AMDP loss: regress the vector field against sigma * (1/alpha) * dQ/dX """
    tau = torch.rand((obs.shape[0],), device=obs.device)

    sig_sq_tau = self.actor.sigma_sq_int_fn(tau).unsqueeze(-1)
    sig_sq_one = self.actor.sigma_sq_int_fn(torch.full((obs.shape[0],), self.tau_max, device=obs.device)).unsqueeze(-1)

    X_t = X * (sig_sq_tau / sig_sq_one) + (sig_sq_tau - (sig_sq_tau ** 2 / sig_sq_one)).sqrt() * torch.randn_like(X)

    with torch.amp.autocast("cuda", torch.bfloat16):
      u = self.actor.diffusion_act(obs, X_t, sinusoidal_emb(tau.view(obs.shape[0], 1), self.actor.time_dim))

    # d(erf(erf_k * X))/dX applied to the critic's action-space score: the
    # critic only ever needs to expose dQ/da (via score()), the erf change-of-
    # variables Jacobian is the actor's own business, not the critic's.
    X_detached = X.detach()
    a          = torch.erf(self.actor.erf_k * X_detached)
    da_dX      = self.actor.erf_k * (2 / math.sqrt(math.pi)) * torch.exp(-(self.actor.erf_k * X_detached) ** 2)
    x_score    = self.critic.score(obs, a).float() * da_dX

    sigma_tau = self.actor.sigma_fn(tau).unsqueeze(-1)

    amdp_target = sigma_tau * (1 / alpha.detach()) * x_score
    amdp_loss   = ((u.float() - amdp_target) ** 2 / sigma_tau).mean()

    """ Trust region loss """
    with torch.amp.autocast("cuda", torch.bfloat16):
      u_old = self.actor.diffusion_act_old(obs, X_t, sinusoidal_emb(tau.view(obs.shape[0], 1), self.actor.time_dim))

    tr_loss = F.mse_loss(u.float(), u_old.float())

    policy_loss    = amdp_loss + tr_lambda.detach() * (tr_loss - self.tr_epsilon)
    tr_lambda_loss = tr_lambda * (tr_loss.detach() - self.tr_epsilon)

    """ Entropy target loss """
    entropy_target_loss = (alpha * (ent_loss.detach() - self.actor.entropy_target)).mean()

    log = {
      "train/loss/q":                   q_loss.detach(),
      "train/loss/amdp":                amdp_loss.detach(),
      "train/loss/tr_loss":             tr_loss.detach().mean(),
      "train/loss/entropy":             ent_loss.detach().mean(),
      "train/loss/policy_loss":         policy_loss.detach(),
      "train/loss/entropy_target_loss": entropy_target_loss.detach(),
      "stats/alpha":                    alpha.detach(),
      "stats/tr_lambda":                tr_lambda.detach(),
      "stats/score_norm":               x_score.detach().norm(dim=-1).mean(),
    }
    return {"q": q_loss, "policy": policy_loss, "alpha": entropy_target_loss, "tr_lambda": tr_lambda_loss}, log

  @torch.inference_mode()
  def inference_step(self, observation: torch.Tensor, step_info=None):
    obs = self.normalize_obs(observation.to(self.device))
    return self.actor.sample(obs), {}
