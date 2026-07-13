import math

import copy
import torch
from torch.nn import functional as F

from barl.networks.mlp import MLP
from barl.networks.ema import EMA
from barl.networks.ensemble import Ensemble, ReduceMethod

import gymnasium as gym

from barl.policy import FlowPolicy, Policy, HParam, register_policy

@register_policy
class DAMPolicy(Policy):


  gamma          = HParam(0.99, "Discount factor", range=(0.9, 0.9999))
  tr_lambda_init = HParam(0.5,  "Trust Region lambda", range=(0.01, 5.0))
  tr_epsilon     = HParam(0.1,  "Trust Region KL bound", range=(0.01, 1.0))

  rho           = HParam(0.5,  "Pessimism weight on Q std (lower bound = mean - rho * std)", range=(0.0, 5.0))
  sigma_min     = HParam(0.01,  "Sampling-time diffusion coefficient", range=(0.01, 1.0))
  sigma_max     = HParam(2.5,  "Sampling-time diffusion coefficient", range=(0.01, 100.0))

  tau_min    = HParam(0.01, "Min tau values", range=(0.01, 0.5))
  tau_max    = HParam(1.0,  "Max tau values", range=(0.5, 1.0))
  
  polyak_tau = HParam(0.005, "EMA decay step size for Q target (1 - polyak_tau = decay)", range=(0.001, 0.05))

  def __init__(self, env: gym.Env, latent_dim: int = 512, max_steps : int = 10, **kwargs):
    super().__init__(env, **kwargs)

    assert isinstance(env.action_space, gym.spaces.Box)

    self.entropy_target = -1.0 * self.action_dim 

    self.max_steps = max_steps
    self.time_dim = 8

    self.log_alpha = torch.nn.Parameter(torch.tensor(math.log(0.1)))
    self.log_tr_lambda = torch.nn.Parameter(torch.tensor(math.log(self.tr_lambda_init)))
    
    self.q_function  = Ensemble(10, MLP, input_dim=self.observation_dim + self.action_dim, output_dim=1, latent_dim=latent_dim, hidden_layers=4)
    self.q_ema       = EMA(self.q_function, start_decay=1 - self.polyak_tau, end_decay=1 - self.polyak_tau, warmup_steps=1)

    self.diffusion_act    = MLP(self.observation_dim + self.action_dim + self.time_dim, self.action_dim, latent_dim, 4, layernorm=False)

    self.diffusion_act_old = copy.deepcopy(self.diffusion_act).eval().requires_grad_(False)

    # Precalculate schedules
    taus = torch.linspace(self.tau_min, self.tau_max, self.max_steps + 1)
    self.register_buffer("taus", taus)

    sigmas = torch.tensor([self.sigma_fn(self.taus[i]) for i in range(self.max_steps)])
    self.register_buffer("sigma", sigmas)

    # erf-squashing scale (Eq. 14): chosen so the log-Jacobian gradient term of the change of
    # variables exactly cancels, simplifying the AMDP loss (Remark E.3).
    sig_sq_one = self.sigma_sq_int_fn(torch.tensor(self.tau_max))
    self.register_buffer("erf_k", (2 * sig_sq_one) ** -0.5)

  def on_epoch_end(self):
    # Copy new to old policy
    for p_t, p_s in zip(self.diffusion_act_old.parameters(), self.diffusion_act.parameters()):
      p_t.data.copy_(p_s.data)
    for b_t, b_s in zip(self.diffusion_act_old.buffers(), self.diffusion_act.buffers()):
      b_t.copy_(b_s)

    return super().on_epoch_end()

  def sinusoidal_emb(self, t, time_dim):
    shape = t.shape[:-1]  # all dims except the trailing 1
    t = t.reshape(-1)
    half = time_dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
    args = t[:, None] * 1000.0 * freqs[None]
    time_emb = torch.cat([args.sin(), args.cos()], dim=-1)
    return time_emb.reshape(*shape, time_dim)

  def sigma_fn(self, tau : torch.Tensor):
    return self.sigma_min * (self.sigma_max / self.sigma_min) ** tau
  
  def sigma_sq_int_fn(self, tau : torch.Tensor):
    return self.sigma_min ** 2 * ((self.sigma_max / self.sigma_min) ** (2 * tau) - 1) / (2 * math.log(self.sigma_max / self.sigma_min))
  
  @torch.no_grad()
  def generate_action_raw(self, obs : torch.Tensor, stochastic : bool = True):

    X = torch.zeros((obs.shape[0], self.action_dim), device=self.device)

    path_int = torch.zeros((obs.shape[0], 1), device=self.device)

    for i in range(self.max_steps):

      tau = self.taus[i]
      dtau = self.taus[i+1] - self.taus[i]

      u = self.diffusion_act(obs, X, self.sinusoidal_emb(tau.expand(obs.shape[0], 1), self.time_dim))

      drift = u * dtau

      diffusion = torch.sqrt(dtau) * torch.randn_like(u)

      X = X + self.sigma[i] * (drift + diffusion)

      path_int = path_int + (u ** 2).sum(dim=-1, keepdim=True) * dtau

    return X, path_int


  def create_optimizer(self):
    return {
      "q":          torch.optim.Adam(self.q_function.parameters(),  lr=3e-4),
      "policy":     torch.optim.Adam(self.diffusion_act.parameters(),    lr=3e-4),
      "alpha" :     torch.optim.Adam([self.log_alpha],  lr=3e-4),
      "tr_lambda":  torch.optim.Adam([self.log_tr_lambda],  lr=3e-4, maximize=True),
    }
  
  def train_step(self, batch: torch.DictType, frozen = None) -> torch.Tensor:
    obs      = batch["obs"].to(self.device)
    next_obs = batch["next_obs"].to(self.device)
    action   = batch["act"].to(self.device)
    reward   = batch["rew"].to(self.device)
    term     = batch["term"].to(self.device)

    X = torch.erfinv(action.clamp(0.99, -0.99)) / self.erf_k

    obs      = self.normalize_obs(obs)
    next_obs = self.normalize_obs(next_obs)
    
    reward   = reward.view(-1, 1)
    term     = term.view(-1, 1).to(torch.float32)

    self.q_ema.step()

    alpha = self.log_alpha.exp()
    tr_lambda = self.log_tr_lambda.exp()

    log_dict = {}

    """ Critic """
    with torch.amp.autocast("cuda", torch.bfloat16):
      with torch.no_grad():
        
        act_raw, path_int = self.generate_action_raw(next_obs)

        target_q_mean, target_q_std = self.q_ema(next_obs, torch.erf(self.erf_k * act_raw), reduce=ReduceMethod.MEANSTD)
        target_q = target_q_mean - self.rho * target_q_std

        ent_loss = -1 / 2 * path_int + math.log(2)

        target = reward + (1.0 - term) * self.gamma * (target_q + alpha * ent_loss)
      
      q_preds = self.q_function(obs, action)
  
    q_loss  = F.mse_loss(q_preds.float(), target.float().expand(self.q_function.n_models, -1, -1))

    """ AMDP loss """

    tau = torch.rand((obs.shape[0],), device=self.device)

    sig_sq_tau = self.sigma_sq_int_fn(tau).unsqueeze(-1)

    sig_sq_one = self.sigma_sq_int_fn(torch.full((obs.shape[0],), self.tau_max, device=self.device)).unsqueeze(-1)

    X_t = X * (sig_sq_tau / sig_sq_one) + (sig_sq_tau - (sig_sq_tau ** 2 / sig_sq_one)).sqrt() * torch.randn_like(X)

    with torch.amp.autocast("cuda", torch.bfloat16):
      u = self.diffusion_act(obs, X_t, self.sinusoidal_emb(tau.view(obs.shape[0], 1), self.time_dim))

      X_grad = X.detach().requires_grad_(True)
      q_score_mean, q_score_std = self.q_function(obs, torch.erf(self.erf_k * X_grad), reduce=ReduceMethod.MEANSTD)
      q_score = q_score_mean - self.rho * q_score_std
      q_score.sum().backward()

    x_score = X_grad.grad.float()

    sigma_tau = self.sigma_fn(tau).unsqueeze(-1)

    amdp_target = sigma_tau * (1 / alpha.detach()) * x_score
    amdp_loss = ((u.float() - amdp_target) ** 2 / sigma_tau).mean()


    """ Trust region loss """

    with torch.amp.autocast("cuda", torch.bfloat16):
      u_old = self.diffusion_act_old(obs, X_t, self.sinusoidal_emb(tau.view(obs.shape[0], 1), self.time_dim))

    tr_loss = F.mse_loss(u.float(), u_old.float())


    policy_loss = amdp_loss + tr_lambda.detach() *  (tr_loss - self.tr_epsilon)
    tr_lambda_loss = tr_lambda * (tr_loss.detach() - self.tr_epsilon)

    """ Entropy target loss """


    # ent_loss is the entropy lower bound L_ENT(u) from Prop. 4.1 (can be negative), matching
    # the dual objective J(alpha) = E[alpha * (L_ENT(u) - target_entropy)] (Appendix E.6).
    entropy_target_loss = (alpha * (ent_loss.detach() - self.entropy_target)).mean()

    log_dict["train/loss/amdp"]             = amdp_loss.detach()
    log_dict["train/loss/q_loss"]             = q_loss.detach()
    log_dict["train/loss/tr_loss"]             = tr_loss.detach().mean()
    log_dict["train/loss/entropy"]             = ent_loss.detach().mean()
    log_dict["train/loss/policy_loss"]             = policy_loss.detach()
    log_dict["train/loss/entropy_target_loss"]             = entropy_target_loss.detach()

    log_dict["stats/q_mean"]        = q_preds.detach().float().mean()
    log_dict["stats/q_std"]         = q_preds.detach().float().std()
    log_dict["stats/q_max"]         = q_preds.detach().float().max()
    log_dict["stats/target_q_mean"] = target_q.detach().float().mean()
    log_dict["stats/target_mean"]   = target.detach().float().mean()
    log_dict["stats/tr_loss"]       = tr_loss.detach()
    log_dict["stats/alpha"]         = alpha.detach()
    log_dict["stats/tr_lambda"]     = tr_lambda.detach()
    log_dict["stats/score_norm"]    = x_score.detach().norm(dim=-1).mean()
    log_dict["stats/path_int_mean"] = path_int.detach().mean()

    return {
      "q":          q_loss,
      "policy":     policy_loss,
      "alpha" :     entropy_target_loss,
      "tr_lambda":  tr_lambda_loss,
    }, log_dict

  def inference_step(self, observation: torch.Tensor, stochastic: bool = True):

    B   = observation.shape[0]
    obs = self.normalize_obs(observation.to(self.device))

    with torch.no_grad(), torch.amp.autocast("cuda", torch.bfloat16):
      action_raw, path = self.generate_action_raw(obs, stochastic=stochastic)
      action = torch.erf(self.erf_k * action_raw)

    return action.clamp(self.act_low, self.act_high), {}

  def evaluate(self, env: gym.Env, runs=3):
    return super().evaluate(env, runs=runs, stochastic=False)