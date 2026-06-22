import math

import torch
from torch.nn import functional as F

from barl.networks.mlp import MLP
from barl.networks.ema import EMA
from barl.networks.ensemble import Ensemble, ReduceMethod

import gymnasium as gym

from barl.policy import FlowPolicy, Policy, HParam, register_policy

"""
Flow Q implementation guided by https://arxiv.org/pdf/2502.02538
"""


@register_policy
class QAMPolicy(FlowPolicy):

  gamma      = HParam(0.99, "Discount factor",                                           range=(0.9, 0.9999))
  tau        = HParam(0.2,  "Scale for terminal adjoint from Q gradient",              range=(0.0, 5.0))
  rho        = HParam(0.5,  "Pessimism weight on Q std (lower bound = mean - rho * std)", range=(0.0, 5.0))
  alpha      = HParam(1.0,  "Weight on distillation loss relative to Q loss",             range=(0.0, 10.0))
  polyak_tau = HParam(0.005,"EMA decay step size for Q target (1 - polyak_tau = decay)",  range=(0.001, 0.05))
  max_steps  = HParam(10,   "Euler steps for flow ODE at inference",                      range=(2, 100))

  def __init__(self, env: gym.Env, latent_dim: int = 512, time_dim : int = 8, flow_steps : int = 8, **kwargs):
    super().__init__(env, flow_steps, **kwargs)

    assert isinstance(env.action_space, gym.spaces.Box)

    self.latent_dim   = latent_dim
    self.time_dim = time_dim   
    self.q_function  = Ensemble(MLP(self.observation_dim + self.action_dim, 1, latent_dim, 4), 10)
    self.q_ema       = EMA(self.q_function, start_decay=1 - self.polyak_tau, end_decay=1 - self.polyak_tau, warmup_steps=1)

    self.act_flow    = MLP(self.observation_dim + self.action_dim + self.time_dim, self.action_dim, latent_dim, 4)

  def create_optimizer(self):
    return {
      "critic":          torch.optim.AdamW(self.q_function.parameters(),  lr=3e-4, betas=(0.9, 0.95), weight_decay=1e-4),
      "am":       torch.optim.AdamW(self.act_flow.parameters(),    lr=3e-4, betas=(0.9, 0.95), weight_decay=1e-4),
    }

  def train_step(self, batch: torch.DictType, frozen : FlowPolicy) -> torch.Tensor:

    bc_policy = frozen

    obs      = batch["obs"].to(self.device)
    next_obs = batch["next_obs"].to(self.device)
    action   = batch["act"].to(self.device)
    reward   = batch["rew"].to(self.device)
    term     = batch["term"].to(self.device)

    reward = reward.view(-1, 1)
    term   = term.view(-1, 1).to(reward.dtype)

    self.q_ema.step()

    log_dict = {}

    """ Critic """
    with torch.no_grad():
      
      next_action  = self.inference_step(next_obs)

      q_mean, q_std = self.q_ema(next_obs, next_action, reduce=ReduceMethod.MEANSTD)
      target_q     = q_mean - self.rho * q_std
      target       = reward + (1.0 - term) * self.gamma * target_q

    q_preds = self.q_function(obs, action)
    q_loss  = F.mse_loss(q_preds.float(), target.float().expand(self.q_function.n_models, -1, -1))

    """ Adjoint Matching """
    
    at = torch.randn_like(action)
    h = 1 / self.max_steps

    xs = [at.detach().clone()]
    ts = list()
    for i in range(self.max_steps):
      t = i / self.max_steps * torch.ones_like((obs[:, 0:1]))
      noise = torch.randn_like(at)
      sigma = torch.sqrt(2 * (1 - t + h) / (t + h))

      if i != self.max_steps - 1:
        v = self.step_flow(obs, at, t)
        at = at + h * (2 * v - at / (t + h)) + math.sqrt(h) * sigma * noise
      else:
        at = at + h * bc_policy.step_flow(obs, at, t)

      xs.append(at.detach().clone())
      ts.append(t)

    last_actions = xs[-1].requires_grad_(True)
    q_field = self.q_ema(obs, last_actions, reduce=ReduceMethod.MEAN).sum()
    q_field.backward()

    adj = -self.tau * last_actions.grad


    adjs = list()
    for i in reversed(range(self.max_steps)):
      t = i / self.max_steps * torch.ones_like(action[:, 0:1])

      def fn(xi):  
        return 2 * bc_policy.step_flow(obs, xi, t + h) - xi / (t + h)

      vjp = torch.func.vjp(fn, xs[i])[1](adj)[0]
      adj = adj + h * vjp
      adjs.append(adj)

    ts = torch.stack(ts, dim=0)
    xs = torch.stack(xs[:-1], dim=0)
    adjs = torch.stack(list(reversed(adjs)), dim=0)

    sigmas = torch.sqrt(2 * (1 - ts + h) / (ts + h))

    observations = obs[None, ...].repeat(self.max_steps, 1, 1)

    vf_fine = self.step_flow(observations, xs, ts)
    
    with torch.no_grad():
      vf_base = bc_policy.step_flow(observations, xs, ts)

    adj_loss = torch.sum(torch.square(2 * (vf_fine - vf_base) / sigmas + sigmas * adjs), dim=-1).mean(dim=0).mean()

    log_dict["stats/critic_loss"]        = q_loss.detach()
    log_dict["stats/q_max"]              = q_preds.detach().max()
    log_dict["stats/q_mean"]             = q_preds.detach().mean()
    log_dict["stats/q_std"]              = q_preds.detach().std()
    log_dict["stats/adj_mean"]           = adjs.detach().mean()
    log_dict["stats/sigma_mean"]         = sigmas.detach().mean()
    log_dict["train/loss/adjoint_matching_loss"]      = adj_loss.detach()
    log_dict["train/loss/q"]             = q_loss.detach()
    
    return {"am": adj_loss, "critic": q_loss}, log_dict

  def step_flow(self, obs, action, dt, inference = False):
    with torch.amp.autocast("cuda", torch.bfloat16):
      return self.act_flow(obs, action, self.sinusoidal_emb(dt, self.time_dim))