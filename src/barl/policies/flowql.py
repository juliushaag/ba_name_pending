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
class FlowQLPolicy(FlowPolicy):

  rho        = HParam(0.5,  "Pessimism weight on Q std (lower bound = mean - rho * std)", range=(0.0, 5.0))
  gamma      = HParam(0.99, "Discount factor",                                        range=(0.9, 0.9999))
  alpha      = HParam(100.0,  "Weight on distillation loss relative to Q loss",         range=(0.1, 300.0))
  polyak_tau = HParam(0.005, "EMA decay step size for Q target (1 - polyak_tau = decay)", range=(0.001, 0.05))

  def __init__(self, env: gym.Env, latent_dim: int = 512, max_steps : int = 16, **kwargs):
    super().__init__(env, max_steps, **kwargs)

    assert isinstance(env.action_space, gym.spaces.Box)
    
    self.time_dim = 8

    self.q_function  = Ensemble(MLP(self.observation_dim + self.action_dim, 1, latent_dim, 4), 10)
    self.q_ema       = EMA(self.q_function, start_decay=1 - self.polyak_tau, end_decay=1 - self.polyak_tau, warmup_steps=1)

    self.act_flow    = MLP(self.observation_dim + self.action_dim + self.time_dim, self.action_dim, latent_dim, 4, layernorm=False)
    self.act_direct  = MLP(self.observation_dim + self.action_dim, self.action_dim, latent_dim, 4)

  def create_optimizer(self):
    return {
      "q":          torch.optim.AdamW(self.q_function.parameters(),  lr=3e-4, betas=(0.9, 0.95), weight_decay=1e-4),
      "flow":       torch.optim.AdamW(self.act_flow.parameters(),    lr=3e-4, betas=(0.9, 0.95), weight_decay=1e-4),
      "direct_act": torch.optim.AdamW(self.act_direct.parameters(),  lr=3e-4, betas=(0.9, 0.95), weight_decay=1e-4),
    }
  
  def train_step(self, batch: torch.DictType, frozen = None) -> torch.Tensor:
    obs      = batch["obs"].to(self.device)
    next_obs = batch["next_obs"].to(self.device)
    action   = batch["act"].to(self.device)
    reward   = batch["rew"].to(self.device)
    term     = batch["term"].to(self.device)

    obs      = self.normalize_obs(obs)
    next_obs = self.normalize_obs(next_obs)
    
    reward   = reward.view(-1, 1)
    term     = term.view(-1, 1).to(reward.dtype)

    self.q_ema.step()

    log_dict = {}

    """ Critic """
    with torch.no_grad():
      z = torch.randn_like(action)
      next_action = self.generate_action(next_obs, z)

      with torch.amp.autocast("cuda", torch.bfloat16):
        q_mean, q_std = self.q_ema(next_obs, next_action, reduce=ReduceMethod.MEANSTD)
        target_q     = q_mean - self.rho * q_std
        target      = reward + (1.0 - term) * self.gamma * target_q

    with torch.amp.autocast("cuda", torch.bfloat16):
      q_preds = self.q_function(obs, action)
  
    q_loss  = F.mse_loss(q_preds.float(), target.float().expand(self.q_function.n_models, -1, -1))

    """ Flow actor """
    x_0    = torch.randn_like(action)
    x_1    = action
    t      = torch.sigmoid(torch.randn((action.shape[0], 1), device=self.device))
    x_t    = x_0 * (1 - t) + t * x_1

    flow_pred   = self.step_flow(obs, x_t, t)
  
    flow_loss   = F.mse_loss(flow_pred.float(), (x_1 - x_0).float())

    """ Direct actor """
    z            = torch.randn_like(action)
    
    with torch.amp.autocast("cuda", torch.bfloat16):
      action_pred  = self.act_direct(obs, z)

    action_flow  = self.generate_action(obs, z)

    distill_loss = F.mse_loss(action_pred.float(), action_flow.float().detach())


    self.q_function.requires_grad_(False)
    with torch.amp.autocast("cuda", torch.bfloat16):
      qs = self.q_function(obs, action_pred, reduce=ReduceMethod.MEAN)
    self.q_function.requires_grad_(True)

    direct_act_q_loss = -qs.float().mean()

    direct_act_loss   = direct_act_q_loss + self.alpha * distill_loss

    log_dict["stats/distill_loss"]       = distill_loss.detach().mean()
    log_dict["stats/direct_act_q_loss"]  = direct_act_q_loss.detach().mean()
    log_dict["stats/direct_q_val"]       = qs.detach().mean()

    log_dict["train/loss/act_flow"]      = flow_loss.detach()
    log_dict["train/loss/act_direct"]    = direct_act_loss.detach()
    log_dict["train/loss/q"]             = q_loss.detach()

    log_dict["stats/q_mean"]       = q_mean.detach().mean()
    log_dict["stats/q_std"]       = q_std.detach().mean()
    
    return {"flow": flow_loss, "q": q_loss, "direct_act": direct_act_loss}, log_dict

  @torch.no_grad()
  def inference_step(self, observation: torch.Tensor, z : torch.Tensor = None):
    B   = observation.shape[0]
    obs = self.normalize_obs(observation.to(self.device))

    with torch.amp.autocast("cuda", torch.bfloat16):
      z      = z if z is not None else torch.randn((B, self.action_dim), device=self.device)
      action = self.act_direct(obs, z)

    return action.clamp(self.act_low, self.act_high)

  def step_flow(self, obs, action, dt, inference = False):
    with torch.amp.autocast("cuda", torch.bfloat16):
      return self.act_flow(obs, action, self.sinusoidal_emb(dt, self.time_dim))