
import copy
import math

import einops
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.barl.models.resnet import SmallResNet
from src.barl.models.mlp import MLP

import gymnasium as gym


from src.barl.policies.policy import Policy
from src.barl.models.ensemble import Ensemble, ReduceMethod

"""
FLow Q implementation guided by
https://arxiv.org/pdf/2502.02538
"""

class FlowQLPolicy(Policy):

  def __init__(self, env : gym.Env, gamma : float, alpha : float, latent_dim : int = 512, max_flow_steps : int = 10):

    super().__init__(env)

    assert isinstance(env.action_space, gym.spaces.Box)

    self.alpha = alpha
    self.gamma = gamma
    self.latent_dim = latent_dim

    self.max_steps = max_flow_steps

    self.q_function  = Ensemble(MLP(self.observation_dim + self.action_dim, 1, latent_dim, 4), 2)
    self.q_target  = self.register_ema(self.q_function, start_decay=0.99)

    self.act_flow   = MLP(self.observation_dim + self.action_dim + self.latent_dim, self.action_dim, latent_dim, 4)

    self.act_direct = MLP(self.observation_dim + self.action_dim, self.action_dim, latent_dim, 4, layernorm=False)


  def create_optimizer(self): 
    return {
      "q" : torch.optim.AdamW(self.q_function.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=1e-4),
      "flow" : torch.optim.AdamW(self.act_flow.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=1e-4),
      "direct_act" : torch.optim.AdamW(list(self.act_direct.parameters()), lr=3e-4, betas=(0.9, 0.95), weight_decay=1e-4),
    }
  

  def sinusoidal_emb(self, t):          # t: (B,) or (B,1)
    t = t.view(-1)
    half = self.latent_dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
    args = t[:, None] * 1000.0 * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)
  

  @torch.no_grad()
  def generate_flow_action(self, state : torch.Tensor, action : torch.Tensor):

    t = torch.zeros((state.shape[0], 1), device=self.device)
    for i in range(self.max_steps):
      t.fill_(i / self.max_steps)
      action = action + self.act_flow(state, action,  self.sinusoidal_emb(t)) / self.max_steps

    return action
   
  def train_step(self, batch : torch.DictType) -> torch.Tensor:
    obs =       batch["obs"].to(self.device)
    next_obs=   batch["next_obs"].to(self.device)
    action=     batch["act"].to(self.device)
    reward=     batch["rew"].to(self.device)
    term=       batch["term"].to(self.device)

    obs = self.normalize_obs(obs)
    next_obs = self.normalize_obs(next_obs)

    reward = reward.view(-1, 1)
    term   = term.view(-1, 1).to(reward.dtype)

    self.update_ema()

    log_dict = dict()

    """ Train Critic Q """

    with torch.no_grad():
      
      z = torch.randn_like(action)
      next_action = self.generate_flow_action(next_obs, z)
      
      target_q = self.q_target(next_obs, next_action, reduce=ReduceMethod.MIN)
      
      target = reward  + (1.0 - term.to(reward.dtype)) * self.gamma * target_q

    q_preds = self.q_function(obs, action)

    q_loss = F.mse_loss(q_preds.float(), target.float().expand(2, -1, -1))
                        
    """ Train vector field (flow actor) """

    x_0 = torch.randn_like(action)
    x_1 = action
  
    t  = torch.sigmoid(torch.randn((action.shape[0], 1), device=self.device))
    x_t = x_0 * (1 - t) + t * x_1

    flow_pred = self.act_flow(obs, x_t, self.sinusoidal_emb(t))

    flow_target = (x_1 - x_0)
    flow_loss = F.mse_loss(flow_pred.float(), flow_target.float())

 
    """ Direct actor update """
    z = torch.randn_like(action)

    action_pred = self.act_direct(obs, z)

    action_flow = self.generate_flow_action(obs, z)

    distill_loss = F.mse_loss(action_pred.float(), action_flow.float().detach())


    q_val = self.q_target(obs, action_pred, reduce=ReduceMethod.MIN)
  
    lam = 1.0 / q_val.float().abs().mean().detach()
    
    direct_act_q_loss = -lam * q_val.float().mean()


    direct_act_loss = direct_act_q_loss + self.alpha * distill_loss


    log_dict["stats/distill_loss"]            = distill_loss.detach().mean()
    log_dict["stats/direct_act_q_loss"]       = direct_act_q_loss.detach().mean()
    log_dict["stats/direct_q_val"]            = q_val.detach().mean()
    
    log_dict["train/loss/act_flow"]     = flow_loss.detach()
    log_dict["train/loss/act_direct"]   = direct_act_loss.detach()
    log_dict["train/loss/q"]            = q_loss.detach()

    return {"flow": flow_loss, "q": q_loss, "direct_act": direct_act_loss}, log_dict
    
  @torch.inference_mode()
  def inference_step(self, observation: torch.Tensor):
    
    B = observation.shape[0]
    obs = observation.to(self.device)

    obs = self.normalize_obs(obs)

    with torch.amp.autocast("cuda", torch.bfloat16):
      z = torch.randn((B, self.action_dim), device=self.device)
      action = self.act_direct(obs, z) 

    return action.clamp(self.act_low, self.act_high)