
import copy

import einops
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.barl.models.resnet import SmallResNet
from src.barl.models.mlp import MLP

import gymnasium as gym


from src.barl.policies.policy import Policy

EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0



def asymmetric_l2_loss(u : torch.Tensor, tau : float) -> torch.Tensor:
  return torch.mean(torch.abs(tau - (u < 0).to(u.dtype)) * u.pow(2))

class IQLPolicy(Policy):

  def __init__(self, env : gym.Env, gamma : float, tau : float, beta : float, polyak_tau : float, latent_dim : int = 256):

    super().__init__(env)

    self.tau = tau
    self.beta = beta
    self.gamma = gamma
    self.polyak_tau = polyak_tau

    self.q_function = MLP(self.observation_dim + self.action_dim, 1, latent_dim, 6)

    self.v_function = MLP(self.observation_dim, 1, latent_dim, 6)

    self.actor = MLP(self.observation_dim, self.action_dim, latent_dim, 6)

    self.q_function_target = self.register_ema(self.q_function, start_decay=1-polyak_tau, end_decay=1-polyak_tau)



  def create_optimizer(self):
    return {
      "value" : torch.optim.AdamW(self.v_function.parameters(), lr=3e-4, betas=(0.9, 0.95)),
      "q" : torch.optim.AdamW(self.q_function.parameters(), lr=3e-4, betas=(0.9, 0.95)),
      "act" : torch.optim.AdamW(list(self.actor.parameters()) + ([self.actor_log_std] if not self.action_discrete else []), lr=3e-4, betas=(0.9, 0.95)),
    }


   
  def train_step(self, batch : torch.DictType) -> torch.Tensor:
    obs =       batch["obs"].to(self.device)
    next_obs=   batch["next_obs"].to(self.device)
    action=     batch["act"].to(self.device)
    reward=     batch["rew"].to(self.device)
    term=       batch["term"].to(self.device)

    log_dict = {}
    if self.action_discrete:
      a_idx = action.long().view(-1)                                 # (B,)
      a_in  = F.one_hot(a_idx, self.action_dim).float()
    else:
      action = action.float().view(action.shape[0], -1)              # (B, act_dim)
      a_in   = action

    reward = reward.view(-1, 1)
    term   = term.view(-1, 1).to(reward.dtype)

    self.update_ema()

    """ Value function loss """
    with torch.amp.autocast("cuda", torch.bfloat16):
      
      with torch.no_grad():
        
        next_v = self.v_function(next_obs)
        
        target_q = self.q_function_target(obs, a_in)
  
      v = self.v_function(obs)
    
    adv = target_q.float() - v.float()
    v_loss = asymmetric_l2_loss(adv, self.tau)

    """ Q function loss """
    with torch.amp.autocast("cuda", torch.bfloat16):
      qs = self.q_function(obs, a_in)

    targets = reward  + (1.0 - term.to(reward.dtype)) * self.gamma * next_v
    q_loss = F.mse_loss(qs.float(), targets.float())
 
    """ Actor update """
    exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX).squeeze(1)

    act_out = self.actor(obs)

    if self.action_discrete:
      dist = torch.distributions.Categorical(logits=act_out.float())
      bc_losses = -dist.log_prob(a_idx)                              # (B,)
    else:
      log_std = self.actor_log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
      dist = torch.distributions.Normal(act_out.float(), log_std.exp())
      bc_losses = -dist.log_prob(action).sum(dim=-1)

    act_loss = torch.mean(exp_adv * bc_losses)

    log_dict["stats/adv"]          = adv.detach().mean()
    log_dict["stats/exp_adv"]      = exp_adv.detach().mean()
    log_dict["stats/bc_losses"]    = bc_losses.detach().mean()
    log_dict["stats/targets"]      = targets.detach().mean()
    log_dict["stats/q_values"]     = qs.detach().mean()
    log_dict["stats/q_rel_err"] = ((qs - targets).pow(2).mean().sqrt()
                               / targets.abs().mean().clamp(min=1e-6)).detach()
    log_dict["stats/adv_std"]   = adv.detach().std()

    log_dict["train/loss/act"]     = act_loss.detach()
    log_dict["train/loss/value"]   = v_loss.detach()
    log_dict["train/loss/q"]       = q_loss.detach()

    return {"value": v_loss, "q": q_loss, "act": act_loss}, log_dict
    
  @torch.inference_mode()
  def inference_step(self, obs):
    
    obs = obs.to(self.device)
    if obs.dim() < 2:
      obs = obs.unsqueeze(0)

    obs = self.normalize_obs(obs)

    with torch.amp.autocast("cuda", torch.bfloat16):
      preds = self.actor(obs)

    if self.action_discrete:
      return preds.argmax(dim=1, keepdim=True)
    
    return preds.float().clamp(self.act_low, self.act_high)   # Gaussian mean
