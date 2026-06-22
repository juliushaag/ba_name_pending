import einops
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from barl.networks.resnet import SmallResNet
from barl.networks.mlp import MLP

import gymnasium as gym


from barl.policy import HParam, Policy, register_policy


@register_policy
class QPolicy(Policy):
 
  gamma       = HParam(0.99,  "Discount factor",                          range=(0.9, 0.9999))

  def __init__(self, env : gym.Env, latent_dim : int = 128, **kwargs):

    super().__init__(env, **kwargs)

    self.eps = 0.0
   
    self.obs_encoder = MLP(self.observation_dim, latent_dim, latent_dim, 0)

    self.layers = MLP(latent_dim + self.action_dim, 1, latent_dim, 3)
        

  def set_eps(self, eps : float):
    self.eps = eps


  def forward(self, obs : torch.Tensor, actions : torch.Tensor):
    
    obs_enc = self.obs_encoder(obs)
    
    action_oh = F.one_hot(actions, num_classes=self.action_dim).float().squeeze(1)
    
    output = self.layers(torch.cat([obs_enc, action_oh], dim=-1))

    return output
  
  def create_optimizer(self):
    return torch.optim.AdamW(self.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=1e-2)
   
  def train_step(self, batch : torch.DictType, target : "QPolicy") -> torch.Tensor:
    obs =       batch["obs"].to(self.device)
    next_obs=   batch["next_obs"].to(self.device)
    action=     batch["act"].to(self.device)
    reward=     batch["rew"].to(self.device)
    term=       batch["term"].to(self.device)

    with torch.amp.autocast("cuda", torch.bfloat16):
      with torch.no_grad():
        best_actions = self.inference_step(next_obs, eps_greedy=False)
        future = target(next_obs, best_actions).squeeze(1)
      
      tgt = reward + self.gamma * (1 - term.long()) * future
      pred = self(obs, action).squeeze(1)
    
    loss = F.huber_loss(pred.float(), tgt.float())

    return loss
    
  @torch.inference_mode()
  def inference_step(self, obs : torch.Tensor, eps_greedy : bool = True):

    obs = obs.to(self.device)
    
    if len(obs.shape) < 2:
      obs.unsqueeze_(0)

    B = obs.shape[0] # We may have more than one observation

    if (torch.rand((1,)).item() < self.eps) and eps_greedy:
      return torch.randint(0, self.action_dim, (B, 1), device=self.device)


    with torch.amp.autocast("cuda", torch.bfloat16):

      reshaped_obs = einops.repeat(obs, "b ... -> (b o) ...", o=self.action_dim)

      actions = einops.repeat(torch.arange(0, self.action_dim, device=self.device), "a -> (b a) 1", b=B)

      preds = self(reshaped_obs, actions)

      preds = einops.rearrange(
        preds,
        "(b o) 1 -> b o",
        o=self.action_dim
      )

      actions = preds.argmax(dim=1, keepdim=True)

    return actions

