import math

import numpy as np
from torch.nn import functional as F

from src.barl.models.mlp import MLP
from src.barl.models.resnet import SmallResNet

from src.barl.policies.policy import Policy

import torch
from torch import nn
import gymnasium as gym


class DirectPolicy(Policy):

  def __init__(self, env : gym.Env, latent_dim=128, hidden_layers=4):
    super().__init__(env)

    if len(self.observation_shape) == 3: # This is an image
      self.obs_encoder = SmallResNet(latent_dim)
    elif len(self.observation_shape) == 1: 
      self.obs_encoder = MLP(self.observation_shape[0], latent_dim, latent_dim, 0)


    self.layers = MLP(self.observation_dim, self.action_dim, latent_dim, hidden_layers)


  def create_optimizer(self):

    return { "direct" : torch.optim.AdamW(self.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=1e-4, fused=True) }
  
  
  def forward(self, x : torch.Tensor):
    return self.layers(x)
  
  def train_step(self, batch):
    obs     = batch["obs"].to(self.device)
    actions = batch["act"].to(self.device)

    pred = self(obs)

    loss = F.mse_loss(pred, actions.float())

    return { "direct" : loss }, { "stats/pred_norm": pred.norm(dim=-1).mean().detach() }
  
  @torch.inference_mode()
  def inference_step(self, observation, history = None):

    obs = self.normalize_obs(observation.to(self.device))

    actions = self(obs)

    return actions.clamp(self.act_low, self.act_high)