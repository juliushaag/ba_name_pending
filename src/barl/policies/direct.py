import torch
from torch.nn import functional as F

from barl.networks.mlp import MLP
from barl.networks.resnet import SmallResNet

from barl.policy import Policy, register_policy

import gymnasium as gym


@register_policy
class DirectPolicy(Policy):

  def __init__(self, env: gym.Env, latent_dim=128, hidden_layers=4):
    super().__init__(env)

    if len(self.observation_shape) == 3:
      self.obs_encoder = SmallResNet(latent_dim)
    elif len(self.observation_shape) == 1:
      self.obs_encoder = MLP(self.observation_shape[0], latent_dim, latent_dim, 0)

    self.layers = MLP(self.observation_dim, self.action_dim, latent_dim, hidden_layers)

  def create_optimizer(self):
    return {"direct": torch.optim.AdamW(self.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=1e-4, fused=True)}

  def forward(self, x: torch.Tensor):
    return self.layers(x)

  def train_step(self, batch):
    obs     = batch["obs"].to(self.device)
    actions = batch["act"].to(self.device)

    pred = self(obs)
    loss = F.mse_loss(pred, actions.float())

    return {"direct": loss}, {"stats/pred_norm": pred.norm(dim=-1).mean().detach()}

  @torch.inference_mode()
  def inference_step(self, observation, history=None):
    obs = self.normalize_obs(observation.to(self.device))
    return self(obs).clamp(self.act_low, self.act_high)
