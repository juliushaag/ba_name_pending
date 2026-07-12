import math
import torch
from torch.nn import functional as F
from barl.policy import HParam, register_policy, FlowPolicy

from barl.networks.mlp import MLP
from barl.networks.ema import EMA
import gymnasium as gym


@register_policy
class FlowBCPolicy(FlowPolicy):

  def __init__(self, env: gym.Env, latent_dim=512, max_steps=16, hidden_layers=4, **kwargs):
    super().__init__(env, max_steps, **kwargs)
 
    self.time_dim = 8

    self.flow     = MLP(self.observation_dim + self.time_dim + self.action_dim, self.action_dim, latent_dim, hidden_layers, layernorm=False)
    self.flow_ema = EMA(self.flow, start_decay=0.995, end_decay=0.995)

  def create_optimizer(self):
    return { "flow": torch.optim.AdamW(self.flow.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=1e-4, fused=True) }
  
  def train_step(self, batch, frozen=None):
    obs     = self.normalize_obs(batch["obs"].to(self.device))
    actions = batch["act"].to(self.device)

    self.flow_ema.step()

    B = actions.shape[0]

    with torch.amp.autocast("cuda", torch.bfloat16):
      z1 = actions
      t  = torch.sigmoid(torch.randn((B, 1), device=self.device))  # logit normal sampling
      z0 = torch.randn_like(z1)
      zt = t * z1 + (1 - t) * z0

      v_pred = self.step_flow(obs, zt, t)

    loss = F.mse_loss(v_pred.float(), (z1 - z0).float())

    return {"flow": loss}, {"stats/v_norm": v_pred.norm(dim=-1).mean().detach(), "loss/flow": loss.detach()}


  def step_flow(self, obs, action, dt, inference = False):
    if inference:
      return self.flow_ema(obs, action, self.sinusoidal_emb(dt, self.time_dim))
    else:
      return self.flow(obs, action, self.sinusoidal_emb(dt, self.time_dim))