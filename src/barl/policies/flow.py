import math
import torch
from torch.nn import functional as F
from src.barl.policies.policy import Policy

from src.barl.models.mlp import MLP
import gymnasium as gym

class FlowPolicy(Policy):
  def __init__(self, env : gym.Env, latent_dim=128, hidden_layers=4, max_steps=8):
    super().__init__(env)

    self.max_steps = max_steps

    self.flow = MLP(self.observation_dim + latent_dim + self.action_dim, self.action_dim, latent_dim, hidden_layers)

    self.target_flow = self.register_ema(self.flow)


  def create_optimizer(self):

    return { "flow" : torch.optim.AdamW(self.flow.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=1e-4, fused=True) }
  

  def sinusoidal_emb(self, t):          # t: (B,) or (B,1)
    t = t.view(-1)
    half = self.flow.latent_dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
    args = t[:, None] * 1000.0 * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)
  
  def forward(self, obs, actions, t):
    t_emb = self.sinusoidal_emb(t)
    return self.flow(obs, actions, t_emb)
    
  def train_step(self, batch):

    obs =       batch["obs"].to(self.device)
    actions =   batch["act"].to(self.device)

    self.update_ema()
    
    B = actions.shape[0]

    with torch.amp.autocast("cuda", torch.bfloat16):
      z1 = actions 
      t  = torch.sigmoid(torch.randn((B, 1), device=self.device)) # logit normal sampling
      z0 = torch.randn_like(z1)

      zt = t * z1 + (1 - t) * z0

      v_pred = self(obs, zt, t)
      v      = z1 - z0

    loss = F.mse_loss(v_pred.float(), v.float())

    return { "flow" : loss }, { "stats/v_norm": v_pred.norm(dim=-1).mean().detach(), "loss/flow" : loss.detach()}

  @torch.inference_mode()
  def inference_step(self, observation: torch.Tensor):
    
    B = observation.shape[0]
    obs = observation.to(self.device)

    obs = self.normalize_obs(obs)

    dt = 1.0 / self.max_steps
    with torch.amp.autocast("cuda", torch.bfloat16):
      z = torch.randn((B, self.action_dim), device=self.device)
      for i in range(self.max_steps):
        t = torch.full((B, 1), i * dt, device=self.device)
        v = self.target_flow(obs, z, self.sinusoidal_emb(t))                      # velocity at t
        z_mid = z + v * (dt / 2)
        v_mid = self.target_flow(obs, z_mid, self.sinusoidal_emb(t + dt / 2))  # velocity at midpoint
        z = z + v_mid * dt

    return z.clamp(self.act_low, self.act_high)