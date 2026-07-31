import torch
from torch import nn
from torch.nn import functional as F

from barl.policy import Policy, Actor, Critic, register_policy
from barl.networks.mlp import MLP
from barl.networks.ema import EMA
from barl.networks.flow import sinusoidal_emb, midpoint_flow_sample
from barl.networks.registry import register_component

import gymnasium as gym


@register_component
class FlowBCActor(Actor, nn.Module):
  """Pure flow-matching BC actor. Inference samples through an EMA snapshot
  (`flow_ema`); the loss trains the live net (`flow`)."""

  def __init__(self, env_stats, latent_dim: int, flow_steps: int, time_dim: int,
               hidden_layers: int, **kwargs):
    super().__init__(env_stats, **kwargs)

    self.flow_steps  = flow_steps
    self.time_dim    = time_dim

    self.flow     = MLP(self.obs_dim + time_dim + self.action_dim, self.action_dim,
                        latent_dim, hidden_layers, layernorm=False)
    self.flow_ema = EMA(self.flow, start_decay=0.995, end_decay=0.995)

  def before_step(self):
    self.flow_ema.step()

  def step_flow(self, obs: torch.Tensor, action: torch.Tensor, t: torch.Tensor, inference: bool = False) -> torch.Tensor:
    net = self.flow_ema if inference else self.flow
    with torch.amp.autocast("cuda", torch.bfloat16):
      return net(obs, action, sinusoidal_emb(t, self.time_dim))

  def sample(self, s: torch.Tensor, critic: Critic = None, z: torch.Tensor = None) -> torch.Tensor:
    step_fn = lambda o, a, t: self.step_flow(o, a, t, inference=True)
    return midpoint_flow_sample(step_fn, s, self.action_dim, self.flow_steps, z=z)

  def create_optimizer(self) -> dict:
    return {"flow": torch.optim.AdamW(self.flow.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=1e-4, fused=True)}


@register_policy
class FlowBCPolicy(Policy):

  def __init__(self, env: gym.Env, **kwargs):
    super().__init__(env, **kwargs)

    self.actor = self.build_component("actor", FlowBCActor)

  def create_optimizer(self):
    return self.actor.create_optimizer()

  def train_step(self, batch):
    obs     = self.normalize_obs(batch["obs"].to(self.device))
    actions = batch["act"].to(self.device)


    B = actions.shape[0]

    z1 = actions  # already canonical [-1, 1] -- see ReplayBuffer.rollout()
    t  = torch.sigmoid(torch.randn((B, 1), device=obs.device))  # logit normal sampling
    z0 = torch.randn_like(z1)
    zt = t * z1 + (1 - t) * z0

    v_pred = self.actor.step_flow(obs, zt, t)
    loss = F.mse_loss(v_pred.float(), (z1 - z0).float())

    self.actor.metrics.vec("velocity", v_pred)

    return {"flow": loss}

  @torch.no_grad()
  def inference_step(self, observation: torch.Tensor, z: torch.Tensor = None):
    obs = self.normalize_obs(observation.to(self.device))
    return self.actor.sample(obs, z=z), {}
