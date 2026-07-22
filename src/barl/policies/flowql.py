import torch
from torch import nn
from torch.nn import functional as F

from barl.policy import Policy, Actor, Critic, HParam, register_policy
from barl.networks.mlp import MLP
from barl.networks.flow import sinusoidal_emb, midpoint_flow_sample
from barl.networks.critic import QCriticBase
from barl.networks.registry import register_component

import gymnasium as gym

# FlowQL implementation guided by https://arxiv.org/pdf/2502.02538


@register_component
class DistilledFlowActor(Actor, nn.Module):
  """Multi-step flow-matching generator (`act_flow`, BC-trained) distilled
  into a one-step network (`act_direct`, also pushed by the critic's value
  gradient) -- inference only ever runs `act_direct`."""

  def __init__(self, obs_dim: int, action_dim: int,
               latent_dim: int, flow_steps: int, time_dim: int, **kwargs):
    super().__init__(**kwargs)

    self.action_dim = action_dim
    self.flow_steps  = flow_steps
    self.time_dim    = time_dim

    self.act_flow   = MLP(obs_dim + action_dim + time_dim, action_dim, latent_dim, 4, layernorm=False)
    self.act_direct = MLP(obs_dim + action_dim, action_dim, latent_dim, 4)

  def step_flow(self, obs: torch.Tensor, action: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    with torch.amp.autocast("cuda", torch.bfloat16):
      return self.act_flow(obs, action, sinusoidal_emb(t, self.time_dim))

  def generate_flow_action(self, obs: torch.Tensor, z: torch.Tensor = None) -> torch.Tensor:
    return midpoint_flow_sample(self.step_flow, obs, self.action_dim, self.flow_steps, z=z)

  def sample(self, s: torch.Tensor, critic: Critic = None, z: torch.Tensor = None) -> torch.Tensor:
    B = s.shape[0]
    z = z if z is not None else torch.randn((B, self.action_dim), device=s.device)
    with torch.amp.autocast("cuda", torch.bfloat16):
      action = self.act_direct(s, z)
    return action.clamp(-1.0, 1.0)

  def create_optimizer(self) -> dict:
    return {
      "flow":       torch.optim.AdamW(self.act_flow.parameters(),   lr=3e-4, betas=(0.9, 0.95), weight_decay=1e-4),
      "direct_act": torch.optim.AdamW(self.act_direct.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=1e-4),
    }


@register_policy
class FlowQLPolicy(Policy):

  alpha = HParam(100.0, "Weight on distillation loss relative to the critic-gradient (DPG) loss", range=(0.1, 300.0))

  def __init__(self, env: gym.Env, **kwargs):
    super().__init__(env, **kwargs)

    assert isinstance(env.action_space, gym.spaces.Box)

    self.critic = self.build_component("critic", QCriticBase, self.observation_dim, self.action_dim, pessimistic_score=False)
    self.actor  = self.build_component("actor", DistilledFlowActor, self.observation_dim, self.action_dim)

  def create_optimizer(self):
    return {**self.critic.create_optimizer(), **self.actor.create_optimizer()}

  def train_step(self, batch: torch.DictType) -> tuple[dict, dict]:
    obs      = self.normalize_obs(batch["obs"].to(self.device))
    next_obs = self.normalize_obs(batch["next_obs"].to(self.device))
    action   = batch["act"].to(self.device)
    reward   = batch["rew"].to(self.device)
    term     = batch["term"].to(self.device)

    losses, log = {}, {}

    # No Polyak-averaged actor -- the live actor supplies the next-state action.
    # No MaxEnt actor either, so the bootstrap penalty is plain 0.
    with torch.no_grad():
      next_action = self.actor.sample(next_obs)

    self.critic.loss(obs, action, reward, term, next_obs, next_action, 0.0, log, losses)

    # Flow teacher: BC loss, ignores the critic entirely.
    x_0 = torch.randn_like(action)
    x_1 = action
    t   = torch.sigmoid(torch.randn((action.shape[0], 1), device=action.device))
    x_t = x_0 * (1 - t) + t * x_1

    flow_pred = self.actor.step_flow(obs, x_t, t)
    flow_loss = F.mse_loss(flow_pred.float(), (x_1 - x_0).float())

    z = torch.randn_like(action)
    with torch.amp.autocast("cuda", torch.bfloat16):
      action_pred = self.actor.act_direct(obs, z)

    action_flow  = self.actor.generate_flow_action(obs, z)
    distill_loss = F.mse_loss(action_pred.float(), action_flow.float().detach())

    # grad leakage is ok here
    q_val = self.critic.value(obs, action_pred)

    direct_act_q_loss = -q_val.float().mean()
    direct_act_loss   = direct_act_q_loss + self.alpha * distill_loss

    losses["flow"]       = flow_loss
    losses["direct_act"] = direct_act_loss
    log.update({
      "train/loss/act_flow":     flow_loss.detach(),
      "train/loss/act_direct":   direct_act_loss.detach(),
      "stats/distill_loss":      distill_loss.detach(),
      "stats/direct_act_q_loss": direct_act_q_loss.detach(),
      "stats/direct_q_val":      q_val.detach().mean(),
    })
    return losses, log

  @torch.inference_mode()
  def inference_step(self, observation: torch.Tensor, z: torch.Tensor = None):
    obs = self.normalize_obs(observation.to(self.device))
    return self.actor.sample(obs, z=z), {}
