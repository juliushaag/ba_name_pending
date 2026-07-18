import torch
from torch import nn
from torch.nn import functional as F

from barl.policy import Policy, Actor, Critic, HParam, register_policy
from barl.networks.mlp import MLP
from barl.networks.flow import sinusoidal_emb, midpoint_flow_sample
from barl.networks.critic import QCritic, q_bellman_loss

import gymnasium as gym

"""
Flow Q implementation guided by https://arxiv.org/pdf/2502.02538
"""


class DistilledFlowActor(nn.Module, Actor):
  """
  FlowQL-style actor: a multi-step flow-matching generator (`act_flow`)
  trained by BC against the buffer, distilled into a one-step network
  (`act_direct`) that is also pushed by the critic's value gradient (DPG).
  Inference only ever runs `act_direct` -- the flow net is a training-time
  teacher, never sampled from at rollout time. No update() -- FlowQLPolicy.
  train_step runs the flow-BC + distillation + DPG losses directly.
  """

  def __init__(self, obs_dim: int, action_dim: int, act_low: torch.Tensor, act_high: torch.Tensor,
               latent_dim: int = 512, flow_steps: int = 16, time_dim: int = 8):
    super().__init__()

    self.action_dim = action_dim
    self.flow_steps  = flow_steps
    self.time_dim    = time_dim

    self.register_buffer("act_low",  act_low.clone())
    self.register_buffer("act_high", act_high.clone())

    self.act_flow   = MLP(obs_dim + action_dim + time_dim, action_dim, latent_dim, 4, layernorm=False)
    self.act_direct = MLP(obs_dim + action_dim, action_dim, latent_dim, 4)

  def step_flow(self, obs: torch.Tensor, action: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    with torch.amp.autocast("cuda", torch.bfloat16):
      return self.act_flow(obs, action, sinusoidal_emb(t, self.time_dim))

  def generate_flow_action(self, obs: torch.Tensor, z: torch.Tensor = None) -> torch.Tensor:
    return midpoint_flow_sample(self.step_flow, obs, self.action_dim, self.act_low, self.act_high,
                                 self.flow_steps, z=z)

  def sample(self, s: torch.Tensor, critic: Critic = None, z: torch.Tensor = None) -> torch.Tensor:
    B = s.shape[0]
    z = z if z is not None else torch.randn((B, self.action_dim), device=s.device)
    with torch.amp.autocast("cuda", torch.bfloat16):
      action = self.act_direct(s, z)
    return action.clamp(self.act_low, self.act_high)

  def create_optimizer(self) -> dict:
    return {
      "flow":       torch.optim.AdamW(self.act_flow.parameters(),   lr=3e-4, betas=(0.9, 0.95), weight_decay=1e-4),
      "direct_act": torch.optim.AdamW(self.act_direct.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=1e-4),
    }


@register_policy
class FlowQLPolicy(Policy):

  gamma      = HParam(0.99,  "Discount factor",                                              range=(0.9, 0.9999))
  rho        = HParam(0.5,   "Pessimism weight on Q std (lower bound = mean - rho * std)",    range=(0.0, 5.0))
  polyak_tau = HParam(0.005, "EMA decay step size for the Q target (1 - polyak_tau = decay)", range=(0.001, 0.05))
  alpha      = HParam(100.0, "Weight on distillation loss relative to the critic-gradient (DPG) loss", range=(0.1, 300.0))

  def __init__(self, env: gym.Env, latent_dim: int = 512, max_steps: int = 16,
               n_models: int = 10, distributional: bool = False, num_quantiles: int = 25, **kwargs):
    super().__init__(env, **kwargs)

    assert isinstance(env.action_space, gym.spaces.Box)

    self.critic = QCritic(self.observation_dim, self.action_dim, hidden_dim=latent_dim, layers=4,
                           n_models=n_models, distributional=distributional, num_quantiles=num_quantiles,
                           polyak_tau=self.polyak_tau, rho=self.rho, pessimistic_score=False)
    self.actor  = DistilledFlowActor(self.observation_dim, self.action_dim, self.act_low, self.act_high,
                                      latent_dim=latent_dim, flow_steps=max_steps)

  def create_optimizer(self):
    return {**self.critic.create_optimizer(), **self.actor.create_optimizer()}

  def train_step(self, batch: torch.DictType) -> tuple[dict, dict]:
    obs      = self.normalize_obs(batch["obs"].to(self.device))
    next_obs = self.normalize_obs(batch["next_obs"].to(self.device))
    action   = batch["act"].to(self.device)
    reward   = batch["rew"].to(self.device)
    term     = batch["term"].to(self.device)

    self.critic.step_ema()

    """ Critic: Bellman regression. FlowQL doesn't Polyak-average the actor
    -- the live actor supplies the next-state action, matching the original
    (pre-split) implementation. """
    with torch.no_grad():
      next_action = self.actor.sample(next_obs)
      bootstrap   = self.critic.next_bootstrap(next_obs, next_action)

    q_loss = q_bellman_loss(self.critic, obs, action, reward, term, bootstrap, self.gamma)

    """ Actor: flow teacher (BC loss, ignores the critic entirely) """
    x_0 = torch.randn_like(action)
    x_1 = action
    t   = torch.sigmoid(torch.randn((action.shape[0], 1), device=action.device))
    x_t = x_0 * (1 - t) + t * x_1

    flow_pred = self.actor.step_flow(obs, x_t, t)
    flow_loss = F.mse_loss(flow_pred.float(), (x_1 - x_0).float())

    """ One-step distillation + critic-guided DPG """
    z = torch.randn_like(action)
    with torch.amp.autocast("cuda", torch.bfloat16):
      action_pred = self.actor.act_direct(obs, z)

    action_flow  = self.actor.generate_flow_action(obs, z)
    distill_loss = F.mse_loss(action_pred.float(), action_flow.float().detach())

    # grad leakage is ok here
    q_val = self.critic.value(obs, action_pred)

    direct_act_q_loss = -q_val.float().mean()
    direct_act_loss   = direct_act_q_loss + self.alpha * distill_loss

    log = {
      "train/loss/q":            q_loss.detach(),
      "train/loss/act_flow":     flow_loss.detach(),
      "train/loss/act_direct":   direct_act_loss.detach(),
      "stats/distill_loss":      distill_loss.detach(),
      "stats/direct_act_q_loss": direct_act_q_loss.detach(),
      "stats/direct_q_val":      q_val.detach().mean(),
    }
    return {"q": q_loss, "flow": flow_loss, "direct_act": direct_act_loss}, log

  @torch.inference_mode()
  def inference_step(self, observation: torch.Tensor, z: torch.Tensor = None):
    obs = self.normalize_obs(observation.to(self.device))
    return self.actor.sample(obs, z=z), {}
