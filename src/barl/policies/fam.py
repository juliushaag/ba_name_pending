import math

import torch
from torch.nn import functional as F

from barl.networks.mlp import MLP
from barl.networks.ema import EMA

import gymnasium as gym

from barl.policy import FlowPolicy, Policy, HParam, register_policy

"""
Freestyle policy, sadly does not work :(
"""

def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
  return torch.mean(torch.abs(tau - (u < 0).to(u.dtype)) * u.pow(2))


@register_policy
class FlowAdjointMatching(FlowPolicy):

  EXP_ADV_MAX = 100.0

  gamma      = HParam(0.99,  "Discount factor",                                           range=(0.9, 0.9999))
  tau        = HParam(0.7,   "Expectile for asymmetric value regression",                 range=(0.5, 0.99))
  beta       = HParam(3.0,   "Inverse temperature for advantage weighting",               range=(0.1, 20.0))
  polyak_tau = HParam(0.005, "EMA decay step size for Q target (1 - polyak_tau = decay)", range=(0.001, 0.05))

  def __init__(self, env: gym.Env, latent_dim: int = 256, hidden_layers: int = 6, flow_steps : int = 16, time_dim : int = 8,  **kwargs):
    super().__init__(env, flow_steps=flow_steps, **kwargs)

    self.max_steps = flow_steps
    self.time_dim = time_dim
    self.q_function  = MLP(self.observation_dim + self.action_dim, 1, latent_dim, hidden_layers)
    self.q_ema       = EMA(self.q_function, start_decay=1 - self.polyak_tau, end_decay=1 - self.polyak_tau, warmup_steps=1)

    self.v_function  = MLP(self.observation_dim, 1, latent_dim, hidden_layers)
    self.flow_actor  = MLP(self.observation_dim + self.action_dim + latent_dim, self.action_dim, latent_dim, hidden_layers)

  def create_optimizer(self):
    return {
      "value": torch.optim.AdamW(self.v_function.parameters(),   lr=3e-4, betas=(0.9, 0.95)),
      "q":     torch.optim.AdamW(self.q_function.parameters(),   lr=3e-4, betas=(0.9, 0.95)),
      "flow":  torch.optim.AdamW(self.flow_actor.parameters(),   lr=3e-4, betas=(0.9, 0.95)),
    }

  @torch.no_grad()
  def generate_flow_action(self, state: torch.Tensor, action: torch.Tensor):
    with torch.amp.autocast("cuda", torch.bfloat16):
      t = torch.zeros((state.shape[0], 1), device=self.device)
      for i in range(self.max_steps):
        t.fill_(i / self.max_steps)
        action = action + self.flow_actor(state, action, self.sinusoidal_emb(t)) / self.max_steps
    return action.clamp(self.act_low, self.act_high)

  def train_step(self, batch: torch.DictType) -> torch.Tensor:
    obs      = batch["obs"].to(self.device)
    next_obs = batch["next_obs"].to(self.device)
    action   = batch["act"].to(self.device)
    reward   = batch["rew"].to(self.device)
    term     = batch["term"].to(self.device)

    log_dict = {}
    reward   = reward.view(-1, 1)
    term     = term.view(-1, 1).to(reward.dtype)

    self.q_ema.step()

    """ Value function loss """
    with torch.amp.autocast("cuda", torch.bfloat16):
      with torch.no_grad():
        next_v   = self.v_function(next_obs)
        target_q = self.q_ema(obs, action)
      v = self.v_function(obs)

    adv    = target_q.float() - v.float()
    v_loss = asymmetric_l2_loss(adv, self.tau)

    """ Q function loss """
    with torch.amp.autocast("cuda", torch.bfloat16):
      qs = self.q_function(obs, action)

    targets = reward + (1.0 - term) * self.gamma * next_v
    q_loss  = F.mse_loss(qs.float(), targets.float())

    """ Flow actor loss """
    t      = torch.sigmoid(torch.randn((action.shape[0], 1), device=self.device))
    x_1    = action
    x_0    = torch.randn_like(action)
    x_t    = x_0 * (1 - t) + t * x_1

    flow_pred   = self.flow_actor(obs, x_t, self.sinusoidal_emb(t))
    flow_target = x_1 - x_0

    partial_action  = x_t.detach().requires_grad_(True)
    qflow_target    = self.q_ema(obs, partial_action)
    q_grad          = torch.autograd.grad(qflow_target.sum(), partial_action)[0]

    q_grad_normalized = q_grad / (q_grad.norm(dim=-1, keepdim=True) + 1e-8)
    qflow_target      = q_grad_normalized * flow_target.norm(dim=-1, keepdim=True)

    sched        = lambda x: 0.5 + 0.5 * x
    actor_target = flow_target * sched(1 - t) + qflow_target * t

    exp_adv  = torch.exp(self.beta * adv.detach()).clamp(max=FlowAdjointMatching.EXP_ADV_MAX).squeeze(1)
    flow_loss = (exp_adv * F.mse_loss(flow_pred.float(), actor_target.float(), reduction="none").mean(dim=-1)).mean()

    log_dict["stats/adv"]               = adv.detach().mean()
    log_dict["stats/targets"]           = targets.detach().mean()
    log_dict["stats/q_values"]          = qs.detach().mean()
    log_dict["stats/q_rel_err"]         = ((qs - targets).pow(2).mean().sqrt() / targets.abs().mean().clamp(min=1e-6)).detach()
    log_dict["stats/adv_std"]           = adv.detach().std()
    log_dict["stats/qflow_target_mean"] = qflow_target.detach().mean()
    log_dict["train/loss/flow"]         = flow_loss.detach()
    log_dict["train/loss/value"]        = v_loss.detach()
    log_dict["train/loss/q"]            = q_loss.detach()

    return {"value": v_loss, "q": q_loss, "flow": flow_loss}, log_dict

  @torch.inference_mode()
  def inference_step(self, obs):
    obs = obs.to(self.device)
    if obs.dim() < 2:
      obs = obs.unsqueeze(0)
    obs = self.normalize_obs(obs)
    z   = torch.randn((obs.shape[0], self.action_dim), device=self.device)
    return self.generate_flow_action(obs, z)

  def step_flow(self, obs, action, dt, inference = False):
    return self.flow_actor(obs, action, self.sinusoidal_emb(dt, self.time_dim))