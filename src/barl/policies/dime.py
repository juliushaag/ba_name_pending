import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from barl.policy import Policy, HParam, register_policy
from barl.networks.actor import ScoreDiscreteActor
from barl.networks.critic import QCriticBase
from barl.networks import bfloat16, float32
from barl.networks.squash import Squash

import gymnasium as gym


@register_policy
class DIMEPolicy(Policy):
  """Diffusion-policy MaxEnt actor-critic over a ScoreDiscreteActor's
  DDPM-style reverse process, with an entropy dual (`log_alpha`, auto-tuned
  against `target_entropy`). Unlike AMDP, needs no squash-Jacobian correction:
  `entropy` feeds in as a plain additive bonus/dual target, not a log-density."""

  target_entropy_scale = HParam(-4.0, "target_entropy = scale * action_dim", range=(-10.0, 10.0))

  def __init__(self, env: gym.Env, **kwargs):

    super().__init__(env, **kwargs)

    assert isinstance(env.action_space, gym.spaces.Box)

    self.critic = self.build_component("critic", QCriticBase, self.observation_dim, self.action_dim, pessimistic_score=True)
    self.actor  = self.build_component("actor", ScoreDiscreteActor, self.observation_dim, self.action_dim)

    self.squash = self.build_component("squash", Squash)

    self.log_alpha = nn.Parameter(torch.tensor(math.log(1.0)))
    self.target_entropy = self.target_entropy_scale * self.action_dim

  def on_epoch_start(self):
    self.critic.on_epoch_start()
    return super().on_epoch_start()

  def create_optimizer(self):
    return {**self.critic.create_optimizer(), **self.actor.create_optimizer(), "alpha" : torch.optim.Adam([self.log_alpha], lr=1e-3)}

  def clamp_log_alpha(self, lo: float = 1e-6, hi: float = 10.0):
    with torch.no_grad():
      self.log_alpha.clamp_(math.log(lo), math.log(hi))

  def train_step(self, batch) -> tuple[dict, dict]:

    obs      = self.normalize_obs(batch["obs"].to(self.device))
    next_obs = self.normalize_obs(batch["next_obs"].to(self.device))

    action   = batch["act"].to(self.device)
    reward   = batch["rew"].to(self.device)
    term     = batch["term"].to(self.device)

    self.clamp_log_alpha()

    alpha = self.log_alpha.exp().detach()

    losses, log = {}, {}

    with bfloat16():
      with torch.no_grad():
        next_action, entropy_penalty = self.actor.generate_action(next_obs)
        next_action = self.squash(next_action)

        entropy_bonus_val = (alpha * entropy_penalty).detach()

      self.critic.loss(obs, action, reward, term, next_obs, next_action, entropy_bonus_val, log, losses)

      raw_action, entropy = self.actor.generate_action(obs)
      action = self.squash(raw_action)

      with float32():
        q_new = self.critic.target_value(obs, action)
        dime_loss = (-q_new -alpha * entropy).mean()

        alpha_loss = (self.log_alpha * (entropy.detach() - self.target_entropy)).mean()
    
    losses["alpha"] = alpha_loss
    losses["actor"]  = dime_loss

    log.update({
      "train/loss/actor":           dime_loss.detach(),
      "train/loss/alpha":           alpha_loss.detach(),

      "stats/entropy":                  entropy_penalty.detach().mean(),
      "stats/alpha":                    alpha.detach()
    })
    return losses, log

  @torch.inference_mode()
  def inference_step(self, observation: torch.Tensor, step_info=None):
    obs = self.normalize_obs(observation.to(self.device))
    return self.squash(self.actor.sample(obs)), {}
