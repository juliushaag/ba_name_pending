import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from barl.networks.entro import EntropyRegulator
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

  def __init__(self, env: gym.Env, **kwargs):

    super().__init__(env, **kwargs)

    assert isinstance(env.action_space, gym.spaces.Box)

    self.critic : QCriticBase = self.build_component("critic", QCriticBase, self.observation_dim, self.action_dim, pessimistic_score=True)
    self.actor : ScoreDiscreteActor = self.build_component("actor", ScoreDiscreteActor, self.observation_dim, self.action_dim)

    self.squash : Squash = self.build_component("squash", Squash)

    self.entropy : EntropyRegulator = self.build_component("entropy", EntropyRegulator, self.action_dim)

  def on_epoch_start(self):
    self.critic.on_epoch_start()
    return super().on_epoch_start()

  def create_optimizer(self):
    return {**self.critic.create_optimizer(), **self.actor.create_optimizer(), **self.entropy.create_optimizer()}

  def train_step(self, batch) -> tuple[dict, dict]:

    obs      = self.normalize_obs(batch["obs"].to(self.device))
    next_obs = self.normalize_obs(batch["next_obs"].to(self.device))

    action   = batch["act"].to(self.device)
    reward   = batch["rew"].to(self.device)
    term     = batch["term"].to(self.device)

    alpha = self.entropy.value()

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
        q_new = self.critic.value(obs, action)
        dime_loss = -(q_new + alpha * entropy).mean()

        self.entropy.loss(entropy.detach(), log, losses)
    
    losses["actor"]  = dime_loss

    return losses, log

  @torch.inference_mode()
  def inference_step(self, observation: torch.Tensor, step_info=None):
    obs = self.normalize_obs(observation.to(self.device))
    return self.squash(self.actor.sample(obs)), {}
