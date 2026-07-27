import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from barl.networks.regulator import AutotuneRegulator
from barl.policy import Policy, register_policy
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

    params = self.resolve_params()
    self.entropy_scale        = params["entropy_scale"]

    assert isinstance(env.action_space, gym.spaces.Box)

    self.critic : QCriticBase = self.build_component("critic", QCriticBase, self.observation_dim, self.action_dim, pessimistic_score=True)
    self.actor : ScoreDiscreteActor = self.build_component("actor", ScoreDiscreteActor, self.observation_dim, self.action_dim)

    self.squash : Squash = self.build_component("squash", Squash)

    self.entropy : AutotuneRegulator = self.build_component("entropy", AutotuneRegulator, target=self.action_dim * self.entropy_scale, name="entropy")

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

    self.critic.step()
    alpha = self.entropy.value()

    losses, log = {}, {}

    with bfloat16():

      next_X, entropy = self.actor.generate_action(next_obs)
      next_action = self.squash(next_X)
 
      entropy_bonus = (alpha * entropy).detach()
      self.critic.loss(obs, action, reward, term, next_obs, next_action, entropy_bonus, log, losses)

      q_new = self.critic.value(next_obs, next_action)

      with float32():
        dime_loss = -(q_new + (alpha * entropy)).mean()
        self.entropy.loss(entropy.detach(), log, losses)
    
    losses["actor"]  = dime_loss

    log.update({
      "stats/reward_mean" :             reward.mean(),
      "stats/act_abs" :                 next_action.abs().mean(),
    })

    return losses, log

  @torch.inference_mode()
  def inference_step(self, observation: torch.Tensor, step_info=None):
    obs = self.normalize_obs(observation.to(self.device))
    return self.squash(self.actor.sample(obs)), {}
