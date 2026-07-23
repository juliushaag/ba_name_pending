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
class SDAC(Policy):

  def __init__(self, env: gym.Env, **kwargs):

    super().__init__(env, **kwargs)

    assert isinstance(env.action_space, gym.spaces.Box)

    self.actor : ScoreDiscreteActor = self.build_component("actor", ScoreDiscreteActor, self.observation_dim, self.action_dim)
    self.critic : QCriticBase = self.build_component("critic", QCriticBase, self.observation_dim, self.action_dim, pessimistic_score=True)

    self.squash : Squash = self.build_component("squash", Squash)

    self.entropy : EntropyRegulator = self.build_component("entropy", EntropyRegulator)

    self.kl_div = None

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

      t = torch.randint(0, self.actor.num_steps, (obs.shape[0],), device=self.device)

      x_t = self.actor.diffuse(action, t)


      with torch.no_grad():
        action_rec, entropy_penalty = self.actor.generate_action(obs, x_t, t)


      qv = self.critic.value(obs, action_rec) / self.kl_div
      w = torch.softmax(qv / self.kl_div, dim=0) 


      sdac_loss = (w.exp() * (self.actor.predict_score(obs, x_t, t) - self.actor.score_target(action, x_t, t)) ** 2).mean()

    losses["actor"]  = sdac_loss

    return losses, log

  @torch.inference_mode()
  def inference_step(self, observation: torch.Tensor, step_info=None):
    obs = self.normalize_obs(observation.to(self.device))
    return self.squash(self.actor.sample(obs)), {}
