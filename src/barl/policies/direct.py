import torch
from torch.nn import functional as F

from barl.policy import Policy, register_policy
from barl.networks.actor import DeterministicActor

import gymnasium as gym


@register_policy
class DirectPolicy(Policy):
  """Pure behavior cloning: a deterministic point-estimate actor regressed
  directly against buffer actions, no critic."""

  def __init__(self, env: gym.Env, **kwargs):
    super().__init__(env, **kwargs)

    self.actor = self.build_component("actor", DeterministicActor)

  def create_optimizer(self):
    return self.actor.create_optimizer()

  def train_step(self, batch):
    obs     = self.normalize_obs(batch["obs"].to(self.device))
    actions = batch["act"].to(self.device)

    pred = self.actor(obs)
    loss = F.mse_loss(pred, actions.float())

    self.actor.metrics.vec("pred", pred, limit=1.0)

    return {"actor": loss}

  @torch.no_grad()
  def inference_step(self, observation: torch.Tensor, step_info=None):
    obs = self.normalize_obs(observation.to(self.device))
    return self.actor.sample(obs), {}
