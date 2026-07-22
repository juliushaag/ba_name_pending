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

    self.actor = self.build_component("actor", DeterministicActor, self.observation_dim, self.action_dim)

  def create_optimizer(self):
    return self.actor.create_optimizer()

  def train_step(self, batch):
    obs     = self.normalize_obs(batch["obs"].to(self.device))
    actions = batch["act"].to(self.device)

    pred = self.actor(obs)
    loss = F.mse_loss(pred, actions.float())

    return {"actor": loss}, {"stats/pred_norm": pred.norm(dim=-1).mean().detach()}

  @torch.inference_mode()
  def inference_step(self, observation: torch.Tensor, step_info=None):
    obs = self.normalize_obs(observation.to(self.device))
    return self.actor.sample(obs), {}
