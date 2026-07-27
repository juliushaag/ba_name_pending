import math

from barl.networks import bfloat16, float32
import torch
from torch import nn

from barl.networks.regulator import AutotuneRegulator
from barl.policy import Policy, register_policy
from barl.networks.actor import StochasticActor
from barl.networks.critic import QCriticBase

import gymnasium as gym


@register_policy
class SACPolicy(Policy):
  """Soft Actor-Critic (Haarnoja et al., 2018): entropy-regularized
  off-policy actor-critic, reparameterized tanh-Gaussian actor, automatic
  temperature tuning. Twin-critic pessimism reuses MeanStdEnsembleCritic's
  mean-minus-rho*std bound rather than an exact min(Q1,Q2)."""

  def __init__(self, env: gym.Env, **kwargs):
    super().__init__(env, **kwargs)

    assert isinstance(env.action_space, gym.spaces.Box)


    params = self.resolve_params()
    self.entropy_scale        = params["entropy_scale"]

    self.critic : QCriticBase = self.build_component("critic", QCriticBase, self.observation_dim, self.action_dim, pessimistic_score=False)
    self.actor : StochasticActor = self.build_component("actor", StochasticActor, self.observation_dim, self.action_dim)

    self.entropy : AutotuneRegulator = self.build_component("entropy", AutotuneRegulator, target=self.action_dim * self.entropy_scale, name="entropy")

  def on_epoch_start(self):
    self.critic.on_epoch_start()
    return super().on_epoch_start()

  def create_optimizer(self):
    return {
      **self.critic.create_optimizer(),
      **self.actor.create_optimizer(),
      **self.entropy.create_optimizer()
    }

  def rsample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reparameterized (action, log_prob), differentiable through obs. The
    tanh Jacobian correction needs the extra `act_scale` term since log_prob
    is the density of an affine-transformed variable, not of `action` itself."""
    dist = self.actor.dist(obs)
    # Hand-rolled, not dist.rsample(): its in-place .normal_() fill can't trace under vmap.
    x = dist.mean + dist.stddev * torch.randn_like(dist.mean)
    y = self.actor.squash(x)

    log_prob = dist.log_prob(x) - self.actor.squash.log_abs_det_jacobian(x, y) - torch.log(self.act_scale)
    return y, -log_prob.sum(dim=-1, keepdim=True)

  def train_step(self, batch):
    obs      = self.normalize_obs(batch["obs"].to(self.device))
    next_obs = self.normalize_obs(batch["next_obs"].to(self.device))
    action   = batch["act"].to(self.device)
    reward   = batch["rew"].to(self.device)
    term     = batch["term"].to(self.device)


    self.critic.step()
    losses, log = {}, {}
    alpha = self.entropy.value()

    

    with bfloat16():

      with torch.no_grad():
        next_action, next_entropy = self.rsample(next_obs)
        entropy_bonus = alpha * next_entropy

      new_action, entropy = self.rsample(obs)
      q_new      = self.critic.value(obs, new_action)

    with float32():

      self.critic.loss(obs, action, reward, term, next_obs, next_action, entropy_bonus, log, losses)
      self.entropy.loss(entropy.detach(), log, losses)

      actor_loss = -(q_new + (alpha * entropy)).mean() # Gradient descent therfore the minus


    losses["actor"] = actor_loss

    log["stats/reward_mean"]  = reward.mean()
    log["stats/act_abs"]      = action.abs().mean()  

    return losses, log

  @torch.inference_mode()
  def inference_step(self, observation: torch.Tensor, step_info=None):
    obs = self.normalize_obs(observation.to(self.device))
    return self.actor.sample(obs), {}
