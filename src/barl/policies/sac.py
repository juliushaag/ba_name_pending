import math

import torch
from torch import nn

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

    self.critic = self.build_component("critic", QCriticBase, self.observation_dim, self.action_dim, pessimistic_score=False)
    self.actor  = self.build_component("actor", StochasticActor, self.observation_dim, self.action_dim)

    self.entropy_target = -float(self.action_dim)
    self.log_alpha = nn.Parameter(torch.tensor(0.0))

  def create_optimizer(self):
    return {
      **self.critic.create_optimizer(),
      **self.actor.create_optimizer(),
      "alpha": torch.optim.Adam([self.log_alpha], lr=3e-4),
    }

  def clamp_log_alpha(self, lo: float = 0.01, hi: float = 5.0):
    with torch.no_grad():
      self.log_alpha.clamp_(math.log(lo), math.log(hi))

  def rsample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reparameterized (action, log_prob), differentiable through obs. The
    tanh Jacobian correction needs the extra `act_scale` term since log_prob
    is the density of an affine-transformed variable, not of `action` itself."""
    dist = self.actor.dist(obs)
    # Hand-rolled, not dist.rsample(): its in-place .normal_() fill can't trace under vmap.
    x = dist.mean + dist.stddev * torch.randn_like(dist.mean)
    y = self.actor.squash(x)

    log_prob = dist.log_prob(x) - self.actor.squash.log_abs_det_jacobian(x, y) - torch.log(self.act_scale)
    return y, log_prob.sum(dim=-1, keepdim=True)

  def train_step(self, batch):
    obs      = self.normalize_obs(batch["obs"].to(self.device))
    next_obs = self.normalize_obs(batch["next_obs"].to(self.device))
    action   = batch["act"].to(self.device)
    reward   = batch["rew"].to(self.device)
    term     = batch["term"].to(self.device)

    self.clamp_log_alpha()

    losses, log = {}, {}
    alpha = self.log_alpha.exp()

    with torch.no_grad():
      next_action, next_log_prob = self.rsample(next_obs)
      entropy_bonus = -alpha * next_log_prob

    self.critic.loss(obs, action, reward, term, next_obs, next_action,
                      entropy_bonus, log, losses)

    new_action, log_prob = self.rsample(obs)

    q_new      = self.critic.target_value(obs, new_action)
    actor_loss = (alpha.detach() * log_prob - q_new).mean()

    alpha_loss = (alpha * (-log_prob.detach() - self.entropy_target)).mean()

    losses["actor"] = actor_loss
    losses["alpha"] = alpha_loss

    log.update({
      "train/loss/actor": actor_loss.detach(),
      "train/loss/alpha": alpha_loss.detach(),
      "stats/alpha":       alpha.detach(),
      "stats/log_prob":    log_prob.detach().mean(),
      "stats/q_new":       q_new.detach().mean(),
    })
    return losses, log

  @torch.inference_mode()
  def inference_step(self, observation: torch.Tensor, step_info=None):
    obs = self.normalize_obs(observation.to(self.device))
    return self.actor.sample(obs), {}
