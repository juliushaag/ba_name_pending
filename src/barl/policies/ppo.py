import torch
from torch import nn
from torch.nn import functional as F

from barl.policy import Policy, HParam, register_policy
from barl.networks.mlp import MLP
from barl.networks.actor import StochasticActor

import gymnasium as gym


class PPOCritic(nn.Module):
  """
  Plain state-value baseline V(s) for PPO's GAE advantage estimator.
  Doesn't implement the Critic ABC -- every method there is shaped around a
  Q(s,a) (value(s,a), score(s,a) = the a-gradient of a Q EBM), none of
  which exists for a state-only value function. No update() -- PPOPolicy.
  train_step regresses it against precomputed GAE returns directly.
  """

  def __init__(self, obs_dim: int, hidden_dim: int = 256, layers: int = 4):
    super().__init__()
    self.v_function = MLP(obs_dim, 1, hidden_dim, layers)

  def value(self, s: torch.Tensor) -> torch.Tensor:
    with torch.amp.autocast("cuda", torch.bfloat16):
      return self.v_function(s).float()

  def create_optimizer(self) -> dict:
    return {"value": torch.optim.Adam(self.v_function.parameters(), lr=1e-3)}


@register_policy
class PPOPolicy(Policy):
  """
  Clipped-surrogate PPO (Schulman et al., 2017) -- strictly on-policy
  (`on_policy = True`): scripts/online_learning.py checks that flag and
  clears the replay buffer after each rollout's training pass, instead of
  letting old-policy trajectories linger across rollouts the way every
  off-policy critic in this framework relies on.

  GAE advantages/returns are computed once per rollout over full
  trajectories in dataset.compute_gae, BEFORE SingleStepDataset shuffles
  them into single-step batches -- by the time train_step sees a batch,
  "adv"/"ret"/"log_prob" are just three more extras fields riding along
  per-transition, same as any other actor's extras.
  """

  on_policy = True

  gamma        = HParam(0.99, "Discount factor for GAE",              range=(0.9, 0.9999))
  lam          = HParam(0.95, "GAE lambda (bias/variance trade-off)", range=(0.0, 1.0))
  clip_eps     = HParam(0.2,  "PPO clipped-surrogate ratio bound",    range=(0.05, 0.5))
  entropy_coef = HParam(0.0,  "Weight on the entropy bonus",          range=(0.0, 0.1))

  def __init__(self, env: gym.Env, hidden_dim: int = 256, layers: int = 4, **kwargs):
    super().__init__(env, **kwargs)

    self.critic = PPOCritic(self.observation_dim, hidden_dim=hidden_dim, layers=layers)
    self.actor  = StochasticActor(self.observation_dim, self.action_dim, self.action_discrete,
                                   act_low=getattr(self, "act_low", None), act_high=getattr(self, "act_high", None),
                                   hidden_dim=hidden_dim, layers=layers)

  def create_optimizer(self):
    return {**self.critic.create_optimizer(), **self.actor.create_optimizer()}

  def train_step(self, batch):
    obs = self.normalize_obs(batch["obs"].to(self.device))
    act = batch["act"].long().to(self.device) if self.action_discrete else batch["act"].float().to(self.device)

    old_log_prob = batch["log_prob"].to(self.device)
    adv          = batch["adv"].to(self.device)
    ret          = batch["ret"].to(self.device)

    """ Critic: MSE against precomputed GAE returns """
    with torch.amp.autocast("cuda", torch.bfloat16):
      v = self.critic.v_function(obs)
    v_loss = F.mse_loss(v.float().squeeze(-1), ret.float())

    """ Actor: clipped surrogate """
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    dist         = self.actor.dist(obs)
    new_log_prob = self.actor.log_prob(dist, act)

    ratio = (new_log_prob - old_log_prob).exp()
    surr1 = ratio * adv
    surr2 = ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv

    policy_loss = -torch.min(surr1, surr2).mean()
    entropy     = dist.entropy().mean() if self.action_discrete else dist.entropy().sum(dim=-1).mean()
    actor_loss  = policy_loss - self.entropy_coef * entropy

    log = {
      "train/loss/value":  v_loss.detach(),
      "train/loss/policy": policy_loss.detach(),
      "stats/ratio_mean":  ratio.detach().mean(),
      "stats/entropy":     entropy.detach(),
      "stats/approx_kl":   (old_log_prob - new_log_prob).detach().mean(),
      "stats/clip_frac":   (ratio.detach().sub(1.0).abs() > self.clip_eps).float().mean(),
    }
    return {"value": v_loss, "actor": actor_loss}, log

  @torch.inference_mode()
  def inference_step(self, observation: torch.Tensor, step_info=None):
    obs = self.normalize_obs(observation.to(self.device))

    dist       = self.actor.dist(obs)
    raw_action = dist.sample()

    if self.action_discrete:
      action = raw_action.view(-1, 1).long()
    else:
      action = raw_action.clamp(self.act_low, self.act_high)

    log_prob = self.actor.log_prob(dist, action)
    value    = self.critic.value(obs).squeeze(-1)

    return action, {"log_prob": log_prob, "value": value}
