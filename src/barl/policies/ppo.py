import torch
from torch import nn
from torch.nn import functional as F

from barl.policy import Policy, register_policy
from barl.networks.mlp import MLP
from barl.networks.actor import StochasticActor
from barl.networks.registry import register_component

import gymnasium as gym


@register_component
class PPOCritic(nn.Module):
  """Plain state-value baseline V(s) for PPO's GAE advantage estimator.
  Doesn't implement the Critic ABC -- every method there is shaped around a
  Q(s,a), none of which exists for a state-only value function."""

  def __init__(self, obs_dim: int, hidden_dim: int, layers: int):
    super().__init__()
    self.v_function = MLP(obs_dim, 1, hidden_dim, layers)

  def value(self, s: torch.Tensor) -> torch.Tensor:
    with torch.amp.autocast("cuda", torch.bfloat16):
      return self.v_function(s).float()

  def create_optimizer(self) -> dict:
    return {"value": torch.optim.Adam(self.v_function.parameters(), lr=1e-3)}


@register_policy
class PPOPolicy(Policy):
  """Clipped-surrogate PPO (Schulman et al., 2017), strictly on-policy.
  "adv"/"ret"/"log_prob" arrive as extras fields already computed by
  dataset.compute_gae() before train_step ever sees a batch."""

  on_policy = True

  def __init__(self, env: gym.Env, **kwargs):
    super().__init__(env, **kwargs)

    assert isinstance(env.action_space, gym.spaces.Box)

    params = self.resolve_params()
    self.gamma        = params["gamma"]
    self.lam          = params["lam"]
    self.clip_eps     = params["clip_eps"]
    self.entropy_coef = params["entropy_coef"]

    # PPOCritic takes obs_dim only (no action_dim; it's a state-value V(s), not a Q(s,a)).
    self.critic = self.build_component("critic", PPOCritic, self.observation_dim)
    self.actor  = self.build_component("actor", StochasticActor, self.observation_dim, self.action_dim)

  def create_optimizer(self):
    return {**self.critic.create_optimizer(), **self.actor.create_optimizer()}

  def train_step(self, batch):
    obs = self.normalize_obs(batch["obs"].to(self.device))
    act = batch["act"].float().to(self.device)

    old_log_prob = batch["log_prob"].to(self.device)
    adv          = batch["adv"].to(self.device)
    ret          = batch["ret"].to(self.device)

    with torch.amp.autocast("cuda", torch.bfloat16):
      v = self.critic.v_function(obs)
    v_loss = F.mse_loss(v.float().squeeze(-1), ret.float())

    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    dist         = self.actor.dist(obs)
    new_log_prob = self.actor.log_prob(dist, act)

    ratio = (new_log_prob - old_log_prob).exp()
    surr1 = ratio * adv
    surr2 = ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv

    policy_loss = -torch.min(surr1, surr2).mean()
    entropy     = dist.entropy().sum(dim=-1).mean()
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
    action     = self.actor.squash(raw_action)

    log_prob = self.actor.log_prob(dist, action)
    value    = self.critic.value(obs).squeeze(-1)

    return action, {"log_prob": log_prob, "value": value}
