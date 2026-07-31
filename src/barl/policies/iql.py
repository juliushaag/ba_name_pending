import torch
from torch import nn
from torch.nn import functional as F

from barl.policy import Policy, Critic, register_policy
from barl.networks.mlp import MLP
from barl.networks.ema import EMA
from barl.networks.actor import StochasticActor
from barl.networks.registry import register_component

import gymnasium as gym

EXP_ADV_MAX = 100.0


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
  return torch.mean(torch.abs(tau - (u < 0).to(u.dtype)) * u.pow(2))


@register_component
class IQLCritic(Critic, nn.Module):
  """Expectile-regression critic (Kostrikov et al., IQL): V(s) regressed
  toward an asymmetric expectile of Q(s,a) from the same buffer transition
  -- never samples an action from an actor. `advantage()` is an IQL-specific
  extension beyond the base Critic ABC, for IQLPolicy's AWR weighting."""

  def __init__(self, env_stats, hidden_dim: int, layers: int,
               polyak_tau: float, gamma: float, **kwargs):
    super().__init__(env_stats, **kwargs)

    self.polyak_tau = polyak_tau
    self.gamma      = gamma

    self.q_function = MLP(self.obs_dim + self.action_dim, 1, hidden_dim, layers)
    self.v_function = MLP(self.obs_dim, 1, hidden_dim, layers)
    self.q_ema = EMA(self.q_function, start_decay=1 - self.polyak_tau, end_decay=1 - self.polyak_tau, warmup_steps=1)

  def before_step(self):
    self.q_ema.step()

  def value(self, s: torch.Tensor, a: torch.Tensor, target: bool = False) -> torch.Tensor:
    with torch.amp.autocast("cuda", torch.bfloat16):
      return (self.q_ema if target else self.q_function)(s, a)

  def score(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    # torch.func.grad, not a.requires_grad_(True) + eager .backward(), composes with vmap.
    a = a.detach()
    return torch.func.grad(lambda a: self.value(s, a).float().sum())(a)

  def state_value(self, s: torch.Tensor) -> torch.Tensor:
    """V(s), the expectile head -- IQL-specific, like `advantage()`: the Critic
    ABC only knows how to price an (s, a) pair."""
    with torch.amp.autocast("cuda", torch.bfloat16):
      return self.v_function(s)

  def advantage(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    return self.value(s, a, target=True).float() - self.state_value(s).float()

  def create_optimizer(self) -> dict:
    return {
      "value": torch.optim.AdamW(self.v_function.parameters(), lr=3e-4, betas=(0.9, 0.95), fused=True),
      "q":     torch.optim.AdamW(self.q_function.parameters(), lr=3e-4, betas=(0.9, 0.95), fused=True),
    }


@register_policy
class IQLPolicy(Policy):
  """Implicit Q-Learning (Kostrikov et al.): critic learns entirely off the
  buffer; actor is a plain advantage-weighted regression (AWR)."""

  def __init__(self, env: gym.Env, **kwargs):
    super().__init__(env, **kwargs)

    assert isinstance(env.action_space, gym.spaces.Box)

    params = self.resolve_params()
    self.tau  = params["tau"]
    self.beta = params["beta"]

    self.critic = self.build_component("critic", IQLCritic)
    self.actor  = self.build_component("actor", StochasticActor)

  def create_optimizer(self):
    return {**self.critic.create_optimizer(), **self.actor.create_optimizer()}

  def train_step(self, batch):
    obs      = self.normalize_obs(batch["obs"].to(self.device))
    next_obs = self.normalize_obs(batch["next_obs"].to(self.device))
    action   = batch["act"].to(self.device)
    reward   = batch["rew"].view(-1, 1).to(self.device)
    term     = batch["term"].view(-1, 1).to(self.device).to(reward.dtype)

    act_idx = action.float().view(action.shape[0], -1)
    act_enc = act_idx


    with torch.no_grad():
      next_v   = self.critic.state_value(next_obs)
      target_q = self.critic.value(obs, act_enc, target=True)
    v = self.critic.state_value(obs)

    adv    = target_q.float() - v.float()
    v_loss = asymmetric_l2_loss(adv, self.tau)

    qs = self.critic.value(obs, act_enc)

    targets = reward + (1.0 - term) * self.critic.gamma * next_v
    q_loss  = F.mse_loss(qs.float(), targets.float())

    awr_adv = self.critic.advantage(obs, act_enc).detach()
    exp_adv = torch.exp(self.beta * awr_adv).clamp(max=EXP_ADV_MAX).squeeze(1)

    dist      = self.actor.dist(obs)
    log_probs = self.actor.log_prob(dist, act_idx)
    act_loss  = torch.mean(exp_adv * -log_probs)

    self.metrics.dist("adv", adv)
    self.metrics.dist("exp_adv", exp_adv)
    self.critic.metrics.dist("q", qs)
    self.critic.metrics.dist("q_target", targets)
    self.critic.metrics.dist("v", v)
    self.critic.metrics.ema_gap("ema", self.critic.q_ema)
    self.actor.metrics.dist("log_prob", log_probs)

    return {"value": v_loss, "q": q_loss, "actor": act_loss}

  @torch.no_grad()
  def inference_step(self, obs: torch.Tensor, step_info=None):
    obs = obs.to(self.device)
    if obs.dim() < 2:
      obs = obs.unsqueeze(0)
    obs = self.normalize_obs(obs)
    return self.actor.mode(obs), {}
