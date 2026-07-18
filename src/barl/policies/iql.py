import torch
from torch import nn
from torch.nn import functional as F

from barl.policy import Policy, Critic, HParam, register_policy
from barl.networks.mlp import MLP
from barl.networks.ema import EMA
from barl.networks.actor import StochasticActor

import gymnasium as gym

EXP_ADV_MAX = 100.0


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
  return torch.mean(torch.abs(tau - (u < 0).to(u.dtype)) * u.pow(2))


class IQLCritic(nn.Module, Critic):
  """
  Expectile-regression critic (Kostrikov et al., IQL): a state-value net
  V(s) is regressed toward an asymmetric expectile of Q(s,a) from the SAME
  buffer transition, so the critic never needs to sample an action from an
  actor at all. Exposes `advantage(s,a)` beyond the base Critic ABC for the
  AWR weighting IQLPolicy.train_step needs -- a plain Critic has nowhere to
  put V(s), so this is a genuine, documented IQL-specific extension.
  """

  def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256,
               layers: int = 6, polyak_tau: float = 0.005):
    super().__init__()

    self.q_function = MLP(obs_dim + action_dim, 1, hidden_dim, layers)
    self.v_function = MLP(obs_dim, 1, hidden_dim, layers)
    self.q_ema = EMA(self.q_function, start_decay=1 - polyak_tau, end_decay=1 - polyak_tau, warmup_steps=1)

  def step_ema(self):
    self.q_ema.step()

  def value(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    with torch.amp.autocast("cuda", torch.bfloat16):
      return self.q_function(s, a)

  def score(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    # torch.func.grad, not a.requires_grad_(True) + eager .backward() (see
    # QCritic.score()'s docstring) -- composes with vmap for multi-seed training.
    a = a.detach()
    return torch.func.grad(lambda a: self.value(s, a).float().sum())(a)

  def target_value(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    with torch.amp.autocast("cuda", torch.bfloat16):
      return self.q_ema(s, a)

  def advantage(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Q_target(s,a) - V(s), the AWR weighting signal IQLPolicy needs."""
    with torch.amp.autocast("cuda", torch.bfloat16):
      v = self.v_function(s)
    return self.target_value(s, a).float() - v.float()

  def create_optimizer(self) -> dict:
    return {
      "value": torch.optim.AdamW(self.v_function.parameters(), lr=3e-4, betas=(0.9, 0.95)),
      "q":     torch.optim.AdamW(self.q_function.parameters(), lr=3e-4, betas=(0.9, 0.95)),
    }


@register_policy
class IQLPolicy(Policy):
  """
  Implicit Q-Learning (Kostrikov et al.): critic learns entirely off the
  buffer (expectile regression, no actor-sampled next action); actor is a
  plain advantage-weighted regression (AWR) against buffer actions.
  """

  gamma      = HParam(0.99,  "Discount factor",                                              range=(0.9, 0.9999))
  tau        = HParam(0.7,   "Expectile for asymmetric value regression",                     range=(0.5, 0.99))
  polyak_tau = HParam(0.005, "EMA decay step size for the Q target (1 - polyak_tau = decay)", range=(0.001, 0.05))
  beta       = HParam(3.0,   "Inverse temperature for advantage weighting",                   range=(0.1, 20.0))

  def __init__(self, env: gym.Env, hidden_dim=256, layers=6, **kwargs):
    super().__init__(env, **kwargs)

    self.critic = IQLCritic(self.observation_dim, self.action_dim, hidden_dim=hidden_dim,
                             layers=layers, polyak_tau=self.polyak_tau)
    self.actor  = StochasticActor(self.observation_dim, self.action_dim, self.action_discrete,
                                   act_low=getattr(self, "act_low", None), act_high=getattr(self, "act_high", None),
                                   hidden_dim=hidden_dim, layers=layers)

  def create_optimizer(self):
    return {**self.critic.create_optimizer(), **self.actor.create_optimizer()}

  def train_step(self, batch):
    obs      = self.normalize_obs(batch["obs"].to(self.device))
    next_obs = self.normalize_obs(batch["next_obs"].to(self.device))
    action   = batch["act"].to(self.device)
    reward   = batch["rew"].view(-1, 1).to(self.device)
    term     = batch["term"].view(-1, 1).to(self.device).to(reward.dtype)

    if self.action_discrete:
      act_idx = action.long().view(-1)
      act_enc = F.one_hot(act_idx, self.action_dim).float()
    else:
      act_idx = action.float().view(action.shape[0], -1)
      act_enc = act_idx

    self.critic.step_ema()

    """ Critic: expectile V regression + plain Q regression, both off the buffer transition """
    with torch.amp.autocast("cuda", torch.bfloat16):
      with torch.no_grad():
        next_v   = self.critic.v_function(next_obs)
        target_q = self.critic.q_ema(obs, act_enc)
      v = self.critic.v_function(obs)

    adv    = target_q.float() - v.float()
    v_loss = asymmetric_l2_loss(adv, self.tau)

    with torch.amp.autocast("cuda", torch.bfloat16):
      qs = self.critic.q_function(obs, act_enc)

    targets = reward + (1.0 - term) * self.gamma * next_v
    q_loss  = F.mse_loss(qs.float(), targets.float())

    """ Actor: advantage-weighted regression (AWR) against buffer actions """
    awr_adv = self.critic.advantage(obs, act_enc).detach()
    exp_adv = torch.exp(self.beta * awr_adv).clamp(max=EXP_ADV_MAX).squeeze(1)

    dist      = self.actor.dist(obs)
    log_probs = self.actor.log_prob(dist, act_idx)
    act_loss  = torch.mean(exp_adv * -log_probs)

    log = {
      "train/loss/value": v_loss.detach(),
      "train/loss/q":     q_loss.detach(),
      "train/loss/actor": act_loss.detach(),
      "stats/adv":         adv.detach().mean(),
      "stats/adv_std":     adv.detach().std(),
      "stats/q_values":    qs.detach().mean(),
      "stats/targets":     targets.detach().mean(),
      "stats/exp_adv":     exp_adv.detach().mean(),
    }
    return {"value": v_loss, "q": q_loss, "actor": act_loss}, log

  @torch.inference_mode()
  def inference_step(self, obs: torch.Tensor, step_info=None):
    obs = obs.to(self.device)
    if obs.dim() < 2:
      obs = obs.unsqueeze(0)
    obs = self.normalize_obs(obs)
    return self.actor.mode(obs), {}
