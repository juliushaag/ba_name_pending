import einops
import torch
from torch import nn
from torch.nn import functional as F

from barl.policy import Policy, Actor, Critic, HParam, register_policy
from barl.networks.mlp import MLP
from barl.networks.ema import EMA
from barl.networks.registry import register_component

import gymnasium as gym


class _QNet(nn.Module):

  def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
    super().__init__()
    self.action_dim  = action_dim
    self.obs_encoder = MLP(obs_dim, hidden_dim, hidden_dim, 0)
    self.layers      = MLP(hidden_dim + action_dim, 1, hidden_dim, 3)

  def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    obs_enc   = self.obs_encoder(obs)
    action_oh = F.one_hot(actions, num_classes=self.action_dim).float().squeeze(1)
    return self.layers(torch.cat([obs_enc, action_oh], dim=-1))


@register_component
class DiscreteQCritic(Critic, nn.Module):
  """Q(s,a) over a discrete action set, action indices (not one-hot) in and
  out. `score()` is undefined -- no action gradient for a discrete space."""

  polyak_tau = HParam(0.005, "EMA decay step size for the Q target (1 - polyak_tau = decay)", range=(0.001, 0.05))
  gamma      = HParam(0.99,  "Discount factor",                                               range=(0.9, 0.9999))

  def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, **kwargs):
    super().__init__(**kwargs)

    self.action_dim = action_dim
    self.q_net = _QNet(obs_dim, action_dim, hidden_dim)
    self.q_ema = EMA(self.q_net, start_decay=1 - self.polyak_tau, end_decay=1 - self.polyak_tau, warmup_steps=1)

  def step_ema(self):
    self.q_ema.step()

  def value(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    with torch.amp.autocast("cuda", torch.bfloat16):
      return self.q_net(s, a)

  def score(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("DiscreteQCritic has no action gradient -- actions are discrete indices.")

  def target_value(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    with torch.amp.autocast("cuda", torch.bfloat16):
      return self.q_ema(s, a)

  def create_optimizer(self) -> dict:
    return {"q": torch.optim.AdamW(self.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=1e-2)}


@register_component
class GreedyDiscreteActor(Actor, nn.Module):
  """No generative net of its own -- eps-greedy argmax over whatever
  discrete critic it's handed; sample() requires one."""

  def __init__(self, action_dim: int, eps: float = 0.0, **kwargs):
    super().__init__(**kwargs)
    self.action_dim = action_dim
    self.eps = eps

  def set_eps(self, eps: float):
    self.eps = eps

  def sample(self, s: torch.Tensor, critic: Critic = None, eps_greedy: bool = True) -> torch.Tensor:
    if critic is None:
      raise ValueError("GreedyDiscreteActor.sample requires a critic to argmax over -- "
                        "it has no generative net of its own.")

    B = s.shape[0]

    with torch.amp.autocast("cuda", torch.bfloat16):
      reshaped_obs = einops.repeat(s, "b ... -> (b o) ...", o=self.action_dim)
      actions      = einops.repeat(torch.arange(0, self.action_dim, device=s.device), "a -> (b a) 1", b=B)
      values       = critic.value(reshaped_obs, actions)
      values       = einops.rearrange(values, "(b o) 1 -> b o", o=self.action_dim)

    greedy_action = values.argmax(dim=1, keepdim=True)
    if not eps_greedy:
      return greedy_action

    # Per-example coin flip, not per-batch -- also keeps this vmap-safe (torch.rand().item() isn't).
    random_action = torch.randint(0, self.action_dim, (B, 1), device=s.device)
    explore = torch.rand((B, 1), device=s.device) < self.eps
    return torch.where(explore, random_action, greedy_action)

  def create_optimizer(self) -> dict:
    return {}


@register_policy
class QPolicy(Policy):
  """Double-DQN: greedy action picked by the online critic, evaluated by
  the target/EMA net."""

  def __init__(self, env: gym.Env, **kwargs):
    super().__init__(env, **kwargs)

    assert self.action_discrete

    self.critic = self.build_component("critic", DiscreteQCritic, self.observation_dim, self.action_dim)
    self.actor  = self.build_component("actor", GreedyDiscreteActor, self.action_dim)  # action_dim only, no obs_dim

  def set_eps(self, eps: float):
    self.actor.set_eps(eps)

  def create_optimizer(self):
    return self.critic.create_optimizer()

  def train_step(self, batch):
    obs      = batch["obs"].to(self.device)
    next_obs = batch["next_obs"].to(self.device)
    action   = batch["act"].to(self.device)
    reward   = batch["rew"].view(-1, 1).to(self.device)
    term     = batch["term"].view(-1, 1).to(self.device).to(reward.dtype)

    self.critic.step_ema()

    with torch.no_grad():
      # Double-DQN-style decoupling: the greedy action is picked by the
      # *online* critic (self.critic, passed explicitly -- the actor has no
      # net of its own to pick with), and evaluated by the target/EMA net.
      best_action = self.actor.sample(next_obs, critic=self.critic, eps_greedy=False)
      target = reward + (1.0 - term) * self.critic.gamma * self.critic.target_value(next_obs, best_action)

    pred = self.critic.value(obs, action)
    loss = F.huber_loss(pred.float(), target.float())

    log = {
      "train/loss/q": loss.detach(),
      "stats/q_mean": pred.detach().float().mean(),
    }
    return {"q": loss}, log

  @torch.inference_mode()
  def inference_step(self, obs: torch.Tensor, eps_greedy: bool = True):
    obs = obs.to(self.device)
    if obs.dim() < 2:
      obs = obs.unsqueeze(0)
    return self.actor.sample(obs, self.critic, eps_greedy=eps_greedy), {}
