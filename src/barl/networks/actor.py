import torch
from torch import nn

from barl.networks.mlp import MLP

from barl.policy import Actor, Critic

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class DeterministicActor(nn.Module, Actor):
  """
  Generic flat regression actor (TD3-style point estimate / pure BC),
  configured purely by architecture. Used standalone for BC pretraining
  (DirectPolicy); the regression loss itself lives in that Policy's
  train_step, not here.
  """

  def __init__(self, obs_dim: int, action_dim: int, act_low: torch.Tensor,
               act_high: torch.Tensor, hidden_dim: int = 128, layers: int = 4):
    super().__init__()

    self.action_dim = action_dim
    # .clone(): callers pass the owning Policy's own act_low/act_high
    # buffer straight through -- registering the SAME tensor object here
    # too makes named_buffers() (default remove_duplicate=True) silently
    # drop this copy's qualified name, which breaks stack_module_state +
    # functional_call's by-name substitution under vmap (see vmap_policy.py).
    self.register_buffer("act_low",  act_low.clone())
    self.register_buffer("act_high", act_high.clone())

    self.net = MLP(obs_dim, action_dim, hidden_dim, layers)

  def forward(self, s: torch.Tensor) -> torch.Tensor:
    return self.net(s)

  def sample(self, s: torch.Tensor, critic: Critic = None) -> torch.Tensor:
    return self(s).clamp(self.act_low, self.act_high)

  def create_optimizer(self) -> dict:
    return {"actor": torch.optim.AdamW(self.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=1e-4, fused=True)}


class StochasticActor(nn.Module, Actor):
  """
  Generic MLP actor over a Categorical (discrete) or diagonal-Normal
  (continuous) distribution, configured purely by architecture -- shared by
  every policy whose update is just "reweight a stochastic policy's
  log-prob by some scalar" (IQL's advantage-weighted regression, PPO's
  clipped surrogate). The weighting scheme itself lives in each owning
  Policy's train_step, not here.
  """

  def __init__(self, obs_dim: int, action_dim: int, discrete: bool,
               act_low: torch.Tensor = None, act_high: torch.Tensor = None,
               hidden_dim: int = 256, layers: int = 6):
    super().__init__()

    self.action_dim = action_dim
    self.discrete    = discrete

    if not discrete:
      # .clone(): see DeterministicActor's identical comment above --
      # aliasing the Policy's own act_low/act_high buffer object here
      # breaks stack_module_state + functional_call's by-name substitution
      # under vmap.
      self.register_buffer("act_low",  act_low.clone())
      self.register_buffer("act_high", act_high.clone())
      self.log_std = nn.Parameter(torch.zeros(action_dim))

    self.net = MLP(obs_dim, action_dim, hidden_dim, layers)

  def dist(self, s: torch.Tensor) -> torch.distributions.Distribution:
    with torch.amp.autocast("cuda", torch.bfloat16):
      out = self.net(s)
    if self.discrete:
      return torch.distributions.Categorical(logits=out.float())
    log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
    return torch.distributions.Normal(out.float(), log_std.exp())

  def log_prob(self, dist: torch.distributions.Distribution, action: torch.Tensor) -> torch.Tensor:
    if self.discrete:
      return dist.log_prob(action.view(-1))
    return dist.log_prob(action).sum(dim=-1)

  def sample(self, s: torch.Tensor, critic: Critic = None) -> torch.Tensor:
    action = self.dist(s).sample()
    if self.discrete:
      return action.view(-1, 1).long()
    return action.clamp(self.act_low, self.act_high)

  def mode(self, s: torch.Tensor) -> torch.Tensor:
    """Deterministic action (argmax / distribution mean) rather than a
    stochastic draw -- e.g. IQL trains this actor via AWR against a
    Categorical/Normal log-prob, but evaluates/deploys the mode, not a
    sample from it."""
    dist = self.dist(s)
    if self.discrete:
      return dist.logits.argmax(dim=-1, keepdim=True)
    return dist.mean.clamp(self.act_low, self.act_high)

  def create_optimizer(self) -> dict:
    return {"actor": torch.optim.Adam(self.parameters(), lr=3e-4)}
