import torch
from torch import nn

from barl.policy import Policy, Critic, MaxEntActor, HParam, register_policy
from barl.networks.mlp import MLP
from barl.networks.critic import QCritic, q_bellman_loss

import gymnasium as gym

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class TanhGaussianActor(nn.Module, MaxEntActor):
  """
  Reparameterized tanh-squashed Gaussian actor (Haarnoja et al., 2018 SAC).
  Unlike every other stochastic actor in this framework, its policy-
  improvement step (in SACPolicy.train_step, not here -- no update())
  backprops THROUGH the sampled action into critic.value() (the
  reparameterization trick), so sampling uses rsample(), not sample(); the
  tanh squash needed to keep actions in [act_low, act_high] then needs the
  standard change-of-variables correction on the log-prob, or the entropy
  term feeding both the actor loss and the critic's soft-Q bootstrap would
  be wrong. Carries its own entropy dual (log_alpha, auto-tuned against a
  heuristic target entropy of -action_dim) -- that's what makes it a
  MaxEntActor, same as DAM's FlowSDEActor.
  """

  def __init__(self, obs_dim: int, action_dim: int, act_low: torch.Tensor, act_high: torch.Tensor,
               hidden_dim: int = 256, layers: int = 2, log_alpha_init: float = 0.0):
    super().__init__()

    self.action_dim      = action_dim
    self.entropy_target  = -float(action_dim)

    self.register_buffer("act_scale", (act_high - act_low) / 2)
    self.register_buffer("act_bias",  (act_high + act_low) / 2)

    self.log_alpha = nn.Parameter(torch.tensor(log_alpha_init))
    self.net = MLP(obs_dim, action_dim * 2, hidden_dim, layers)

  def dist(self, s: torch.Tensor) -> torch.distributions.Normal:
    with torch.amp.autocast("cuda", torch.bfloat16):
      out = self.net(s)
    mean, log_std = out.float().chunk(2, dim=-1)
    log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
    return torch.distributions.Normal(mean, log_std.exp())

  def rsample(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reparameterized (action, log_prob) -- differentiable through s (and
    the actor's own params), unlike sample()/sample_with_entropy() which
    detach for use as fixed rollout/bootstrap values."""
    dist = self.dist(s)
    # dist.rsample() itself (not just x = mean + std*eps): its internal
    # _standard_normal() does an in-place .normal_() fill on a freshly
    # torch.empty()'d tensor, which vmap's randomness="different" (needed
    # to train several seeds at once, see vmap_policy.py) can't handle.
    # Reparameterizing by hand with torch.randn_like is the exact same
    # math Normal.rsample() computes internally, just via a vmap-safe path.
    x = dist.mean + dist.stddev * torch.randn_like(dist.mean)
    y = torch.tanh(x)
    action = y * self.act_scale + self.act_bias
    # standard tanh-squash log-prob Jacobian correction (SAC appendix C)
    log_prob = dist.log_prob(x) - torch.log(self.act_scale * (1 - y.pow(2)) + 1e-6)
    return action, log_prob.sum(dim=-1, keepdim=True)

  def sample(self, s: torch.Tensor, critic: Critic = None) -> torch.Tensor:
    with torch.no_grad():
      action, _ = self.rsample(s)
    return action

  def sample_with_entropy(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
      action, log_prob = self.rsample(s)
      alpha = self.log_alpha.exp()
    return action, (-alpha * log_prob).detach()

  def create_optimizer(self) -> dict:
    return {
      "actor": torch.optim.Adam(self.net.parameters(), lr=3e-4),
      "alpha": torch.optim.Adam([self.log_alpha], lr=3e-4),
    }


@register_policy
class SACPolicy(Policy):
  """
  Soft Actor-Critic (Haarnoja et al., 2018): entropy-regularized off-policy
  actor-critic with a reparameterized tanh-Gaussian actor and automatic
  temperature tuning. `n_models=2` (the traditional SAC twin-critic count)
  reuses QCritic's ensemble mean-minus-rho*std pessimism as the Bellman
  target's lower confidence bound, same mechanism FlowQL/DAM already share
  -- not an exact min(Q1,Q2), but the same conservative-estimate family
  used throughout this framework rather than a SAC-specific reduction rule.
  """

  gamma      = HParam(0.99,  "Discount factor",                                              range=(0.9, 0.9999))
  rho        = HParam(0.5,   "Pessimism weight on Q std (lower bound = mean - rho * std)",    range=(0.0, 5.0))
  polyak_tau = HParam(0.005, "EMA decay step size for the Q target (1 - polyak_tau = decay)", range=(0.001, 0.05))

  def __init__(self, env: gym.Env, hidden_dim: int = 256, layers: int = 2, n_models: int = 2,
               distributional: bool = False, num_quantiles: int = 25, **kwargs):
    super().__init__(env, **kwargs)

    assert isinstance(env.action_space, gym.spaces.Box)

    self.critic = QCritic(self.observation_dim, self.action_dim, hidden_dim=hidden_dim, layers=layers,
                           n_models=n_models, distributional=distributional, num_quantiles=num_quantiles,
                           polyak_tau=self.polyak_tau, rho=self.rho, pessimistic_score=False)
    self.actor  = TanhGaussianActor(self.observation_dim, self.action_dim, self.act_low, self.act_high,
                                     hidden_dim=hidden_dim, layers=layers)

  def create_optimizer(self):
    return {**self.critic.create_optimizer(), **self.actor.create_optimizer()}

  def train_step(self, batch):
    obs      = self.normalize_obs(batch["obs"].to(self.device))
    next_obs = self.normalize_obs(batch["next_obs"].to(self.device))
    action   = batch["act"].to(self.device)
    reward   = batch["rew"].to(self.device)
    term     = batch["term"].to(self.device)

    self.critic.step_ema()

    """ Critic: soft-Q Bellman regression, entropy bonus folded into the bootstrap """
    with torch.no_grad():
      next_action, entropy_bonus = self.actor.sample_with_entropy(next_obs)
      bootstrap = self.critic.next_bootstrap(next_obs, next_action) + entropy_bonus

    q_loss = q_bellman_loss(self.critic, obs, action, reward, term, bootstrap, self.gamma)

    """ Actor: maximize Q - alpha*log_prob via the reparameterization trick
    (gradients flow through the sampled action into critic.value()) """
    new_action, log_prob = self.actor.rsample(obs)
    alpha = self.actor.log_alpha.exp()

    q_new      = self.critic.value(obs, new_action)
    actor_loss = (alpha.detach() * log_prob - q_new).mean()

    """ Entropy dual: push mean log-prob toward the target entropy """
    alpha_loss = (alpha * (-log_prob.detach() - self.actor.entropy_target)).mean()

    log = {
      "train/loss/q":     q_loss.detach(),
      "train/loss/actor": actor_loss.detach(),
      "train/loss/alpha": alpha_loss.detach(),
      "stats/alpha":       alpha.detach(),
      "stats/log_prob":    log_prob.detach().mean(),
      "stats/q_new":       q_new.detach().mean(),
    }
    return {"q": q_loss, "actor": actor_loss, "alpha": alpha_loss}, log

  @torch.inference_mode()
  def inference_step(self, observation: torch.Tensor, step_info=None):
    obs = self.normalize_obs(observation.to(self.device))
    return self.actor.sample(obs), {}
