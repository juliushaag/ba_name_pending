import torch
from torch import nn
from torch.nn import functional as F

from barl.networks.mlp import MLP
from barl.networks.ema import EMA
from barl.networks.ensemble import Ensemble

from barl.policy import Critic


class QCritic(nn.Module, Critic):
  """
  Generic Q(s,a) critic shared by every continuous-action off-policy
  algorithm in this framework (FlowQL, DAM, QAM) -- configured purely by
  architecture, never by algorithm hyperparameters. gamma, discounting, the
  Bellman target, and the regression loss itself all live in each Policy's
  train_step (see q_bellman_loss below), not here; this class only ever
  computes Q(s,a) and its Polyak-averaged target.

  Two mutually exclusive architectures, chosen by `distributional`:
    - ensemble (default): `n_models` independently-initialized MLP heads,
      each predicting a scalar Q. Spread across heads reflects epistemic
      (model) uncertainty.
    - distributional: a single MLP head predicting `num_quantiles` atoms of
      the return distribution (QR-DQN-style quantile regression). Spread
      across atoms reflects aleatoric (environment/policy) uncertainty --
      genuinely different from the ensemble's epistemic spread, which is
      why the two are mutually exclusive rather than combined.

  `rho`/`pessimistic_score` control how that spread feeds into value
  estimates: target_value() always uses a mean-minus-rho*std lower
  confidence bound (every Bellman target in this framework wants this);
  score() -- the ∇_a Q used for actor updates -- uses the plain mean unless
  `pessimistic_score=True` (DAM's direct-score regression wants the
  pessimistic gradient; FlowQL/QAM's DPG/adjoint-matching losses want the
  plain one).
  """

  def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 512, layers: int = 4,
               n_models: int = 10, distributional: bool = False, num_quantiles: int = 25,
               polyak_tau: float = 0.005, rho: float = 0.5, pessimistic_score: bool = False):
    super().__init__()

    self.distributional    = distributional
    self.rho               = rho
    self.pessimistic_score = pessimistic_score

    if distributional:
      taus = (torch.arange(num_quantiles, dtype=torch.float32) + 0.5) / num_quantiles
      self.register_buffer("taus", taus)
      self.q_function = MLP(obs_dim + action_dim, num_quantiles, hidden_dim, layers, layernorm=False)
    else:
      self.q_function = Ensemble(n_models, MLP, input_dim=obs_dim + action_dim,
                                  output_dim=1, latent_dim=hidden_dim, hidden_layers=layers)

    self.q_ema = EMA(self.q_function, start_decay=1 - polyak_tau, end_decay=1 - polyak_tau, warmup_steps=1)

  def step_ema(self):
    self.q_ema.step()

  def raw(self, s: torch.Tensor, a: torch.Tensor, target: bool = False) -> torch.Tensor:
    """Un-reduced prediction: (n_models, B, 1) ensemble heads, or (B, num_quantiles)
    quantile atoms. Policy.train_step regresses against this directly (see
    q_bellman_loss); value()/target_value()/score() reduce it further below."""
    net = self.q_ema if target else self.q_function
    with torch.amp.autocast("cuda", torch.bfloat16):
      return net(s, a)

  def _mean_std(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dim = -1 if self.distributional else 0
    raw = raw.float()
    return raw.mean(dim=dim, keepdim=self.distributional), raw.std(dim=dim, keepdim=self.distributional)

  def value(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    mean, _ = self._mean_std(self.raw(s, a))
    return mean

  def target_value(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    mean, std = self._mean_std(self.raw(s, a, target=True))
    return mean - self.rho * std

  def next_bootstrap(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Whatever shape q_bellman_loss needs to build the Bellman target with:
    raw target-net quantile atoms (B, num_quantiles) if distributional
    (transported atom-wise through the Bellman equation), else the reduced
    pessimistic scalar target_value() (B, 1)."""
    return self.raw(s, a, target=True) if self.distributional else self.target_value(s, a)

  def score(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    # torch.func.grad, not a.requires_grad_(True) + eager .backward(): the
    # latter populates a.grad as a side effect outside the batched-tensor
    # dataflow vmap traces, so it can't compose with vmap (needed to train
    # several seeds at once, see vmap_policy.py); torch.func.grad is a
    # proper functional transform that does.
    a = a.detach()

    def q_scalar(a):
      mean, std = self._mean_std(self.raw(s, a))
      q = (mean - self.rho * std) if self.pessimistic_score else mean
      return q.sum()

    return torch.func.grad(q_scalar)(a)

  def create_optimizer(self) -> dict:
    return {"q": torch.optim.AdamW(self.q_function.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=1e-4)}


def quantile_huber_loss(pred: torch.Tensor, target: torch.Tensor, taus: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
  """
  QR-DQN pairwise quantile Huber (pinball) loss: every predicted atom i is
  regressed against every target atom j, weighted by the asymmetric
  |tau_i - 1{target_j < pred_i}| factor that makes each output atom settle
  at its assigned quantile of the target distribution rather than its mean.
  `pred`/`target`: (..., N) atoms. `taus`: (N,) quantile fractions.
  """
  diff   = target.unsqueeze(-2) - pred.unsqueeze(-1)   # (..., N_pred, N_target)
  huber  = torch.where(diff.abs() <= kappa, 0.5 * diff ** 2, kappa * (diff.abs() - 0.5 * kappa))
  weight = (taus.view(*([1] * (diff.dim() - 2)), -1, 1) - (diff < 0).float()).abs()
  return (weight * huber).mean()


def q_bellman_loss(critic: QCritic, obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor,
                    term: torch.Tensor, bootstrap: torch.Tensor, gamma: float) -> torch.Tensor:
  """
  Critic regression loss shared by every QCritic-based policy (FlowQL, DAM,
  QAM) -- MSE against a scalar target for an ensemble critic (every head
  regresses toward the same reward + gamma*bootstrap scalar, since ensemble
  members represent epistemic disagreement, not samples of a return
  distribution), or QR-DQN-style pairwise quantile-Huber regression for a
  distributional critic (every predicted atom regresses against every
  Bellman-transported next-state atom, since here the atoms ARE meant to
  approximate the return distribution's spread).

  `bootstrap` is whatever the caller already computed for the next state --
  typically critic.next_bootstrap(next_obs, next_action), optionally with an
  entropy bonus folded in by the caller (e.g. DAMPolicy) -- broadcastable
  against reward/term either way, since this function never distinguishes
  ensemble vs. distributional beyond that broadcast.
  """
  reward = reward.view(-1, 1)
  term   = term.view(-1, 1).to(reward.dtype)
  target = reward + (1.0 - term) * gamma * bootstrap

  pred = critic.raw(obs, action)
  if critic.distributional:
    return quantile_huber_loss(pred, target.detach(), critic.taus)
  return F.mse_loss(pred.float(), target.detach().float().expand(pred.shape))
