import torch
from torch import nn
from torch.nn import functional as F

from barl.networks.mlp import MLP
from barl.networks.ema import EMA, Shadow
from barl.networks.ensemble import Ensemble
from barl.networks.registry import register_component

from barl.policy import Critic


class QCriticBase(Critic, nn.Module):
  """Shared scaffolding for every concrete critic below: Polyak target-net
  bookkeeping, mean-minus-rho*std value reduction. Not instantiated
  directly. `rho`/`polyak_tau`/`gamma` are config/models.yaml-supplied, not
  Python-level defaults -- see each policy's `critic:` slot kwargs."""

  def __init__(self, q_function: nn.Module, pessimistic_score: bool = False, *,
               rho: float, polyak_tau: float, gamma: float, **kwargs):
    super().__init__(**kwargs)

    self.rho        = rho
    self.polyak_tau = polyak_tau
    self.gamma      = gamma

    self.pessimistic_score = pessimistic_score
    self.q_function = q_function

    self.q_ema = EMA(self.q_function)

  def on_epoch_start(self):
    # self.q_ema.on_epoch_start()
    ...

  def step(self):
    self.q_ema.step()

  def raw(self, s: torch.Tensor, a: torch.Tensor, target: bool = False) -> torch.Tensor:
    """Un-reduced prediction: (n_models, B, 1) ensemble heads, or (B, num_atoms) atoms/logits."""
    net = self.q_ema if target else self.q_function
    with torch.amp.autocast("cuda", torch.bfloat16):
      return net(s, a)

  def _mean_std(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError

  def value(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    mean, std = self._mean_std(self.raw(s, a))
    return mean - self.rho * std

  def target_value(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    mean, std = self._mean_std(self.raw(s, a, target=True))
    return mean - self.rho * std

  def create_optimizer(self) -> dict:
    return {"q": torch.optim.AdamW(self.q_function.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=1e-4)}

  def _bellman_target(self, reward: torch.Tensor, term: torch.Tensor, bootstrap: torch.Tensor) -> torch.Tensor:
    reward = reward.view(-1, 1)
    term   = term.view(-1, 1).to(reward.dtype)
    return reward + (1.0 - term) * self.gamma * bootstrap

  def loss(self, obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, term: torch.Tensor,
           next_obs: torch.Tensor, next_act: torch.Tensor, penalty: torch.Tensor,
           log: dict, losses: dict) -> None:
    """Writes into losses["q"]/log["train/loss/q"] in place. `next_act` must
    already be sampled; `penalty` is an entropy bonus folded into the
    bootstrap (or zeros)."""
    raise NotImplementedError


class EnsembleCriticBase(QCriticBase):
  """Shared __init__/loss for the three ensemble-reduction variants below --
  `n_models` independent MLP heads, all regressed to the same scalar Bellman
  target (epistemic disagreement across heads, not a return distribution).
  Only `_mean_std` (and hence the pessimism baked into value()/target_value())
  differs between them. Not instantiated directly."""

  def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, layers: int,
               n_models: int, pessimistic_score: bool = False, **kwargs):
    q_function = Ensemble(n_models, MLP, input_dim=obs_dim + action_dim,
                           output_dim=1, latent_dim=hidden_dim, hidden_layers=layers)
    super().__init__(q_function, pessimistic_score, **kwargs)

  def loss(self, obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, term: torch.Tensor,
           next_obs: torch.Tensor, next_act: torch.Tensor, penalty: torch.Tensor,
           log: dict, losses: dict) -> None:
    
    with torch.no_grad():
      bootstrap = self.target_value(next_obs, next_act) + penalty
      target    = self._bellman_target(reward, term, bootstrap)


    pred   = self.raw(obs, action)
    q_loss = F.mse_loss(pred.float(), target.detach().float().expand(pred.shape))

    losses["q"] = q_loss
    log["train/loss/q"] = q_loss.detach()


    log["stats/q_target"] = bootstrap.detach().mean()     
    log["stats/q_mean"] = pred.mean().detach()
    log["stats/q_std"]  = pred.std(0).mean().detach()


@register_component
class MeanStdEnsembleCritic(EnsembleCriticBase):
  """Q(s,a) critic backed by `n_models` independent MLP heads; spread =
  epistemic uncertainty, pessimism = mean - rho*std. Default for every
  continuous-action off-policy algorithm here. See MeanEnsembleCritic/
  MinEnsembleCritic for the other reduction variants, and
  QRDistributionalCritic/C51Critic for the per-head-distribution alternative."""

  def _mean_std(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    raw = raw.float()
    return raw.mean(dim=0), raw.std(dim=0)


@register_component
class MeanEnsembleCritic(EnsembleCriticBase):
  """Same `n_models` MLP-head ensemble as MeanStdEnsembleCritic, but reduced
  by plain mean -- no pessimism, `rho` is ignored (std is forced to 0 so
  value()/target_value()'s mean - rho*std collapses to the mean)."""

  def _mean_std(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    raw = raw.float()
    return raw.mean(dim=0), torch.zeros_like(raw[0])


@register_component
class MinEnsembleCritic(EnsembleCriticBase):
  """Same `n_models` MLP-head ensemble as MeanStdEnsembleCritic, but reduced
  by TD3-style min-over-heads -- `rho` is ignored (std is forced to 0 so
  value()/target_value()'s mean - rho*std collapses to the min)."""

  def _mean_std(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    raw = raw.float()
    min_val = raw.min(dim=0).values
    return min_val, torch.zeros_like(min_val)


@register_component
class QRDistributionalCritic(QCriticBase):
  """Q(s,a) critic backed by a single MLP head predicting `num_quantiles`
  return-distribution atoms, trained with the QR-DQN pinball loss (Dabney et
  al., 2017); spread = aleatoric uncertainty. Not to be confused with the
  fixed-support categorical C51Critic below."""

  def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, layers: int,
               num_quantiles: int, pessimistic_score: bool = False, **kwargs):
    q_function = MLP(obs_dim + action_dim, num_quantiles, hidden_dim, layers, layernorm=False)
    super().__init__(q_function, pessimistic_score, **kwargs)

    taus = (torch.arange(num_quantiles, dtype=torch.float32) + 0.5) / num_quantiles
    self.register_buffer("taus", taus)

  def _mean_std(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    raw = raw.float()
    return raw.mean(dim=-1, keepdim=True), raw.std(dim=-1, keepdim=True)

  def loss(self, obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, term: torch.Tensor,
           next_obs: torch.Tensor, next_act: torch.Tensor, penalty: torch.Tensor,
           log: dict, losses: dict) -> None:
    with torch.no_grad():
      # Transported atom-wise -- the atoms approximate the return distribution's spread.
      bootstrap = self.raw(next_obs, next_act, target=True) + penalty
      target    = self._bellman_target(reward, term, bootstrap)

    pred   = self.raw(obs, action)
    q_loss = quantile_huber_loss(pred, target.detach(), self.taus)

    losses["q"] = q_loss
    log["train/loss/q"] = q_loss.detach()

    log["stats/q_target"] = bootstrap.detach().mean()     
    log["stats/q_mean"] = pred.mean().detach()
    log["stats/q_std"]  = pred.std(dim=-1).mean().detach()


@register_component
class C51Critic(QCriticBase):
  """Standard C51 (Bellemare, Dabney & Munos, 2017): a single MLP head
  predicts logits over a fixed categorical support of `num_atoms` values
  spanning [v_min, v_max]; trained by projecting the (entropy-augmented)
  discounted next-state distribution back onto that fixed support
  (`categorical_projection`) and minimizing cross-entropy against it. Spread
  = std of the learned categorical distribution, same pessimism role as
  QRDistributionalCritic's std."""

  def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, layers: int,
               num_atoms: int, v_min: float, v_max: float, pessimistic_score: bool = False, **kwargs):
    q_function = MLP(obs_dim + action_dim, num_atoms, hidden_dim, layers, layernorm=False)
    super().__init__(q_function, pessimistic_score, **kwargs)

    self.v_min, self.v_max = v_min, v_max
    self.delta_z = (v_max - v_min) / (num_atoms - 1)
    self.register_buffer("atoms", torch.linspace(v_min, v_max, num_atoms, dtype=torch.float32))

  def _mean_std(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probs = F.softmax(raw.float(), dim=-1)
    mean  = (probs * self.atoms).sum(dim=-1, keepdim=True)
    var   = (probs * self.atoms.pow(2)).sum(dim=-1, keepdim=True) - mean.pow(2)
    return mean, var.clamp_min(0).sqrt()

  def loss(self, obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, term: torch.Tensor,
           next_obs: torch.Tensor, next_act: torch.Tensor, penalty: torch.Tensor,
           log: dict, losses: dict) -> None:
    self.step_ema()
    with torch.no_grad():
      next_probs  = F.softmax(self.raw(next_obs, next_act, target=True).float(), dim=-1)
      target_dist = categorical_projection(next_probs, self.atoms, reward, term, self.gamma,
                                            penalty, self.v_min, self.v_max, self.delta_z)

    pred_logits = self.raw(obs, action).float()
    q_loss = -(target_dist * F.log_softmax(pred_logits, dim=-1)).sum(dim=-1).mean()

    losses["q"] = q_loss
    log["train/loss/q"] = q_loss.detach()

    mean, std = self._mean_std(pred_logits)
    log["stats/q_mean"] = mean.mean().detach()
    log["stats/q_std"]  = std.mean().detach()


def quantile_huber_loss(pred: torch.Tensor, target: torch.Tensor, taus: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
  """QR-DQN pairwise quantile Huber (pinball) loss: every predicted atom is
  regressed against every target atom, asymmetrically weighted so each
  settles at its assigned quantile rather than the mean."""
  diff   = target.unsqueeze(-2) - pred.unsqueeze(-1)   # (..., N_pred, N_target)
  huber  = torch.where(diff.abs() <= kappa, 0.5 * diff ** 2, kappa * (diff.abs() - 0.5 * kappa))
  weight = (taus.view(*([1] * (diff.dim() - 2)), -1, 1) - (diff < 0).float()).abs()
  return (weight * huber).mean()


def categorical_projection(next_probs: torch.Tensor, atoms: torch.Tensor, reward: torch.Tensor,
                            term: torch.Tensor, gamma: float, penalty, v_min: float, v_max: float,
                            delta_z: float) -> torch.Tensor:
  """C51's categorical Bellman projection (Bellemare et al., 2017): shift the
  next-state atom support by the (entropy-augmented) discounted reward, clip
  to [v_min, v_max], then split each shifted atom's probability mass onto its
  two nearest fixed-support neighbors."""
  reward    = reward.view(-1, 1)
  term      = term.view(-1, 1).to(reward.dtype)
  num_atoms = atoms.shape[0]

  Tz = reward + (1.0 - term) * gamma * (atoms.view(1, -1) + penalty)
  Tz = Tz.clamp(v_min, v_max)

  b = (Tz - v_min) / delta_z
  l = b.floor().long().clamp(0, num_atoms - 1)
  u = b.ceil().long().clamp(0, num_atoms - 1)

  # b exactly integer -> l == u; route all mass to l so it isn't dropped by
  # the (u - b) / (b - l) split weights both being zero.
  on_atom = (l == u)
  m_l = torch.where(on_atom, next_probs, next_probs * (u.float() - b))
  m_u = torch.where(on_atom, torch.zeros_like(next_probs), next_probs * (b - l.float()))

  batch_size = next_probs.shape[0]
  offset = (torch.arange(batch_size, device=next_probs.device) * num_atoms).unsqueeze(1)

  m = next_probs.new_zeros(batch_size * num_atoms)
  m.index_add_(0, (l + offset).reshape(-1), m_l.reshape(-1))
  m.index_add_(0, (u + offset).reshape(-1), m_u.reshape(-1))
  return m.view(batch_size, num_atoms)
