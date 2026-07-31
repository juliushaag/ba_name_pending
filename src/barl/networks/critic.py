import torch
from torch import nn
from torch.nn import functional as F

from barl.networks.mlp import MLP
from barl.networks.ema import EMA, Shadow
from barl.networks.ensemble import Ensemble
from barl.networks.registry import register_component

from barl.policy import Critic


class Bootstrap:
  """What `bellman_target()` consumes: a target-net prediction in the critic's
  own representation, plus a per-sample value shift. Keeping the shift separate
  is what lets a Policy write `critic.bootstrap(...) + alpha * entropy` no
  matter which critic it holds -- for a fixed categorical support a value shift
  moves the support, not the probabilities, so plain tensor addition would
  silently corrupt it."""

  def __init__(self, pred: torch.Tensor, shift=0.0):
    self.pred  = pred
    self.shift = shift

  def __add__(self, other) -> "Bootstrap":
    return Bootstrap(self.pred, self.shift + other)

  __radd__ = __add__


class QCriticBase(Critic, nn.Module):
  """Shared scaffolding for every concrete critic below: Polyak target-net
  bookkeeping and the bootstrap -> bellman_target -> loss regression path. Not
  instantiated directly. `polyak_tau`/`gamma` are config/models.yaml-supplied,
  not Python-level defaults -- see each policy's `critic:` slot kwargs.

  Two accessors, both meaningful for every representation: `expectation()` is
  E[Z(s,a)], `value()` is the price a Policy improves against -- the same thing
  unless a subclass reduces with a spread (see MeanStdEnsembleCritic's
  epistemic lower bound, or the distributional critics' risk aversion)."""

  def __init__(self, env_stats, q_function: nn.Module, *,
               polyak_tau: float, gamma: float, **kwargs):
    super().__init__(env_stats, **kwargs)

    self.polyak_tau = polyak_tau
    self.gamma      = gamma

    self.q_function = q_function

    self.q_ema = EMA(self.q_function)

  def before_step(self):
    self.q_ema.step()

  def raw(self, s: torch.Tensor, a: torch.Tensor, target: bool = False) -> torch.Tensor:
    """Un-reduced prediction: (n_models, B, 1) ensemble heads, or (B, num_atoms) atoms/logits.
    A critic's own loss() and nothing else -- a Policy prices with value()/expectation()."""
    net = self.q_ema if target else self.q_function
    return net(s, a)

  def expectation(self, s: torch.Tensor, a: torch.Tensor, target: bool = False) -> torch.Tensor:
    raise NotImplementedError

  def value(self, s: torch.Tensor, a: torch.Tensor, target: bool = False) -> torch.Tensor:
    return self.expectation(s, a, target=target)

  def create_optimizer(self) -> dict:
    return {"q": torch.optim.Adam(self.q_function.parameters(), lr=3e-4, fused=True)}

  def bootstrap(self, next_obs: torch.Tensor, next_act: torch.Tensor) -> Bootstrap:
    """The target net's next-state prediction, for a Policy to add its entropy
    bonus to before handing it to bellman_target()."""
    return Bootstrap(self.value(next_obs, next_act, target=True))

  def bellman_target(self, reward: torch.Tensor, term: torch.Tensor, bootstrap: Bootstrap) -> torch.Tensor:
    reward = reward.view(-1, 1)
    term   = term.view(-1, 1).to(reward.dtype)
    return reward + (1.0 - term) * self.gamma * (bootstrap.pred + bootstrap.shift)

  def terminal_target(self, value: torch.Tensor) -> torch.Tensor:
    """`value` itself, lifted into this critic's target representation -- the
    Bellman target of a transition that doesn't bootstrap, which is how a Policy
    regresses onto an arbitrary scalar (NC-LQL's smoothing loss)."""
    return value

  def loss(self, obs: torch.Tensor, action: torch.Tensor, target : torch.Tensor, losses: dict) -> None:
    raise NotImplementedError


class EnsembleCriticBase(QCriticBase):
  """Shared __init__/loss for the three ensemble-reduction variants below --
  `n_models` independent MLP heads, all regressed to the same scalar Bellman
  target (epistemic disagreement across heads, not a return distribution).
  Only `value()`'s reduction differs between them. Not instantiated directly."""

  def __init__(self, env_stats, hidden_dim: int, layers: int, n_models: int, **kwargs):
    q_function = Ensemble(n_models, MLP, input_dim=env_stats["obs_dim"] + env_stats["act_dim"],
                           output_dim=1, latent_dim=hidden_dim, hidden_layers=layers)
    super().__init__(env_stats, q_function, **kwargs)

  def expectation(self, s: torch.Tensor, a: torch.Tensor, target: bool = False) -> torch.Tensor:
    return self.raw(s, a, target=target).float().mean(dim=0)

  def loss(self, obs: torch.Tensor, action: torch.Tensor, target : torch.Tensor, losses: dict) -> None:

    pred   = self.raw(obs, action)
    q_loss = F.mse_loss(pred.float(), target.detach().float().expand(pred.shape))

    losses["q"] = q_loss

    self.metrics.dist("q", pred.mean(dim=0))
    self.metrics.dist("q_target", target)


@register_component
class MeanStdEnsembleCritic(EnsembleCriticBase):
  """Q(s,a) critic backed by `n_models` independent MLP heads; spread =
  epistemic uncertainty, pessimism = mean - rho*std. Default for every
  continuous-action off-policy algorithm here. See MeanEnsembleCritic/
  MinEnsembleCritic for the other reduction variants, and
  QRDistributionalCritic/C51Critic for the per-head-distribution alternative."""

  def __init__(self, env_stats, hidden_dim: int, layers: int, n_models: int, *,
               rho: float, **kwargs):
    super().__init__(env_stats, hidden_dim, layers, n_models, **kwargs)
    self.rho = rho

  def value(self, s: torch.Tensor, a: torch.Tensor, target: bool = False) -> torch.Tensor:
    raw = self.raw(s, a, target=target).float()
    return raw.mean(dim=0) - self.rho * raw.std(dim=0)


@register_component
class MeanEnsembleCritic(EnsembleCriticBase):
  """Same `n_models` MLP-head ensemble as MeanStdEnsembleCritic, but reduced
  by plain mean -- no pessimism, so value() is the ensemble expectation and
  there is no `rho` to set."""


@register_component
class MinEnsembleCritic(EnsembleCriticBase):
  """Same `n_models` MLP-head ensemble as MeanStdEnsembleCritic, but reduced
  by TD3-style min-over-heads -- its pessimism has no scale to tune, so there
  is no `rho` here either."""

  def value(self, s: torch.Tensor, a: torch.Tensor, target: bool = False) -> torch.Tensor:
    return self.raw(s, a, target=target).float().min(dim=0).values


@register_component
class QRDistributionalCritic(QCriticBase):
  """Q(s,a) critic backed by a single MLP head predicting `num_quantiles`
  return-distribution atoms, trained with the QR-DQN pinball loss (Dabney et
  al., 2017); spread = aleatoric uncertainty, so `risk_aversion` prices a
  mean-variance risk measure over the return rather than epistemic pessimism.
  Not to be confused with the fixed-support categorical C51Critic below."""

  def __init__(self, env_stats, hidden_dim: int, layers: int, num_quantiles: int, *,
               risk_aversion: float, **kwargs):
    q_function = MLP(env_stats["obs_dim"] + env_stats["act_dim"],
                     num_quantiles, hidden_dim, layers, layernorm=False)
    super().__init__(env_stats, q_function, **kwargs)

    self.risk_aversion = risk_aversion

    taus = (torch.arange(num_quantiles, dtype=torch.float32) + 0.5) / num_quantiles
    self.register_buffer("taus", taus)

  def expectation(self, s: torch.Tensor, a: torch.Tensor, target: bool = False) -> torch.Tensor:
    return self.raw(s, a, target=target).float().mean(dim=-1, keepdim=True)

  def value(self, s: torch.Tensor, a: torch.Tensor, target: bool = False) -> torch.Tensor:
    raw = self.raw(s, a, target=target).float()
    return raw.mean(dim=-1, keepdim=True) - self.risk_aversion * raw.std(dim=-1, keepdim=True)

  def bootstrap(self, next_obs: torch.Tensor, next_act: torch.Tensor) -> Bootstrap:
    # Transported atom-wise -- the atoms approximate the return distribution's spread.
    return Bootstrap(self.raw(next_obs, next_act, target=True))

  def terminal_target(self, value: torch.Tensor) -> torch.Tensor:
    return value.expand(-1, self.taus.numel())

  def loss(self, obs: torch.Tensor, action: torch.Tensor, target : torch.Tensor, losses: dict) -> None:

    pred   = self.raw(obs, action)
    q_loss = quantile_huber_loss(pred, target.detach(), self.taus)

    losses["q"] = q_loss

    self.metrics.dist("q", pred.mean(dim=-1, keepdim=True))
    self.metrics.dist("q_target", target.mean(dim=-1, keepdim=True))


@register_component
class C51Critic(QCriticBase):
  """Standard C51 (Bellemare, Dabney & Munos, 2017): a single MLP head
  predicts logits over a fixed categorical support of `num_atoms` values
  spanning [v_min, v_max]; trained by projecting the (entropy-augmented)
  discounted next-state distribution back onto that fixed support
  (`categorical_projection`) and minimizing cross-entropy against it. Spread
  = std of the learned categorical distribution, same aleatoric
  `risk_aversion` role as QRDistributionalCritic's."""

  def __init__(self, env_stats, hidden_dim: int, layers: int, num_atoms: int,
               v_min: float = None, v_max: float = None, *,
               risk_aversion: float, **kwargs):
    # Read off the argument rather than self.env_stats: the support has to
    # exist before super().__init__(), which is what sets that attribute.
    v_min = (env_stats or {}).get("return_min") if v_min is None else v_min
    v_max = (env_stats or {}).get("return_max") if v_max is None else v_max
    if v_min is None or v_max is None:
      raise ValueError(
        "C51Critic's support is fixed and must be known at construction time, but neither "
        f"config/models.yaml (v_min/v_max on the slot) nor this env's EnvStats "
        f"(return_min/return_max, declared via Environment.stats) supplies it -- got "
        f"v_min={v_min!r}, v_max={v_max!r}."
      )

    q_function = MLP(env_stats["obs_dim"] + env_stats["act_dim"],
                     num_atoms, hidden_dim, layers, layernorm=False)
    super().__init__(env_stats, q_function, **kwargs)

    self.risk_aversion = risk_aversion

    self.v_min, self.v_max = v_min, v_max
    self.delta_z = (v_max - v_min) / (num_atoms - 1)
    self.register_buffer("atoms", torch.linspace(v_min, v_max, num_atoms, dtype=torch.float32))

  def probs(self, s: torch.Tensor, a: torch.Tensor, target: bool = False) -> torch.Tensor:
    return F.softmax(self.raw(s, a, target=target).float(), dim=-1)

  def _moments(self, probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = (probs * self.atoms).sum(dim=-1, keepdim=True)
    var  = (probs * self.atoms.pow(2)).sum(dim=-1, keepdim=True) - mean.pow(2)
    return mean, var.clamp_min(0).sqrt()

  def expectation(self, s: torch.Tensor, a: torch.Tensor, target: bool = False) -> torch.Tensor:
    return (self.probs(s, a, target=target) * self.atoms).sum(dim=-1, keepdim=True)

  def value(self, s: torch.Tensor, a: torch.Tensor, target: bool = False) -> torch.Tensor:
    mean, std = self._moments(self.probs(s, a, target=target))
    return mean - self.risk_aversion * std

  def bootstrap(self, next_obs: torch.Tensor, next_act: torch.Tensor) -> Bootstrap:
    return Bootstrap(self.probs(next_obs, next_act, target=True))

  def bellman_target(self, reward: torch.Tensor, term: torch.Tensor, bootstrap: Bootstrap) -> torch.Tensor:
    return categorical_projection(bootstrap.pred, self.atoms, reward, term, self.gamma,
                                  bootstrap.shift, self.v_min, self.v_max, self.delta_z)

  def terminal_target(self, value: torch.Tensor) -> torch.Tensor:
    # term=1 kills the bootstrap, so every atom maps to `value` and the incoming
    # distribution is irrelevant -- what comes back is a projected Dirac there.
    num_atoms = self.atoms.numel()
    uniform   = value.new_full((value.shape[0], num_atoms), 1.0 / num_atoms)
    return categorical_projection(uniform, self.atoms, value, torch.ones_like(value),
                                  self.gamma, 0.0, self.v_min, self.v_max, self.delta_z)

  def loss(self, obs: torch.Tensor, action: torch.Tensor, target : torch.Tensor, losses: dict) -> None:

    pred_logits = self.raw(obs, action).float()
    q_loss = -(target.detach() * F.log_softmax(pred_logits, dim=-1)).sum(dim=-1).mean()

    losses["q"] = q_loss

    mean, std = self._moments(F.softmax(pred_logits, dim=-1))
    self.metrics.dist("q", mean)
    self.metrics.dist("atom_spread", std)
    self.metrics.ema_gap("ema", self.q_ema)


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
