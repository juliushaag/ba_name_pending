import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.components import Buffer
from barl_jax.optimization import adam
from barl_jax.registration import register_component

from .base import QCritic, QCriticConfig, scalar_td_target, value_range


@dataclass(frozen=True)
class CategoricalCriticConfig(QCriticConfig):
  """`learn_support` switches on DySEL (Learning the Supports for Categorical
  Critic in RL, arXiv:2607.01880): `[v_min, v_max]` becomes a trained parameter
  pair, started at `support_init_*` (the configured / env return range is then
  unused), instead of a fixed range. The bins stay `num_bins` equal-width and
  index-fixed while the range moves.

  The support minimises `support_alpha * max(|v_min|, |v_max|)` subject to the
  expected mass the TD target leaks outside it staying below `support_eps`,
  enforced by a Lagrange multiplier trained by gradient ascent. The cross-entropy
  is scaled by 1 / `support_alpha`, as in the paper, and sigma -- the width of the
  Gaussian the leak is measured under, `sigma_to_bin_ratio` bin widths -- is
  floored at `support_sigma_min` so shrinking it cannot fake coverage. HL-Gauss
  smooths its target with that same sigma; C51 uses it only for the leak."""
  num_bins: int = 100
  v_min: float | None = None
  v_max: float | None = None
  entropy_coef: float = 0.005
  sigma_to_bin_ratio: float = 0.75
  learn_support: bool = True
  support_init_min: float = -10.0
  support_init_max: float = 10.0
  support_alpha: float = 0.3
  support_eps: float = 0.005
  support_sigma_min: float = 0.3
  support_lr: float = 3.0e-3
  dual_lr: float = 1.0e-3
  dual_init: float = 1.0

  def problems(self) -> list[str]:
    problems = super().problems()
    if self.num_bins < 2:
      problems.append(f"num_bins={self.num_bins!r}: must be >= 2")
    if self.sigma_to_bin_ratio <= 0:
      problems.append(f"sigma_to_bin_ratio={self.sigma_to_bin_ratio!r}: must be > 0")
    if not self.learn_support:
      return problems
    if not self.support_init_min < self.support_init_max:
      problems.append("support_init_min must be < support_init_max")
    for name in ("support_alpha", "support_sigma_min", "support_lr", "dual_lr"):
      if not getattr(self, name) > 0:
        problems.append(f"{name}={getattr(self, name)!r}: must be > 0")
    if not 0 < self.support_eps < 1:
      problems.append(f"support_eps={self.support_eps!r}: must be in (0, 1)")
    if self.dual_init < 0:
      problems.append(f"dual_init={self.dual_init!r}: must be >= 0")
    return problems


@dataclass(frozen=True)
class C51CriticConfig(CategoricalCriticConfig):
  class_name: str = "C51Critic"


@dataclass(frozen=True)
class HLGaussCriticConfig(CategoricalCriticConfig):
  class_name: str = "HLGaussCritic"


def categorical_projection(next_probs, atoms, chunk_return, discount, v_min, v_max,
                           entropy_bonus=0.0):
  """Project `next_probs` over the shifted support `chunk_return + discount *
  (atoms + entropy_bonus)` back onto `atoms` by linear interpolation. Linear
  in `next_probs`."""
  num_atoms = atoms.shape[0]
  delta_z = (v_max - v_min) / (num_atoms - 1)
  tz = jnp.clip(chunk_return.reshape(-1, 1) + discount * (atoms.reshape(1, -1) + entropy_bonus),
                v_min, v_max)

  b = (tz - v_min) / delta_z
  l = jnp.clip(jnp.floor(b), 0, num_atoms - 1).astype(jnp.int32)
  u = jnp.clip(jnp.ceil(b), 0, num_atoms - 1).astype(jnp.int32)

  on_atom = (l == u)
  m_l = jnp.where(on_atom, next_probs, next_probs * (u.astype(b.dtype) - b))
  m_u = jnp.where(on_atom, 0.0, next_probs * (b - l.astype(b.dtype)))

  batch_idx = jnp.arange(next_probs.shape[-2])[:, None]
  proj = jnp.zeros_like(next_probs)
  proj = proj.at[..., batch_idx, l].add(m_l)
  return proj.at[..., batch_idx, u].add(m_u)


def to_histogram(value, edges, sigma):
  """HL-Gauss: the mass of N(value, sigma^2) in each bin, renormalized to the
  clipped range."""
  value = jnp.clip(value.reshape(-1, 1), edges[0], edges[-1])
  cdf = 0.5 * (1.0 + jax.scipy.special.erf((edges - value) / (sigma * math.sqrt(2.0))))
  mass = jnp.maximum(cdf[:, 1:] - cdf[:, :-1], 0.0)
  return mass / jnp.maximum(mass.sum(axis=-1, keepdims=True), 1e-8)


class LearnedSupport(nnx.Module):
  """The two trained value-range endpoints. Sorted on read, and kept a minimum
  width apart, so the interval is valid whatever the optimizer does to them."""

  MIN_WIDTH = 1.0e-2

  def __init__(self, v_min: float, v_max: float):
    self.endpoints = nnx.Param(jnp.asarray([v_min, v_max], jnp.float32))

  def bounds(self):
    lo, hi = jnp.sort(self.endpoints[...])
    return lo, jnp.maximum(hi, lo + self.MIN_WIDTH)


class CategoricalCritic(QCritic):
  """Softmax over a fixed support, fit with cross-entropy."""

  config_type = CategoricalCriticConfig

  def __init__(self, env_stats, config, **kwargs):
    super().__init__(env_stats, config, **kwargs)
    if config.learn_support:
      self.v_min, self.v_max = float(config.support_init_min), float(config.support_init_max)
      self.support_net = LearnedSupport(self.v_min, self.v_max)
      self.dual = nnx.Param(jnp.asarray(config.dual_init, jnp.float32))
    else:
      self.v_min, self.v_max = value_range(env_stats, config)

  @classmethod
  def out_dim(cls, config) -> int:
    return config.num_bins

  # ---- the (possibly moving) support ------------------------------------------
  # A head supplies `_geometry(lo, hi)` -> (edges or None, centers, sigma) for a
  # moving support and `_static_geometry()` for the fixed one.

  def _current(self):
    """The geometry the heads and the target use: a constant to every loss."""
    if not self.config.learn_support:
      return self._static_geometry()
    lo, hi = jax.lax.stop_gradient(self.support_net.bounds())
    return self._geometry(lo, hi)

  def support_values(self):
    """The value each bin / atom stands for, (num_bins,)."""
    return self._current()[1]

  def derived(self) -> dict:
    if not self.config.learn_support:
      return self._static_derived()
    lo, hi = self.support_net.bounds()
    _, centers, sigma = self._geometry(lo, hi)
    return {"v_min": float(lo), "v_max": float(hi), "sigma": float(sigma),
            "bin_width": float(centers[1] - centers[0]), "support": "learned"}

  # ---- the learned support ------------------------------------------------------

  def create_optimizer(self) -> dict:
    out = super().create_optimizer()
    if self.config.learn_support:
      out["support"] = adam(self.config.support_lr, 0.9, 1.0, "none")
      out["support_dual"] = adam(self.config.dual_lr, 0.9, 1.0, "none")
    return out

  def param_labels(self, subpath):
    """Which optimizer key trains a parameter, by the head of its path under this
    component; None leaves the network on the default "critic" key."""
    return {"support_net": "support", "dual": "support_dual"}.get(str(subpath[0]))

  def _leak(self, lo, hi, sigma, centers, chunk_return, discount, bootstrap, entropy_bonus):
    """Expected mass of the TD target outside `[lo, hi]` under Gaussians of width
    `sigma`, averaged over the batch; differentiable in `lo`, `hi` and `sigma`."""
    raise NotImplementedError

  def support_losses(self, chunk_return, terminated, horizon, bootstrap, entropy_bonus=0.0) -> dict:
    """The support's Lagrangian: it descends `alpha * max(|v_min|, |v_max|) +
    lambda * (leak - eps)`; lambda ascends `lambda * (leak - eps)`, which a single
    backward pass over the summed losses gets from the negated, stop-gradiented
    copy below."""
    if not self.config.learn_support:
      return {}
    chunk_return, discount = self.n_step_inputs(chunk_return, terminated, horizon)
    lo, hi = self.support_net.bounds()
    _, centers, sigma = self._geometry(lo, hi)
    leak = self._leak(lo, hi, sigma, jax.lax.stop_gradient(centers), chunk_return, discount,
                      jax.lax.stop_gradient(bootstrap), entropy_bonus)
    violation = leak - self.config.support_eps
    dual = self.dual[...]

    self.metrics.scalar("support_min", lo)
    self.metrics.scalar("support_max", hi)
    self.metrics.scalar("support_sigma", sigma)
    self.metrics.scalar("support_leak", leak)
    self.metrics.scalar("support_dual", dual)
    width = self.config.support_alpha * jnp.maximum(jnp.abs(lo), jnp.abs(hi))
    return {"support": width + jax.lax.stop_gradient(dual) * violation,
            "support_dual": -dual * jax.lax.stop_gradient(violation)}

  def after_step(self):
    super().after_step()
    if self.config.learn_support:
      self.dual[...] = jnp.maximum(self.dual[...], 0.0)

  def probs(self, raw):
    return jax.nn.softmax(raw.astype(jnp.float32), axis=-1)

  def expectation_members(self, raw):
    return (self.probs(raw) * self.support_values()).sum(axis=-1, keepdims=True)

  def regression_loss(self, raw, target):
    logp = jax.nn.log_softmax(raw.astype(jnp.float32), axis=-1)
    # Summed over atoms and members, averaged over the batch.
    reduce = lambda x: x.sum(axis=-1).mean(axis=-1).sum()
    loss = reduce(-(target.astype(jnp.float32) * logp))
    if self.config.entropy_coef:
      loss += self.config.entropy_coef * reduce(jnp.exp(logp) * logp)
    return loss / self.config.support_alpha if self.config.learn_support else loss

  def soft_target(self, raw):
    return self.probs(raw).mean(axis=0)

  def target_expectation(self, target):
    return (target.astype(jnp.float32) * self.support_values()).sum(axis=-1, keepdims=True)

  def record(self, raw, target) -> None:
    super().record(raw, target)
    probs = self.probs(raw)
    support = self.support_values()
    exp = (probs * support).sum(axis=-1, keepdims=True)
    var = (probs * (support - exp) ** 2).sum(axis=-1)
    self.metrics.dist("return_std", jnp.sqrt(var).mean(axis=0))


@register_component
class C51Critic(CategoricalCritic):
  """Atoms on a linspace; the bootstrap distribution is projected."""

  kind = "critic"
  config_type = C51CriticConfig

  def __init__(self, env_stats, config, **kwargs):
    super().__init__(env_stats, config, **kwargs)
    if not config.learn_support:
      self.support = Buffer(jnp.linspace(self.v_min, self.v_max, config.num_bins, dtype=jnp.float32))

  def _geometry(self, lo, hi):
    """`(None, atoms, sigma)`: `num_bins` atoms spanning `[lo, hi]` inclusive."""
    k = self.config.num_bins
    atoms = lo + (hi - lo) * jnp.arange(k, dtype=jnp.float32) / (k - 1)
    sigma = jnp.maximum(self.config.sigma_to_bin_ratio * (hi - lo) / (k - 1),
                        self.config.support_sigma_min)
    return None, atoms, sigma

  def _static_geometry(self):
    return None, self.support[...], None

  def _static_derived(self) -> dict:
    return {"v_min": self.v_min, "v_max": self.v_max,
            "bin_width": (self.v_max - self.v_min) / (self.config.num_bins - 1)}

  def bootstrap_members(self, next_raw):
    return self.probs(next_raw)

  def td_target(self, chunk_return, discount, bootstrap, entropy_bonus):
    if not self.config.learn_support:
      return categorical_projection(bootstrap, self.support[...], chunk_return, discount,
                                    self.v_min, self.v_max, entropy_bonus)
    lo, hi = jax.lax.stop_gradient(self.support_net.bounds())
    _, atoms, _ = self._geometry(lo, hi)
    return categorical_projection(bootstrap, atoms, chunk_return, discount, lo, hi, entropy_bonus)

  def _leak(self, lo, hi, sigma, centers, chunk_return, discount, bootstrap, entropy_bonus):
    """The bootstrap distribution shifted onto `chunk_return + discount * (atom +
    bonus)` is a mixture of point masses; each is read as a Gaussian of width
    `sigma`, so this reduces to the HL-Gauss leak for a single atom. Atoms the
    projection would clip count as leaked."""
    tz = chunk_return + discount * (centers[None, :] + entropy_bonus)
    scale = sigma * math.sqrt(2.0)
    outside = 0.5 * (jax.scipy.special.erfc((hi - tz) / scale)
                     + jax.scipy.special.erfc((tz - lo) / scale))
    return (bootstrap.astype(jnp.float32) * outside).sum(axis=-1).mean()


@register_component
class HLGaussCritic(CategoricalCritic):
  """Bin centers as support; the scalar TD target is smoothed into a histogram."""

  kind = "critic"
  config_type = HLGaussCriticConfig

  def __init__(self, env_stats, config, **kwargs):
    super().__init__(env_stats, config, **kwargs)
    if config.learn_support:
      return
    edges = jnp.linspace(self.v_min, self.v_max, config.num_bins + 1, dtype=jnp.float32)
    self.edges = Buffer(edges)
    self.support = Buffer((edges[:-1] + edges[1:]) / 2)
    self.sigma = config.sigma_to_bin_ratio * (self.v_max - self.v_min) / config.num_bins

  def _geometry(self, lo, hi):
    """`(edges, centers, sigma)` of `num_bins` equal bins over `[lo, hi]`."""
    k = self.config.num_bins
    edges = lo + (hi - lo) * jnp.arange(k + 1, dtype=jnp.float32) / k
    sigma = jnp.maximum(self.config.sigma_to_bin_ratio * (hi - lo) / k,
                        self.config.support_sigma_min)
    return edges, (edges[:-1] + edges[1:]) / 2, sigma

  def _static_geometry(self):
    return self.edges[...], self.support[...], self.sigma

  def _static_derived(self) -> dict:
    return {"v_min": self.v_min, "v_max": self.v_max, "sigma": float(self.sigma),
            "bin_width": (self.v_max - self.v_min) / self.config.num_bins}

  def bootstrap_members(self, next_raw):
    return self.expectation_members(next_raw)

  def td_target(self, chunk_return, discount, bootstrap, entropy_bonus):
    edges, _, sigma = self._current()
    return to_histogram(scalar_td_target(chunk_return, discount, bootstrap, entropy_bonus),
                        edges, sigma)

  def _leak(self, lo, hi, sigma, centers, chunk_return, discount, bootstrap, entropy_bonus):
    """Mass of N(y, sigma^2) outside `[lo, hi]` for the unclipped scalar target y."""
    y = scalar_td_target(chunk_return, discount, bootstrap, entropy_bonus).astype(jnp.float32)
    scale = sigma * math.sqrt(2.0)
    return 0.5 * (jax.scipy.special.erfc((hi - y) / scale)
                  + jax.scipy.special.erfc((y - lo) / scale)).mean()
