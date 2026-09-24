import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from barl_jax.components import Buffer
from barl_jax.registration import register_component

from .base import QCritic, QCriticConfig, scalar_td_target, value_range


@dataclass(frozen=True)
class CategoricalCriticConfig(QCriticConfig):
  num_bins: int = 100
  v_min: float | None = None
  v_max: float | None = None
  entropy_coef: float = 0.0

  def problems(self) -> list[str]:
    problems = super().problems()
    if self.num_bins < 2:
      problems.append(f"num_bins={self.num_bins!r}: must be >= 2")
    return problems


@dataclass(frozen=True)
class C51CriticConfig(CategoricalCriticConfig):
  class_name: str = "C51Critic"
  entropy_coef: float = 0.005


@dataclass(frozen=True)
class HLGaussCriticConfig(CategoricalCriticConfig):
  class_name: str = "HLGaussCritic"
  sigma_to_bin_ratio: float = 0.75


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


class CategoricalCritic(QCritic):
  """Softmax over a fixed support, fit with cross-entropy."""

  config_type = CategoricalCriticConfig

  def __init__(self, env_stats, config, **kwargs):
    super().__init__(env_stats, config, **kwargs)
    self.v_min, self.v_max = value_range(env_stats, config)

  @classmethod
  def out_dim(cls, config) -> int:
    return config.num_bins

  def probs(self, raw):
    return jax.nn.softmax(raw.astype(jnp.float32), axis=-1)

  def expectation_members(self, raw):
    return (self.probs(raw) * self.support[...]).sum(axis=-1, keepdims=True)

  def regression_loss(self, raw, target):
    logp = jax.nn.log_softmax(raw.astype(jnp.float32), axis=-1)
    # Summed over atoms and members, averaged over the batch.
    reduce = lambda x: x.sum(axis=-1).mean(axis=-1).sum()
    loss = reduce(-(target.astype(jnp.float32) * logp))
    if self.config.entropy_coef:
      loss += self.config.entropy_coef * reduce(jnp.exp(logp) * logp)
    return loss

  def soft_target(self, raw):
    return self.probs(raw).mean(axis=0)

  def target_expectation(self, target):
    return (target.astype(jnp.float32) * self.support[...]).sum(axis=-1, keepdims=True)

  def record(self, raw, target) -> None:
    super().record(raw, target)
    probs = self.probs(raw)
    exp = (probs * self.support[...]).sum(axis=-1, keepdims=True)
    var = (probs * (self.support[...] - exp) ** 2).sum(axis=-1)
    self.metrics.dist("return_std", jnp.sqrt(var).mean(axis=0))


@register_component
class C51Critic(CategoricalCritic):
  """Atoms on a linspace; the bootstrap distribution is projected."""

  kind = "critic"
  config_type = C51CriticConfig

  def __init__(self, env_stats, config, **kwargs):
    super().__init__(env_stats, config, **kwargs)
    self.support = Buffer(jnp.linspace(self.v_min, self.v_max, config.num_bins, dtype=jnp.float32))

  def derived(self) -> dict:
    return {"v_min": self.v_min, "v_max": self.v_max,
            "bin_width": (self.v_max - self.v_min) / (self.config.num_bins - 1)}

  def bootstrap_members(self, next_raw):
    return self.probs(next_raw)

  def td_target(self, chunk_return, discount, bootstrap, entropy_bonus):
    return categorical_projection(bootstrap, self.support[...], chunk_return, discount,
                                  self.v_min, self.v_max, entropy_bonus)


@register_component
class HLGaussCritic(CategoricalCritic):
  """Bin centers as support; the scalar TD target is smoothed into a histogram."""

  kind = "critic"
  config_type = HLGaussCriticConfig

  def __init__(self, env_stats, config, **kwargs):
    super().__init__(env_stats, config, **kwargs)
    edges = jnp.linspace(self.v_min, self.v_max, config.num_bins + 1, dtype=jnp.float32)
    self.edges = Buffer(edges)
    self.support = Buffer((edges[:-1] + edges[1:]) / 2)
    self.sigma = config.sigma_to_bin_ratio * (self.v_max - self.v_min) / config.num_bins

  def derived(self) -> dict:
    return {"v_min": self.v_min, "v_max": self.v_max, "sigma": float(self.sigma),
            "bin_width": (self.v_max - self.v_min) / self.config.num_bins}

  def bootstrap_members(self, next_raw):
    return self.expectation_members(next_raw)

  def td_target(self, chunk_return, discount, bootstrap, entropy_bonus):
    return to_histogram(scalar_td_target(chunk_return, discount, bootstrap, entropy_bonus),
                        self.edges[...], self.sigma)
