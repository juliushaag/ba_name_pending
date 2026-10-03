from dataclasses import dataclass, field

import jax.numpy as jnp

from barl_jax.registration import register_component

from .base import EnsembleConfig, QCritic, QCriticConfig, scalar_td_target


@dataclass(frozen=True)
class DirectCriticConfig(QCriticConfig):
  class_name: str = "DirectCritic"
  ensemble: EnsembleConfig = field(default_factory=lambda: EnsembleConfig(n_models=1))


@dataclass(frozen=True)
class EnsembleCriticConfig(DirectCriticConfig):
  class_name: str = "EnsembleCritic"
  ensemble: EnsembleConfig = field(default_factory=lambda: EnsembleConfig(
    n_models=2, value_reduce="min", target_reduce="min"))


class ScalarValue:
  """The scalar value head: Q is one output per member, regressed with MSE.
  Mixed into every critic whose network predicts the value itself."""

  @classmethod
  def out_dim(cls, config) -> int:
    return 1

  def expectation_members(self, raw):
    return raw.astype(jnp.float32)

  def bootstrap_members(self, next_raw):
    return self.expectation_members(next_raw)

  def td_target(self, chunk_return, discount, bootstrap, entropy_bonus):
    return scalar_td_target(chunk_return, discount, bootstrap, entropy_bonus)

  def regression_loss(self, raw, target):
    return ((raw.astype(jnp.float32) - target.astype(jnp.float32)) ** 2).mean()

  def soft_target(self, raw):
    return raw.mean(axis=0)

  def target_expectation(self, target):
    return target.astype(jnp.float32)


@register_component
class DirectCritic(ScalarValue, QCritic):
  """Scalar Q regressed with MSE."""

  kind = "critic"
  config_type = DirectCriticConfig


@register_component
class EnsembleCritic(DirectCritic):
  """`DirectCritic` preset: two members, min-reduced (clipped double Q)."""

  kind = "critic"
  config_type = EnsembleCriticConfig
