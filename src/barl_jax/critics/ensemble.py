from dataclasses import asdict, dataclass, field

import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.networks.mlp import MLPConfig
from barl_jax.registration import register_component

from .base import ConcatQ, CriticConfig, QCriticBase, crossq_mlp, target_mlp


@dataclass(frozen=True)
class EnsembleCriticConfig(CriticConfig):
  class_name: str = "EnsembleCritic"
  latent_dim: int = 2048
  layers: int = 1
  n_models: int = 2
  mlp: MLPConfig = field(default_factory=crossq_mlp)
  rho: float = 1.0
  pessimistic_target: bool = True


@dataclass(frozen=True)
class TargetEnsembleCriticConfig(EnsembleCriticConfig):
  class_name: str = "TargetEnsembleCritic"
  polyak_tau: float = 0.005
  mlp: MLPConfig = field(default_factory=target_mlp)


@register_component
class EnsembleCritic(QCriticBase):
  kind = "critic"
  config_type = EnsembleCriticConfig

  def __init__(self, env_stats, config, *, rngs: nnx.Rngs, **kwargs):
    self.mlp_norm = config.mlp.norm
    latent_dim, layers, n_models = config.latent_dim, config.layers, config.n_models

    q_function = ConcatQ(env_stats["obs_dim"] + env_stats["act_dim"], 1,
                         latent_dim, layers, n_models=n_models, rngs=rngs, **asdict(config.mlp))

    super().__init__(env_stats, config, q_function, **kwargs)

    self.n_models = n_models

  def reduce_value(self, raw):
    raw = raw.astype(jnp.float32)
    std = jnp.sqrt(jnp.maximum(raw.var(axis=0), 1e-8))
    return raw.mean(axis=0) - self.config.rho * std

  def regression_loss(self, raw, target):
    pred = raw.astype(jnp.float32)
    tgt  = jax.lax.stop_gradient(target).astype(jnp.float32)
    return ((pred - tgt) ** 2).mean()

  def record(self, raw, target) -> None:
    pred = raw.astype(jnp.float32)

    self.metrics.dist("critic", pred.mean(axis=0))
    self.metrics.dist("q_target", jax.lax.stop_gradient(target).astype(jnp.float32))
    self.metrics.ensemble("ensemble", pred, gap=self.pessimism_gap(raw))


@register_component
class TargetEnsembleCritic(EnsembleCritic):
  kind = "critic"
  config_type = TargetEnsembleCriticConfig
