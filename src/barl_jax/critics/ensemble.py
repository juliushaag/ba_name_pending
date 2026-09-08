import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.config.toolkit import ConfigMixin
from barl_jax.hparam import Config, ConfigDict
from barl_jax.registration import register_component

from .base import ConcatQ, Polyak, QCriticBase


@register_component
class EnsembleCritic(QCriticBase):
  kind = "critic"
  mlp = ConfigDict(
    activation='relu', norm='batchrenorm', norm_stem=True, norm_input=True,
    norm_after_act=True, dropout=0.0, final_activation=False, residual=False,
    norm_kwargs={'momentum': 0.99, 'warmup_steps': 100000, 'epsilon': 0.001, 'r_max': 3.0, 'd_max': 5.0},
    dtype='bfloat16')
  rho                = Config(1.0)
  pessimistic_target = Config(True)

  def __init__(self, env_stats, config: ConfigMixin, *, rngs: nnx.Rngs, **kwargs):
    self.resolve_config(config, env_stats=env_stats)
    latent_dim, layers, n_models = (config.get(k, int) for k in ("latent_dim", "layers", "n_models"))

    q_function = ConcatQ(env_stats["obs_dim"] + env_stats["act_dim"], 1,
                         latent_dim, layers, n_models=n_models, rngs=rngs, **self.mlp.todict())

    super().__init__(env_stats, config, q_function, **kwargs)

    self.n_models = n_models

  def reduce_value(self, raw):
    raw = raw.astype(jnp.float32)
    std = jnp.sqrt(jnp.maximum(raw.var(axis=0), 1e-8))
    return raw.mean(axis=0) - self.rho * std

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
class TargetEnsembleCritic(Polyak, EnsembleCritic):
  kind = "critic"
  mlp = ConfigDict(activation='relu', norm='layer', norm_stem=True, norm_input=True,
                   norm_after_act=True, dropout=0.0, final_activation=False,
                   residual=False, norm_kwargs={'epsilon': 1e-05}, dtype='bfloat16')
