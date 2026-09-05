import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.config.toolkit import ConfigMixin
from barl_jax.hparam import Config, HParam
from barl_jax.networks.mlp import MLP_PRESETS
from barl_jax.registration import register_component

from .base import ConcatQ, Polyak, QCriticBase


@register_component
class EnsembleCritic(QCriticBase):
  kind = "critic"
  mlp_preset = Config("crossq", style=MLP_PRESETS)
  rho                = HParam(1.0)
  pessimistic_target = Config(True)

  def __init__(self, env_stats, config: ConfigMixin, *, rngs: nnx.Rngs, **kwargs):
    self.resolve_hparams(config, env_stats=env_stats)
    latent_dim, layers, n_models = (config.get(k, int) for k in ("latent_dim", "layers", "n_models"))

    q_function = ConcatQ(env_stats["obs_dim"] + env_stats["act_dim"], 1,
                         latent_dim, layers, n_models=n_models,
                         activation=self.activation, norm=self.norm, norm_input=self.norm_input,
                         norm_stem=self.norm_stem, norm_after_act=self.norm_after_act,
                         dropout=self.dropout, final_activation=self.final_activation,
                         residual=self.residual, norm_kwargs=self.hparam("norm_kwargs"), dtype=self.dtype,
                         rngs=rngs)

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
  mlp_preset = Config("layer_relu", style=MLP_PRESETS)
