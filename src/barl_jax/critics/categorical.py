import math

import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.config.toolkit import ConfigMixin
from barl_jax.hparam import HParam
from barl_jax.networks.registry import register_component
from barl_jax.components import Buffer

from .base import CategoricalCriticBase, ConcatQ, categorical_projection


@register_component
class C51Critic(CategoricalCriticBase):
  kind = "critic"
  entropy_coef = HParam(0.005)

  def __init__(self, env_stats, config: ConfigMixin, *, rngs: nnx.Rngs, **kwargs):
    self.resolve_hparams(config, env_stats=env_stats)
    n_models = config.get("n_models", int)
    v_min = config.get("v_min", float, default=env_stats["return_min"])
    v_max = config.get("v_max", float, default=env_stats["return_max"])

    q_function = ConcatQ(env_stats["obs_dim"] + env_stats["act_dim"], self.num_atoms,
                         config.get("latent_dim", int), config.get("layers", int),
                         n_models=n_models,
                         activation=self.activation, norm=self.norm, norm_input=self.norm_input,
                         norm_stem=self.norm_stem, norm_after_act=self.norm_after_act,
                         dropout=self.dropout, final_activation=self.final_activation,
                         residual=self.residual, norm_kwargs=self.hparam("norm_kwargs"),
                         dtype=self.dtype, rngs=rngs)

    super().__init__(env_stats, config, q_function,
                     jnp.linspace(v_min, v_max, self.num_atoms, dtype=jnp.float32), **kwargs)

    self.v_min, self.v_max = v_min, v_max

  def derived(self) -> dict:
    return {"v_min": self.v_min, "v_max": self.v_max,
            "bin_width": (self.v_max - self.v_min) / (self.num_atoms - 1)}

  def bootstrap_from_raw(self, next_raw):
    return jax.lax.stop_gradient(self.probs(next_raw))

  def bellman_target(self, reward, term, bootstrap, shift=0.0, horizon=None):
    if self.pessimistic_target:
      exp = (bootstrap * self.support[...]).sum(axis=-1, keepdims=True)
      bootstrap = jnp.take_along_axis(bootstrap, jnp.argmin(exp, axis=0, keepdims=True), axis=0)
    projected = categorical_projection(bootstrap, self.support[...], reward, term,
                                       self.gamma, shift, self.v_min, self.v_max,
                                       horizon=horizon)
    return projected.mean(axis=0)


@register_component
class HLGaussCritic(CategoricalCriticBase):
  kind = "critic"

  def __init__(self, env_stats, config: ConfigMixin, *, rngs: nnx.Rngs, **kwargs):
    self.resolve_hparams(config, env_stats=env_stats)
    n_models = config.get("n_models", int)

    v_min = config.get("v_min", float, default=env_stats["return_min"])
    v_max = config.get("v_max", float, default=env_stats["return_max"])
    edges = jnp.linspace(v_min, v_max, self.num_bins + 1, dtype=jnp.float32)

    q_function = ConcatQ(env_stats["obs_dim"] + env_stats["act_dim"], self.num_bins,
                         config.get("latent_dim", int), config.get("layers", int),
                         n_models=n_models,
                         activation=self.activation, norm=self.norm, norm_input=self.norm_input,
                         norm_stem=self.norm_stem, norm_after_act=self.norm_after_act,
                         dropout=self.dropout, final_activation=self.final_activation,
                         residual=self.residual, norm_kwargs=self.hparam("norm_kwargs"),
                         dtype=self.dtype, rngs=rngs)

    super().__init__(env_stats, config, q_function, (edges[:-1] + edges[1:]) / 2, **kwargs)

    self.v_min, self.v_max = v_min, v_max
    self.sigma    = self.sigma_to_bin_ratio * (self.v_max - self.v_min) / self.num_bins
    self.edges    = Buffer(edges)

  def derived(self) -> dict:
    return {"v_min": self.v_min, "v_max": self.v_max, "sigma": float(self.sigma),
            "bin_width": (self.v_max - self.v_min) / self.num_bins}

  def _to_histogram(self, value):
    value = jnp.clip(value.reshape(-1, 1), self.v_min, self.v_max)
    z     = (self.edges[...] - value) / (self.sigma * math.sqrt(2.0))
    cdf   = 0.5 * (1.0 + jax.scipy.special.erf(z))
    mass  = cdf[:, 1:] - cdf[:, :-1]
    return mass / jnp.maximum(mass.sum(axis=-1, keepdims=True), 1e-8)

  def bellman_target(self, reward, term, bootstrap, shift=0.0, horizon=None):
    return self._to_histogram(
      super().bellman_target(reward, term, bootstrap, shift, horizon=horizon))
