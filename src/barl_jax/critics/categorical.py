import math

import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.components import Buffer
from barl_jax.config.toolkit import ConfigMixin
from barl_jax.hparam import Config, HParam
from barl_jax.networks.mlp import MLP_PRESETS
from barl_jax.registration import register_component

from .base import ConcatQ, QCriticBase, n_step_discount


class CriticDistConfig:
  num_atoms          = Config(101)
  num_bins           = Config(151)
  sigma_to_bin_ratio = Config(0.75)
  entropy_coef       = HParam(0.0)
  rho                = HParam(0.0)


class CategoricalCriticBase(QCriticBase, CriticDistConfig):
  mlp_preset = Config("crossq", style=MLP_PRESETS)

  def __init__(self, env_stats, config: ConfigMixin, q_function, support, **kwargs):
    super().__init__(env_stats, config, q_function, **kwargs)

    self.support = Buffer(support)
    self.n_models = config.get("n_models", int)

  def probs(self, raw):
    return jax.nn.softmax(raw.astype(jnp.float32), axis=-1)

  def expectation_from_raw(self, raw):
    return (self.probs(raw) * self.support[...]).sum(axis=-1, keepdims=True)

  def reduce_value(self, raw):
    exp = self.expectation_from_raw(raw).astype(jnp.float32)
    std = jnp.sqrt(jnp.maximum(exp.var(axis=0), 1e-8))
    return exp.mean(axis=0) - self.rho * std

  def target_from_raw(self, raw):
    return jax.lax.stop_gradient(self.probs(raw).mean(axis=0))

  def regression_loss(self, raw, target):
    logp = jax.nn.log_softmax(raw.astype(jnp.float32), axis=-1)
    tgt  = jax.lax.stop_gradient(target).astype(jnp.float32)

    reduce = lambda x: x.sum(axis=-1).mean(axis=-1).sum()

    loss = reduce(-(tgt * logp))
    if self.entropy_coef:
      loss += self.entropy_coef * reduce(jnp.exp(logp) * logp)
    return loss

  def record(self, raw, target) -> None:
    tgt     = jax.lax.stop_gradient(target).astype(jnp.float32)
    support = self.support[...]
    probs   = self.probs(raw)
    exp     = (probs * support).sum(axis=-1, keepdims=True)

    self.metrics.dist("critic", exp.mean(axis=0))
    self.metrics.dist("q_target", (tgt * support).sum(axis=-1, keepdims=True))
    self.metrics.ensemble("ensemble", exp, gap=self.pessimism_gap(raw))

    var = (probs * (support - exp) ** 2).sum(axis=-1)
    self.metrics.dist("return_std", jnp.sqrt(var).mean(axis=0))


def categorical_projection(next_probs, atoms, chunk_return, terminated, horizon, gamma,
                           v_min, v_max, entropy_bonus=0.0):
  """Builds the transformed support `chunk_return + bootstrap_discount*(atoms +
  entropy_bonus)` explicitly, then projects `next_probs` onto the fixed atom
  grid by linear interpolation -- the categorical distribution is never
  collapsed to its scalar expectation before projecting. Same discount as
  `QCriticBase.bellman_target` (`n_step_discount`), a different target
  representation."""
  num_atoms = atoms.shape[0]
  delta_z   = (v_max - v_min) / (num_atoms - 1)

  chunk_return = chunk_return.reshape(-1, 1)
  discount = n_step_discount(gamma, horizon, terminated, chunk_return.dtype)
  tz = jnp.clip(chunk_return + discount * (atoms.reshape(1, -1) + entropy_bonus),
                v_min, v_max)

  b = (tz - v_min) / delta_z
  l = jnp.clip(jnp.floor(b), 0, num_atoms - 1).astype(jnp.int32)
  u = jnp.clip(jnp.ceil(b),  0, num_atoms - 1).astype(jnp.int32)

  on_atom = (l == u)
  m_l = jnp.where(on_atom, next_probs, next_probs * (u.astype(b.dtype) - b))
  m_u = jnp.where(on_atom, 0.0,        next_probs * (b - l.astype(b.dtype)))

  proj = jnp.zeros_like(next_probs)
  batch_idx = jnp.arange(next_probs.shape[-2])[:, None]
  proj = proj.at[..., batch_idx, l].add(m_l)
  proj = proj.at[..., batch_idx, u].add(m_u)
  return proj


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

  def bellman_target(self, chunk_return, terminated, horizon, bootstrap, entropy_bonus=0.0):
    if self.pessimistic_target:
      exp = (bootstrap * self.support[...]).sum(axis=-1, keepdims=True)
      bootstrap = jnp.take_along_axis(bootstrap, jnp.argmin(exp, axis=0, keepdims=True), axis=0)
    projected = categorical_projection(bootstrap, self.support[...], chunk_return, terminated,
                                       horizon, self.gamma, self.v_min, self.v_max,
                                       entropy_bonus=entropy_bonus)
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

  def bellman_target(self, chunk_return, terminated, horizon, bootstrap, entropy_bonus=0.0):
    return self._to_histogram(
      super().bellman_target(chunk_return, terminated, horizon, bootstrap, entropy_bonus))
