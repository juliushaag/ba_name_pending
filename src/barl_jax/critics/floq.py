"""Flow matching critics (floq).

The scalar Q value is the endpoint of an Euler integration of a velocity
field.  A value call uses a fixed quadrature over the initial uniform noise so
that ``value`` is deterministic (and therefore safe to use in actor losses).
The TD update samples that same interval and trains the field on the linear
flow-matching target from the sampled noise to the Bellman target.
"""

import math
from dataclasses import asdict, dataclass, field

import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.components import _Static
from barl_jax.networks.ensemble import Ensemble
from barl_jax.networks.mlp import MLP, MLPConfig
from barl_jax.networks.time_embedding import TimeEmbedding, TimeEmbeddingConfig
from barl_jax.registration import register_component

from .base import CriticConfig, QCriticBase


def _histogram(value, v_min, v_max, num_bins, sigma):
  """Smooth scalar inputs into a normalized categorical representation.

  This is the HL-Gauss style input used by the paper.  It keeps the changing
  magnitude of the interpolant from becoming an uncontrolled MLP input while
  retaining a differentiable path from the integrated Q to the actor.
  """
  # Clamp only the encoder input to keep tail histogram probabilities finite.
  value = jnp.clip(value.reshape(-1, 1), v_min, v_max)
  edges = jnp.linspace(v_min, v_max, num_bins, dtype=jnp.float32)
  width = (v_max - v_min) / (num_bins - 1)
  std = jnp.maximum(sigma * width, 1e-6)
  cdf = 0.5 * (1.0 + jax.scipy.special.erf(
    (edges[None, :] - value) / (math.sqrt(2.0) * std)))
  mass = jnp.maximum(cdf[:, 1:] - cdf[:, :-1], 0.0)
  # Out-of-range interpolants can make the unnormalized mass tiny.  Keeping
  # the denominator positive gives finite gradients while preserving the
  # categorical collapse-prevention behavior at the boundaries.
  return mass / jnp.maximum(mass.sum(axis=-1, keepdims=True), 1e-8)


class FloqVectorField(nnx.Module):
  """One velocity field member, conditioned on ``(s, a, z_t, t)``."""

  def __init__(self, obs_dim, action_dim, *, latent_dim, layers, num_bins,
               q_min, q_max, sigma, time_dim, time_embed, time_kwargs,
               activation, norm, norm_input, norm_stem, norm_after_act,
               dropout, final_activation, residual, norm_kwargs, dtype, rngs):
    self.q_min = float(q_min)
    self.q_max = float(q_max)
    self.num_bins = int(num_bins)
    self.sigma = float(sigma)
    self.time = TimeEmbedding(time_dim=time_dim, kind=time_embed,
                              rngs=rngs, **time_kwargs)
    self.net = MLP(obs_dim + action_dim + num_bins - 1 + time_dim, 1,
                   latent_dim, layers, activation=activation, norm=norm,
                   norm_input=norm_input, norm_stem=norm_stem,
                   norm_after_act=norm_after_act, dropout=dropout,
                   final_activation=final_activation, residual=residual,
                   dtype=dtype, rngs=rngs, **(norm_kwargs or {}))

  def __call__(self, obs, action, z, time):
    z = _histogram(z, self.q_min, self.q_max, self.num_bins, self.sigma)
    t = self.time(time)
    return self.net(jnp.concatenate([obs, action, z, t], axis=-1))


class FloqEnsemble(nnx.Module):
  """An ensemble that also accepts one independent integration path/head.

  ``Ensemble`` handles the ordinary `(B, ...)` call.  During integration the
  latent endpoint is `(N, B, 1)`, so this wrapper vmaps each member together
  with its own path; otherwise broadcasting one path across all members would
  accidentally couple the ensemble trajectories.
  """

  def __init__(self, n_models, *args, rngs, **kwargs):
    self.members = Ensemble(n_models, FloqVectorField, *args, rngs=rngs, **kwargs)
    self.n_models = n_models

  def __call__(self, obs, action, z, time):
    if z.ndim == 2:
      return self.members(obs, action, z, time)

    @nnx.vmap(in_axes=(0, None, None, 0, 0), out_axes=0)
    def call(member, obs, action, z, time):
      return member(obs, action, z, time)

    if time.ndim == 2:
      time = jnp.broadcast_to(time, z.shape)
    return call(self.members.members, obs, action, z, time)


@dataclass(frozen=True)
class FloqCriticConfig(CriticConfig):
  class_name: str = "FloqCritic"
  latent_dim: int = 2048
  layers: int = 1
  n_models: int = 2
  mlp: MLPConfig = field(default_factory=lambda: MLPConfig(
    activation="gelu", norm="none", norm_stem=False, norm_input=False,
    norm_after_act=True, dtype="float32"))
  flow_steps: int = 8
  noise_samples: int = 8
  value_samples: int = 8
  noise_coverage: float = 0.1
  train_at_zero_only: bool = False
  num_bins: int = 51
  sigma: float = 16.0
  time_emb: str = "fourier"
  # Deliberately wider than the default actor/NC-critic time code: the
  # velocity must distinguish every Euler interval.
  time: TimeEmbeddingConfig = field(default_factory=lambda: TimeEmbeddingConfig(
    kind="fourier", time_dim=64, hidden=128, num_freqs=32,
    concat_input=True, spacing="geometric"))
  q_min: float | int | None = None
  q_max: float | int | None = None
  rho: float = 0.0
  pessimistic_target: bool = False
  polyak_tau: float = 0.005


@register_component
class FloqCritic(QCriticBase):
  """A flow-matching scalar Q critic compatible with SAC and AMDP.

  ``value`` returns the ensemble reduction of deterministic fixed-noise
  quadrature.  This is a small interface adaptation from the paper, whose
  implementation draws fresh initial noise at every value call: determinism
  is required here because actor objectives call ``frozen(critic).value`` and
  compare/reuse those prices inside a traced gradient.
  """

  kind = "critic"
  config_type = FloqCriticConfig

  def __init__(self, env_stats, config: FloqCriticConfig, *, rngs: nnx.Rngs, **kwargs):
    self.mlp_norm = config.mlp.norm
    time_values = config.time.to_runtime_dict()
    self.time_dim = time_values["time_dim"]
    self.time_embed = time_values["time_embed"]
    self._static_config = _Static({"time_kwargs": time_values["time_kwargs"]})
    self.rngs = rngs

    n_models = config.n_models
    latent_dim = config.latent_dim
    layers = config.layers

    if config.flow_steps < 1:
      raise ValueError(f"FloqCritic: flow_steps must be >= 1, got {config.flow_steps}")
    if config.num_bins < 2:
      raise ValueError(f"FloqCritic: num_bins must be >= 2, got {config.num_bins}")

    stats_min = env_stats.get("return_min")
    stats_max = env_stats.get("return_max")
    q_min = float(config.q_min if config.q_min is not None else
                  (stats_min if stats_min is not None else -1.0))
    q_max = float(config.q_max if config.q_max is not None else
                  (stats_max if stats_max is not None else 1.0))

    self.q_min, self.q_max = q_min, q_max
    self.noise_min = q_min * float(config.noise_coverage)
    self.noise_max = q_max * float(config.noise_coverage)

    self.q_function = FloqEnsemble(
      n_models, env_stats["obs_dim"], env_stats["act_dim"],
      latent_dim=latent_dim, layers=layers, num_bins=config.num_bins,
      q_min=q_min, q_max=q_max, sigma=config.sigma, time_dim=self.time_dim,
      time_embed=self.time_embed, time_kwargs=self.static("time_kwargs"),
      rngs=rngs, **asdict(config.mlp))

    super().__init__(env_stats, config, self.q_function, **kwargs)
    # QCriticBase sets shared optimizer fields; retain the concrete bounds
    # inferred from EnvStats for derived-config logging.
    self.q_min, self.q_max = q_min, q_max
    self.n_models = n_models

  def _integrate(self, field, obs, action, starts):
    """Euler integrate starts shaped `(B, M, 1)` to `(N, B, M, 1)`."""
    b, m = starts.shape[:2]
    obs = jnp.repeat(obs, m, axis=0)
    action = jnp.repeat(action, m, axis=0)
    z = starts.reshape(b * m, 1)
    flow_steps = self.config.flow_steps
    for i in range(flow_steps):
      t = jnp.full_like(z, i / flow_steps)
      if z.ndim == 3:
        t = jnp.broadcast_to(t, z.shape)
      z = z + field(obs, action, z, t) / flow_steps
    return z.reshape(self.n_models, b, m, 1)

  def _starts(self, obs, samples):
    # Midpoint quadrature is deterministic and avoids over-weighting an endpoint.
    points = self.noise_min + (jnp.arange(samples, dtype=jnp.float32) + 0.5) * ((self.noise_max - self.noise_min) / samples)
    return jnp.broadcast_to(points[None, :, None], (obs.shape[0], samples, 1))

  def raw(self, s, a):
    starts = self._starts(s, self.config.value_samples)
    return self._integrate(self.q_function, s, a, starts).mean(axis=2)

  def target_raw(self, s, a):
    starts = self._starts(s, self.config.value_samples)
    return jax.lax.stop_gradient(self._integrate(self.critic_target, s, a, starts).mean(axis=2)
                                 if self.has_target else self.raw(s, a))

  def reduce_value(self, raw):
    raw = raw.astype(jnp.float32)
    std = jnp.sqrt(jnp.maximum(raw.var(axis=0), 1e-8))
    return raw.mean(axis=0) - self.config.rho * std

  def regression_loss(self, raw, target):
    return ((raw.astype(jnp.float32) - jax.lax.stop_gradient(target).astype(jnp.float32)) ** 2).mean()
    

  def td_loss(self, obs, action, next_obs, next_action, chunk_return, terminated, horizon,
              entropy_bonus=0.0, *, key=None) -> dict:
    if key is None:
      # Policy train_step supplies a split key; standalone calls remain usable.
      key = self.rngs.noise()
    target_key, start_key, time_key = jax.random.split(key, 3)
    noise_samples = self.config.noise_samples
    target_starts = jax.random.uniform(
      target_key, (next_obs.shape[0], noise_samples, 1),
      minval=self.noise_min, maxval=self.noise_max)
    target_flow = self._integrate(
      self.critic_target if self.has_target else self.q_function,
      next_obs, next_action, target_starts).mean(axis=2)
    bootstrap = self.bootstrap_from_raw(target_flow)
    target = self.bellman_target(chunk_return, terminated, horizon, bootstrap,
                                 entropy_bonus)
    target = jax.lax.stop_gradient(target)

    starts = jax.random.uniform(
      start_key, (obs.shape[0], noise_samples, 1),
      minval=self.noise_min, maxval=self.noise_max)
    z0 = starts.reshape(-1, 1)
    obs_rep = jnp.repeat(obs, noise_samples, axis=0)
    action_rep = jnp.repeat(action, noise_samples, axis=0)
    if self.config.train_at_zero_only:
      t = jnp.zeros((z0.shape[0], 1), dtype=z0.dtype)
    else:
      t = jax.random.uniform(time_key, z0.shape)
    y = jnp.repeat(target, noise_samples, axis=0)
    zt = (1.0 - t) * z0 + t * y
    pred = self.q_function(obs_rep, action_rep, zt, t)
    velocity = y[None, ...] - z0[None, ...]
    flow_loss = ((pred.astype(jnp.float32) - velocity.astype(jnp.float32)) ** 2).mean()

    current = self.raw(obs, action)

    self.metrics.dist("critic", current.astype(jnp.float32).mean(axis=0))
    self.metrics.dist("q_target", jax.lax.stop_gradient(target).astype(jnp.float32))
    self.metrics.ensemble("ensemble", current.astype(jnp.float32), gap=self.pessimism_gap(current))

    self.metrics.scalar("flow_loss", flow_loss)
    self.metrics.scalar("td_error", ((current.mean(axis=0) - target) ** 2).mean())
    return {"critic": flow_loss}

  def derived(self) -> dict:
    return {"q_min": self.q_min, "q_max": self.q_max,
            "noise_min": self.noise_min, "noise_max": self.noise_max,
            "bin_width": (self.q_max - self.q_min) / (self.config.num_bins - 1)}
