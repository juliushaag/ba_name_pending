"""Flow matching critic (floq, arXiv:2509.06863).

Q is the endpoint of an Euler integration of a velocity field, trained on the
linear flow-matching target to the Bellman target (TD samples the initial noise).

With `distill` (default) actors price through a plain Q-ensemble regressed onto
the flow's mean return under fresh random noise; the flow only bootstraps. The
distillation target is stop-gradient, so the field is trained by flow matching
alone. Without it, `value` is a deterministic midpoint quadrature through the
Euler chain (differentiated by the actor loss).
"""

import math
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.networks.base import NetworkConfig
from barl_jax.networks.ensemble import Ensemble
from barl_jax.networks.mlp import MLPConfig
from barl_jax.networks.time_embedding import FourierTimeConfig, TimeEmbeddingConfig
from barl_jax.registration import register_component

from .base import EnsembleConfig, QCritic, QNetwork, TargetConfig, value_range
from .categorical import to_histogram
from .direct import DirectCriticConfig, ScalarValue


def _histogram(value, v_min, v_max, num_bins, sigma):
  """HL-Gauss encoding of the interpolant, keeping its changing magnitude from
  becoming an unbounded MLP input. `sigma` is in bin widths."""
  edges = jnp.linspace(v_min, v_max, num_bins, dtype=jnp.float32)
  return to_histogram(value, edges, max(sigma * (v_max - v_min) / (num_bins - 1), 1e-6))


class FloqVectorField(nnx.Module):
  """One velocity field member, conditioned on (s, a, z_t, t)."""

  def __init__(self, obs_dim, action_dim, *, latent_dim, layers, num_bins,
               hist_min, hist_max, sigma, time: TimeEmbeddingConfig, mlp: NetworkConfig, rngs):
    self.v_min, self.v_max = float(hist_min), float(hist_max)
    self.num_bins = int(num_bins)
    self.sigma = float(sigma)
    self.time = time.build(rngs=rngs)
    self.net = mlp.build(obs_dim + action_dim + num_bins - 1, 1, latent_dim, layers,
                         cond_dim=time.time_dim, rngs=rngs)

  def __call__(self, obs, action, z, time):
    z = _histogram(z, self.v_min, self.v_max, self.num_bins, self.sigma)
    return self.net((obs, action, z), (self.time(time),))


class FloqEnsemble(nnx.Module):
  """An `Ensemble` of fields that can also integrate one independent path per
  member: an `(N, B, 1)` latent is vmapped member-wise rather than broadcast,
  which would couple the members' trajectories."""

  def __init__(self, n_models, *args, rngs, **kwargs):
    self.members = Ensemble(n_models, FloqVectorField, *args, rngs=rngs, **kwargs)

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
class FloqCriticConfig(DirectCriticConfig):
  class_name: str = "FloqCritic"
  latent_dim: int = 512
  layers: int = 3
  mlp: NetworkConfig = field(default_factory=lambda: MLPConfig(
    activation="gelu", norm="layer", norm_stem=True, norm_input=False,
    norm_after_act=True, dtype="float32"))
  ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
  target: TargetConfig = field(default_factory=lambda: TargetConfig(kind="polyak",
                                                                     joint_forward=False))
  flow_steps: int = 8
  noise_samples: int = 8
  value_samples: int = 8
  noise_coverage: float = 0.1
  # Episode-return bounds can be orders of magnitude wider than a TD reward.
  # Cap the source distribution in scaled return units so its finite-sample
  # mean does not overwhelm bootstrap/distillation targets at initialization.
  # None restores the reference's unrestricted coverage of the value range.
  noise_limit: float | None = 1.0
  train_at_zero_only: bool = False
  distill: bool = True
  num_bins: int = 51
  sigma: float = 16.0
  # Wider than the NC-critic's level code: the velocity must tell every Euler interval apart.
  time: TimeEmbeddingConfig = field(default_factory=lambda: FourierTimeConfig(
    time_dim=64, hidden=128, num_freqs=32))
  reward_offset: float = 0.01
  v_min: float | None = None
  v_max: float | None = None


@dataclass(frozen=True)
class FlowGeometry:
  """Where the flow lives, in reward-scaled return units: the value range, the
  source distribution `z0` is drawn from, and the (slightly wider) range the
  interpolant is histogram-encoded over."""

  v_min: float
  v_max: float
  noise_min: float
  noise_max: float
  hist_min: float
  hist_max: float

  @classmethod
  def build(cls, env_stats, config) -> "FlowGeometry":
    v_min, v_max = value_range(env_stats, config, fallback=(-1.0, 1.0))
    if not (math.isfinite(v_min) and math.isfinite(v_max) and v_min < v_max):
      raise ValueError("FloqCritic: value bounds must be finite with v_min < v_max")
    noise_min = v_min * float(config.noise_coverage)
    noise_max = v_max * float(config.noise_coverage)
    if config.noise_limit is not None:
      magnitude = max(abs(noise_min), abs(noise_max))
      scale = min(1.0, config.noise_limit / magnitude)
      noise_min *= scale
      noise_max *= scale
    # Per-step reward slack widens only the encoding support, so clipped
    # interpolants keep a nonzero gradient at the edges.
    pad = config.reward_offset * config.reward_scale / (1.0 - config.gamma)
    return cls(v_min, v_max, noise_min, noise_max, v_min - pad, v_max + pad)


@register_component
class FloqCritic(ScalarValue, QCritic):
  """Scalar flow-matching critic. Shares the target net, reductions and
  Bellman target with the other scalar critics; only the network, `raw` and the
  regression objective differ."""

  kind = "critic"
  config_type = FloqCriticConfig

  def __init__(self, env_stats, config, *, rngs: nnx.Rngs, **kwargs):
    self.rngs = rngs
    super().__init__(env_stats, config, rngs=rngs, **kwargs)
    if config.distill:
      self.q_net = QNetwork(env_stats["obs_dim"], env_stats["act_dim"], 1, config, rngs=rngs)

  @classmethod
  def config_problems(cls, config) -> list[str]:
    problems = super().config_problems(config)
    if config.noise.enabled:
      problems.append("noise conditioning is not supported")
    if config.weight_norm:
      problems.append("weight_norm is not supported")
    if config.flow_steps < 1:
      problems.append(f"flow_steps must be >= 1, got {config.flow_steps}")
    if config.num_bins < 2:
      problems.append(f"num_bins must be >= 2, got {config.num_bins}")
    for name in ("noise_samples", "value_samples"):
      if getattr(config, name) < 1:
        problems.append(f"{name} must be >= 1")
    for name in ("noise_coverage", "sigma"):
      value = getattr(config, name)
      if not (math.isfinite(value) and value > 0):
        problems.append(f"{name} must be finite and > 0")
    if config.noise_limit is not None and not (
        math.isfinite(config.noise_limit) and config.noise_limit > 0):
      problems.append("noise_limit must be None or finite and > 0")
    if not (math.isfinite(config.gamma) and 0 <= config.gamma < 1):
      problems.append("gamma must be finite and satisfy 0 <= gamma < 1")
    if not (math.isfinite(config.reward_offset) and config.reward_offset >= 0):
      problems.append("reward_offset must be finite and >= 0")
    return problems

  def build_network(self, env_stats, config, *, rngs):
    # Called from `QCritic.__init__`, after `config_problems` has vetted the config.
    self.geometry = FlowGeometry.build(env_stats, config)
    return FloqEnsemble(config.ensemble.n_models, env_stats["obs_dim"], env_stats["act_dim"],
                        latent_dim=config.latent_dim, layers=config.layers,
                        num_bins=config.num_bins, hist_min=self.geometry.hist_min,
                        hist_max=self.geometry.hist_max,
                        sigma=config.sigma, time=config.time, mlp=config.mlp, rngs=rngs)

  v_min = property(lambda self: self.geometry.v_min)
  v_max = property(lambda self: self.geometry.v_max)
  noise_min = property(lambda self: self.geometry.noise_min)
  noise_max = property(lambda self: self.geometry.noise_max)
  hist_min = property(lambda self: self.geometry.hist_min)
  hist_max = property(lambda self: self.geometry.hist_max)

  def _integrate(self, field, obs, action, starts):
    """Euler-integrate starts (B, M, 1) to (N, B, M, 1)."""
    b, m = starts.shape[:2]
    obs = jnp.repeat(obs, m, axis=0)
    action = jnp.repeat(action, m, axis=0)
    z = starts.reshape(b * m, 1)
    flow_steps = self.config.flow_steps
    for i in range(flow_steps):
      t = jnp.full_like(z, i / flow_steps)
      z = z + field(obs, action, z, t) / flow_steps
    return z.reshape(self.n_models, b, m, 1)

  def _starts(self, obs, samples):
    # Midpoint quadrature: deterministic, no endpoint over-weighted.
    step = (self.noise_max - self.noise_min) / samples
    points = self.noise_min + (jnp.arange(samples, dtype=jnp.float32) + 0.5) * step
    return jnp.broadcast_to(points[None, :, None], (obs.shape[0], samples, 1))

  def raw(self, s, a, level=None):
    if self.config.distill:
      return self.q_net(s, a)
    return self._integrate(self.net, s, a, self._starts(s, self.config.value_samples)).mean(axis=2)

  def td_loss(self, obs, action, next_obs, next_action, chunk_return, terminated, horizon,
              entropy_bonus=0.0, *, key=None) -> dict:
    key = self.rngs.noise() if key is None else key
    target_key, start_key, time_key, distill_key = jax.random.split(key, 4)
    samples = self.config.noise_samples
    uniform = lambda k, n: jax.random.uniform(k, (n, samples, 1), minval=self.noise_min,
                                              maxval=self.noise_max)

    field_net = self.net_target if self.has_target else self.net
    next_raw = self._integrate(field_net, next_obs, next_action,
                               uniform(target_key, next_obs.shape[0])).mean(axis=2)
    target = self.bellman_target(chunk_return, terminated, horizon,
                                 self.bootstrap(jax.lax.stop_gradient(next_raw)), entropy_bonus)

    z0 = uniform(start_key, obs.shape[0]).reshape(-1, 1)
    obs_rep = jnp.repeat(obs, samples, axis=0)
    action_rep = jnp.repeat(action, samples, axis=0)
    t = (jnp.zeros_like(z0) if self.config.train_at_zero_only
         else jax.random.uniform(time_key, z0.shape))
    y = jnp.repeat(target, samples, axis=0)
    pred = self.net(obs_rep, action_rep, (1.0 - t) * z0 + t * y, t)
    # Summed over the noise samples (mean over batch and members), as in the reference.
    flow_loss = samples * self.regression_loss(pred, (y - z0)[None, ...])

    losses = {"critic": flow_loss}
    if self.config.distill:
      flow_return = self._integrate(self.net, obs, action, uniform(distill_key, obs.shape[0]))
      flow_return = jax.lax.stop_gradient(flow_return.mean(axis=(0, 2)))
      current = self.q_net(obs, action)
      distill_loss = ((current - flow_return[None]) ** 2).mean()
      self.metrics.scalar("distill_loss", distill_loss)
      losses["critic"] = flow_loss + distill_loss
    else:
      current = self.raw(obs, action)
    self.record(current, target)
    self.metrics.scalar("flow_loss", flow_loss)
    self.metrics.scalar("td_error", ((current.mean(axis=0) - target) ** 2).mean())
    return losses

  def derived(self) -> dict:
    return {"v_min": self.v_min, "v_max": self.v_max,
            "noise_min": self.noise_min, "noise_max": self.noise_max,
            "bin_width": (self.hist_max - self.hist_min) / (self.config.num_bins - 1)}
