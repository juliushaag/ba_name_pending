"""The composable Q critic.

A value head (Direct, HLGauss, C51) is a subclass; every other axis is config
and works for every head:

  target.kind           none (CrossQ: bootstrap through the frozen live net) | polyak
  target.joint_forward  forward [(s,a), (s',a')] as one batch through each net
  mlp.norm              none | layer | batchrenorm | batch
  ensemble.*_reduce     mean | min over the n_models members
  noise.enabled         noise-conditioned TD (NCLQL) on top of plain TD
  weight_norm           XQC's unit-norm weight projection after every step
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.components import TARGET_SUFFIX, Buffer, Critic, frozen
from barl_jax.critics.noise import NoiseConditioning
from barl_jax.networks.base import NetworkConfig, normalize_weights
from barl_jax.networks.ensemble import Ensemble
from barl_jax.networks.mlp import MLPConfig
from barl_jax.networks.schedule import ScheduleConfig
from barl_jax.networks.time_embedding import FourierTimeConfig, TimeEmbeddingConfig
from barl_jax.optimization import CriticOptim, CriticOptimConfig, adam

REDUCTIONS = ("mean", "min")
TARGET_KINDS = ("none", "polyak")
BATCH_STAT_NORMS = ("batch", "batchrenorm")


def crossq_mlp() -> MLPConfig:
  """CrossQ (Bhatt et al., ICLR 2024): post-activation BatchRenorm with input
  and stem normalized, so the joint [(s,a), (s',a')] forward shares statistics."""
  return MLPConfig(activation="gelu", norm="batchrenorm", norm_stem=True, norm_input=True,
                   norm_after_act=True, norm_kwargs={"momentum": 0.99, "warmup_steps": 100000,
                   "epsilon": 0.001, "r_max": 3.0, "d_max": 5.0}, dtype="bfloat16")


def target_mlp() -> MLPConfig:
  """`crossq_mlp` with LayerNorm, for separate target-net forwards."""
  return MLPConfig(activation="gelu", norm="layer", norm_stem=True, norm_input=True,
                   norm_after_act=True, norm_kwargs={"epsilon": 1e-5}, dtype="bfloat16")


def xqc_mlp() -> MLPConfig:
  """XQC (arXiv:2509.25174): pre-activation BatchNorm, input included."""
  return MLPConfig(activation="relu", norm="batch", norm_input=True, norm_stem=True,
                   norm_after_act=False, norm_kwargs={"momentum": 0.99, "epsilon": 0.001},
                   dtype="float32")


@dataclass(frozen=True)
class TargetConfig:
  kind: str = "none"
  tau: float = 0.005
  delay: int = 1
  joint_forward: bool = True


@dataclass(frozen=True)
class EnsembleConfig:
  n_models: int = 2
  value_reduce: str = "mean"
  target_reduce: str = "mean"


@dataclass(frozen=True)
class NoiseConditioningConfig:
  enabled: bool = False
  schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
  time: TimeEmbeddingConfig = field(default_factory=FourierTimeConfig)
  smoothing_coef: float = 1.0


@dataclass(frozen=True)
class QCriticConfig(CriticOptimConfig):
  gamma: float = 0.99
  reward_scale: float = 1.0
  latent_dim: int = 2048
  layers: int = 1
  mlp: NetworkConfig = field(default_factory=crossq_mlp)
  target: TargetConfig = field(default_factory=TargetConfig)
  ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
  noise: NoiseConditioningConfig = field(default_factory=NoiseConditioningConfig)
  weight_norm: bool = False


def n_step_discount(gamma, horizon, terminated, dtype):
  """(1 - terminated) * gamma**horizon, shaped (B, 1). `horizon` is the
  executed plan length and is always passed explicitly."""
  horizon = jnp.asarray(horizon, dtype).reshape(-1, 1)
  terminated = terminated.reshape(-1, 1).astype(dtype)
  return (1.0 - terminated) * jnp.power(jnp.asarray(gamma, dtype), horizon)


def scalar_td_target(chunk_return, discount, bootstrap, entropy_bonus=0.0):
  return chunk_return + discount * (bootstrap + entropy_bonus)


def reduce_members(x, expectation, kind: str):
  """Reduce the member axis of `x` (N, B, K). `min` selects, per sample, the
  whole member with the lowest `expectation` (N, B, 1) -- a scalar min for a
  scalar `x`, the lower head's distribution for a categorical one."""
  if kind == "mean":
    return x.mean(axis=0)
  idx = jnp.argmin(expectation, axis=0, keepdims=True)
  return jnp.take_along_axis(x, idx, axis=0)[0]


def value_range(env_stats, config, fallback=None) -> tuple[float, float]:
  """Configured bounds win; otherwise the env's return range in reward-scaled units."""
  bounds = []
  for configured, stat in ((config.v_min, "return_min"), (config.v_max, "return_max")):
    if configured is not None:
      bounds.append(float(configured))
    elif env_stats.get(stat) is not None:
      bounds.append(float(env_stats[stat]) * config.reward_scale)
    elif fallback is not None:
      bounds.append(fallback[len(bounds)])
    else:
      raise KeyError(f"{stat} is missing from env_stats and not configured")
  return tuple(bounds)


class QNetwork(nnx.Module):
  """An `Ensemble` of MLPs over concat(s, a[, encoder(level)]). The level
  encoder lives inside so a target clone carries its own copy."""

  def __init__(self, obs_dim: int, act_dim: int, out_dim: int, config: QCriticConfig,
               *, rngs: nnx.Rngs):
    time = config.noise.time if config.noise.enabled else None
    self.encoder = time.build(rngs=rngs) if time else None
    in_dim = obs_dim + act_dim + (time.time_dim if time else 0)
    self.ensemble = Ensemble(config.ensemble.n_models, config.mlp.build, in_dim, out_dim,
                             config.latent_dim, config.layers, rngs=rngs)

  def __call__(self, s, a, level=None):
    x = [s, a]
    if self.encoder is not None:
      x.append(self.encoder(level.astype(jnp.float32)[:, None]))
    return self.ensemble(jnp.concatenate(x, axis=-1))


class QCritic(CriticOptim, Critic):
  """Shared TD machinery. A head implements `out_dim`, `expectation_members`,
  `bootstrap_members`, `td_target`, `regression_loss`, `soft_target` and
  `target_expectation`; raw outputs are always (n_models, B, out_dim)."""

  config_type = QCriticConfig
  learning_objective = "td"

  def __init__(self, env_stats, config, *, rngs: nnx.Rngs, **kwargs):
    super().__init__(env_stats, config, **kwargs)
    problems = self.config_problems(config)
    if problems:
      raise ValueError(f"{type(self).__name__}: " + "; ".join(problems))

    self.noise = NoiseConditioning(env_stats, config.noise) if config.noise.enabled else None
    self.net = self.build_network(env_stats, config, rngs=rngs)
    if config.weight_norm:
      normalize_weights(self.net.ensemble.members)
    if self.has_target:
      self.net_target = nnx.clone(self.net)
      self.target_steps = Buffer(jnp.zeros((), jnp.int32))

  @classmethod
  def config_problems(cls, config) -> list[str]:
    problems = []
    if config.target.kind not in TARGET_KINDS:
      problems.append(f"target.kind={config.target.kind!r} not in {TARGET_KINDS}")
    for name in ("value_reduce", "target_reduce"):
      if getattr(config.ensemble, name) not in REDUCTIONS:
        problems.append(f"ensemble.{name}={getattr(config.ensemble, name)!r} not in {REDUCTIONS}")
    if config.mlp.norm in BATCH_STAT_NORMS and not config.target.joint_forward:
      problems.append(f"mlp.norm={config.mlp.norm!r} needs target.joint_forward: batch statistics "
                      "must be shared between the (s,a) and (s',a') halves")
    return problems

  @classmethod
  def learning_objective_for(cls, config) -> str:
    return "noise_td" if config.noise.enabled else "td"

  def build_network(self, env_stats, config, *, rngs):
    return QNetwork(env_stats["obs_dim"], env_stats["act_dim"], self.out_dim(config), config,
                    rngs=rngs)

  @property
  def has_target(self) -> bool:
    return self.config.target.kind == "polyak"

  @property
  def n_models(self) -> int:
    return self.config.ensemble.n_models

  @property
  def supports_bon(self) -> bool:
    return self.noise is None

  def create_optimizer(self) -> dict:
    c = self.config
    return {"critic": adam(c.lr, c.adam_b1, c.grad_clip, c.grad_clip_mode)}

  # ---- pricing -----------------------------------------------------------

  def _level(self, s, level):
    """Noise level per row; the clean level unless given. None without noise."""
    if self.noise is None:
      return None
    level = self.noise.clean_level if level is None else level
    return jnp.broadcast_to(jnp.asarray(level, jnp.int32), (s.shape[0],))

  def raw(self, s, a, level=None):
    return self.net(s, a, self._level(s, level))

  def value(self, s, a, level=None):
    return self.reduce_value(self.raw(s, a, level))

  def expectation(self, s, a, level=None):
    return self.expectation_members(self.raw(s, a, level)).mean(axis=0)

  def reduce_value(self, raw):
    exp = self.expectation_members(raw)
    return reduce_members(exp, exp, self.config.ensemble.value_reduce)

  def pessimism_gap(self, raw):
    return self.expectation_members(raw).mean(axis=0) - self.reduce_value(raw)

  # ---- TD ----------------------------------------------------------------

  @staticmethod
  def _joint(net, items):
    """One forward over the concatenated (s, a, level) items, split back per item."""
    cat = lambda i: None if items[0][i] is None else jnp.concatenate([x[i] for x in items])
    joint = net(cat(0), cat(1), cat(2))
    bounds = [0]
    for s, _, _ in items:
      bounds.append(bounds[-1] + s.shape[0])
    return [joint[..., lo:hi, :] for lo, hi in zip(bounds[:-1], bounds[1:])]

  def _forward(self, items, next_item):
    """Live predictions for `items` and the stop-gradient bootstrap forward."""
    # Sharing one batch only matters for batch-statistic norms; otherwise the extra
    # passes over the other halves are pure waste when a target net exists.
    shared = self.config.target.joint_forward and (
      self.config.mlp.norm in BATCH_STAT_NORMS or not self.has_target)
    if shared:
      *preds, next_raw = self._joint(self.net, items + [next_item])
      if self.has_target:
        next_raw = self._joint(self.net_target, items + [next_item])[-1]
    else:
      preds = self._joint(self.net, items)
      next_raw = (self.net_target if self.has_target else frozen(self.net))(*next_item)
    return preds, jax.lax.stop_gradient(next_raw)

  def bootstrap(self, next_raw):
    return reduce_members(self.bootstrap_members(next_raw), self.expectation_members(next_raw),
                          self.config.ensemble.target_reduce)

  def bellman_target(self, chunk_return, terminated, horizon, bootstrap, entropy_bonus=0.0):
    chunk_return = chunk_return.reshape(-1, 1) * self.config.reward_scale
    discount = n_step_discount(self.config.gamma, horizon, terminated, chunk_return.dtype)
    return jax.lax.stop_gradient(self.td_target(chunk_return, discount, bootstrap, entropy_bonus))

  def td_loss(self, obs, action, next_obs, next_action, chunk_return, terminated, horizon,
              entropy_bonus=0.0, *, key=None) -> dict:
    """chunk_return: (B,) return already discounted within the executed
    horizon; terminated: (B,) true termination (never truncation); horizon:
    (B,) executed plan length; entropy_bonus: (B, 1) or 0.0, added to the
    bootstrap before discounting."""
    items = [(obs, action, self._level(obs, None))]
    if self.noise is not None:
      if key is None:
        raise ValueError("noise-conditioned TD requires an explicit random key")
      noisy, levels = self.noise.perturb(action, key)
      items.append((obs, noisy, levels))

    preds, next_raw = self._forward(items, (next_obs, next_action, self._level(next_obs, None)))
    target = self.bellman_target(chunk_return, terminated, horizon, self.bootstrap(next_raw),
                                 entropy_bonus)
    loss = self.regression_loss(preds[0], target)
    self.record(preds[0], target)

    if self.noise is not None:
      smoothing = self.regression_loss(preds[1], jax.lax.stop_gradient(self.soft_target(preds[0])))
      self.metrics.scalar("clean_td_loss", loss)
      self.metrics.scalar("smoothing_loss", smoothing)
      loss = loss + self.config.noise.smoothing_coef * smoothing
    return {"critic": loss}

  def record(self, raw, target) -> None:
    exp = self.expectation_members(raw)
    self.metrics.dist("critic", exp.mean(axis=0))
    self.metrics.dist("q_target", self.target_expectation(target))
    self.metrics.ensemble("ensemble", exp, gap=self.pessimism_gap(raw))

  def after_step(self):
    if self.config.weight_norm:
      # Before the blend: the target tracks the projected live weights.
      normalize_weights(self.net.ensemble.members)
    if not self.has_target:
      return

    tau, delay = self.config.target.tau, self.config.target.delay
    self.target_steps[...] += 1
    live = nnx.state(self.net, nnx.Param)
    tgt = nnx.state(self.net_target, nnx.Param)
    blend = lambda t, l: t + tau * (l - t)
    if delay > 1:
      due = (self.target_steps[...] % delay) == 0
      step = jax.tree.map(lambda t, l: jnp.where(due, blend(t, l), t), tgt, live)
    else:
      step = jax.tree.map(blend, tgt, live)
    nnx.update(self.net_target, step)
    self.metrics.gap(f"critic{TARGET_SUFFIX}", live, tgt)

  # ---- head interface ----------------------------------------------------

  @classmethod
  def out_dim(cls, config) -> int:
    raise NotImplementedError

  def expectation_members(self, raw):
    """(N, B, 1) float32 expected value per member."""
    raise NotImplementedError

  def bootstrap_members(self, next_raw):
    """(N, B, K) per-member quantity `td_target` consumes."""
    raise NotImplementedError

  def td_target(self, chunk_return, discount, bootstrap, entropy_bonus):
    raise NotImplementedError

  def regression_loss(self, raw, target):
    raise NotImplementedError

  def soft_target(self, raw):
    """Ensemble-mean prediction, in the representation `regression_loss` fits."""
    raise NotImplementedError

  def target_expectation(self, target):
    raise NotImplementedError
