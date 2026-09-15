"""Reducers over N independently-generated candidates that need no scorer:
Mean, Median, Axis-Shuffle. None of their results describe any single
candidate, so none keep `extras` -- a caller wanting extras uses
`DirectExecutor` or one of `executors.scored`'s score-guided executors."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from barl_jax.registration import register_executor

from .base import ActionExecutor, ExecutorConfig, _generate, _validate_num_candidates


@dataclass(frozen=True)
class MeanExecutorConfig(ExecutorConfig):
  class_name: str = "MeanExecutor"
  num_candidates: int = 8

  def __post_init__(self):
    _validate_num_candidates(self.num_candidates)


@register_executor
class MeanExecutor(ActionExecutor):
  """Elementwise mean of `num_candidates` independent samples."""

  config_type = MeanExecutorConfig

  @property
  def blends_elementwise(self) -> bool:
    return True

  def select(self, infer_step, score_fn, observation, key):
    actions, _ = _generate(infer_step, observation, key, self.config.num_candidates)
    return jnp.mean(actions, axis=0), {}


@dataclass(frozen=True)
class MedianExecutorConfig(ExecutorConfig):
  class_name: str = "MedianExecutor"
  num_candidates: int = 8

  def __post_init__(self):
    _validate_num_candidates(self.num_candidates)


@register_executor
class MedianExecutor(ActionExecutor):
  """Elementwise median of `num_candidates` independent samples -- unlike
  `MeanExecutor`, a single wild candidate does not move the result the way
  it would move an average."""

  config_type = MedianExecutorConfig

  @property
  def blends_elementwise(self) -> bool:
    return True

  def select(self, infer_step, score_fn, observation, key):
    actions, _ = _generate(infer_step, observation, key, self.config.num_candidates)
    return jnp.median(actions, axis=0), {}


@dataclass(frozen=True)
class AxisShuffleExecutorConfig(ExecutorConfig):
  class_name: str = "AxisShuffleExecutor"
  num_candidates: int = 8

  def __post_init__(self):
    _validate_num_candidates(self.num_candidates)


@register_executor
class AxisShuffleExecutor(ActionExecutor):
  """Every `(batch, chunk, act_dim)` entry independently picks one of
  `num_candidates` generated plans -- a per-axis recombination of all N, not
  any single candidate's action, like `MeanExecutor` but drawn rather than
  averaged."""

  config_type = AxisShuffleExecutorConfig

  @property
  def blends_elementwise(self) -> bool:
    return True

  def select(self, infer_step, score_fn, observation, key):
    actions, _ = _generate(infer_step, observation, key, self.config.num_candidates)
    # Folded rather than split from `key`: generation still sees `key`
    # itself, unmodified -- keeping num_candidates=1 bit-identical to direct
    # like every other reducer here -- while axis selection still draws
    # independent randomness, uncorrelated with which candidates got sampled.
    select_key = jax.random.fold_in(key, 1)
    winner = jax.random.randint(select_key, actions.shape[1:], 0, self.config.num_candidates)
    return jnp.take_along_axis(actions, winner[None], axis=0)[0], {}
