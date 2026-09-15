"""Score-guided executors: Best-of-N, Top-K-of-N, Perturbation-of-N. All
three need a real critic-backed `score_fn` (`needs_score = True`) -- see
`ActionExecutor.validate`, which rejects a policy without one."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from barl_jax.components import sanitize_action
from barl_jax.registration import register_executor

from .base import (
  ActionExecutor,
  ExecutorConfig,
  _gather_topk,
  _gather_winner,
  _generate,
  _score_candidates,
  _validate_num_candidates,
)


@dataclass(frozen=True)
class BestOfNExecutorConfig(ExecutorConfig):
  class_name: str = "BestOfNExecutor"
  num_candidates: int = 8

  def __post_init__(self):
    _validate_num_candidates(self.num_candidates)


@register_executor
class BestOfNExecutor(ActionExecutor):
  """`num_candidates` independent samples, the single best-scoring one wins.

  `score_fn` is called once over an (N*batch) leading axis rather than
  vmapped over the candidate axis: `score_fn` is a bound `Policy.score`, and
  an nnx critic ensemble (`networks/ensemble.py`) already vmaps internally
  over its members -- an outer `jax.vmap` around that nesting breaks nnx's
  trace tracking. Folding N into the batch (`_score_candidates`) is an
  ordinary larger forward pass, not a second nesting level, and is exactly
  as parallel on-device."""

  config_type = BestOfNExecutorConfig
  needs_score = True

  def select(self, infer_step, score_fn, observation, key):
    actions, extras = _generate(infer_step, observation, key, self.config.num_candidates)
    scores, finite = _score_candidates(score_fn, observation, actions)
    safe_scores = jnp.where(finite, scores, -jnp.inf)
    # All candidates invalid for a row: argmax on an all -inf row deterministically
    # picks index 0 -- a defined fallback, not an accident of the encoding.
    winner = jnp.argmax(safe_scores, axis=0)                                  # (batch,)

    selected_action = _gather_winner(actions, winner)
    selected_extras = jax.tree.map(lambda x: _gather_winner(x, winner), extras)
    selected_extras["selected_score"] = _gather_winner(scores, winner)
    selected_extras["num_invalid"] = jnp.sum(~finite, axis=0)
    return selected_action, selected_extras


@dataclass(frozen=True)
class TopKExecutorConfig(ExecutorConfig):
  class_name: str = "TopKExecutor"
  num_candidates: int = 8
  top_k: int = 2

  def __post_init__(self):
    _validate_num_candidates(self.num_candidates)
    if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) \
       or not (1 <= self.top_k <= self.num_candidates):
      raise ValueError(f"top_k must be a positive int <= num_candidates "
                       f"({self.num_candidates}), got {self.top_k!r}")


@register_executor
class TopKExecutor(ActionExecutor):
  """Best-of-N generalized: average the K best-scoring candidates instead of
  hard-selecting the single best -- softens BoN's failure mode of exploiting
  a miscalibrated critic's noise on one winner, while staying quality-directed
  (unlike `MeanExecutor`, which is blind to the score). `top_k ==
  num_candidates` degenerates to averaging every candidate (`MeanExecutor`);
  `top_k == 1` degenerates to the single best (`BestOfNExecutor`)."""

  config_type = TopKExecutorConfig
  needs_score = True

  @property
  def blends_elementwise(self) -> bool:
    # top_k == 1 is a single winner, exactly like BestOfN -- valid for a
    # discrete action space too; only averaging >1 candidates blends.
    return self.config.top_k > 1

  def select(self, infer_step, score_fn, observation, key):
    actions, _ = _generate(infer_step, observation, key, self.config.num_candidates)
    scores, finite = _score_candidates(score_fn, observation, actions)
    safe_scores = jnp.where(finite, scores, -jnp.inf)
    _, top_idx = jax.lax.top_k(safe_scores.T, self.config.top_k)              # (batch, top_k)
    return jnp.mean(_gather_topk(actions, top_idx), axis=1), {}


@dataclass(frozen=True)
class PerturbExecutorConfig(ExecutorConfig):
  class_name: str = "PerturbExecutor"
  num_candidates: int = 8
  sigma: float = 0.1

  def __post_init__(self):
    _validate_num_candidates(self.num_candidates)
    if isinstance(self.sigma, bool) or not isinstance(self.sigma, (int, float)) \
       or not self.sigma > 0:
      raise ValueError(f"sigma must be a positive number, got {self.sigma!r}")


@register_executor
class PerturbExecutor(ActionExecutor):
  """`num_candidates - 1` local Gaussian nudges around the *one* base action
  -- not `num_candidates` independent fresh draws (`BestOfNExecutor`'s
  approach). Handy when the policy is close to deterministic (annealed
  entropy, or late in training), where independent resampling barely moves
  the action and BoN ends up scoring near-duplicates; a local perturbation
  search still finds genuinely different neighbors to compare. The base
  action is always one of the candidates, so the winner can never score
  worse than not perturbing at all -- and it is the *only* candidate at
  num_candidates=1, bit-identical to direct like every other executor here."""

  config_type = PerturbExecutorConfig
  needs_score = True

  @property
  def blends_elementwise(self) -> bool:
    return True

  def select(self, infer_step, score_fn, observation, key):
    base_action, _ = infer_step(observation, key)
    noise_key = jax.random.fold_in(key, 2)
    noise = self.config.sigma * jax.random.normal(
      noise_key, (self.config.num_candidates - 1,) + base_action.shape, base_action.dtype)
    perturbed = sanitize_action(base_action[None] + noise, xp=jnp)
    candidates = jnp.concatenate([base_action[None], perturbed], axis=0)
    scores, finite = _score_candidates(score_fn, observation, candidates)
    safe_scores = jnp.where(finite, scores, -jnp.inf)
    winner = jnp.argmax(safe_scores, axis=0)
    return _gather_winner(candidates, winner), {
      "selected_score": _gather_winner(scores, winner),
      "num_invalid": jnp.sum(~finite, axis=0),
    }
