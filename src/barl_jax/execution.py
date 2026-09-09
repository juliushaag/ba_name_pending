"""Policy-independent action selection at inference time: Default, Best-of-N
(BoN), Mean-of-N, Median-of-N, Top-K-of-N, Axis-Shuffle-of-N, Perturbation-of-N.
`act()` is a pure transformation over a candidate-generation callable
(`infer_step`, any policy's already-pure `inference_step`) and an optional
scoring callable (`score_fn`) -- it never branches on SAC, AMDP, DIME, SDAC or
NCLQL. The one policy-family-specific piece the score-guided modes need -- how
to normalize observations and flatten an action plan for a critic, and which
critic/reduction to trust -- lives on the policy itself (`Policy.score`,
`Policy.supports_bon`; see `policies/rl.py`'s `RLPolicy.score`), not here.

Bound through `Policy.inference_fn(spec=...)` and `StackedPolicy` (the
`execution_spec` constructor kwarg) so collection and evaluation share this
one operation rather than each policy class reimplementing it."""

from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp

from barl_jax.components import sanitize_action

MODES = ("default", "bon", "mean", "median", "topk", "axis_shuffle", "perturb")

# Modes `act()` needs a real `score_fn` for -- shared with `Policy`/`StackedPolicy`'s
# own `spec.mode == "bon"` check (policy.py, multiseed.py) so a new score-guided
# mode can't be added to `act()` without also being wired up to actually receive
# a scorer at the two call sites that build one.
SCORE_GUIDED_MODES = ("bon", "topk", "perturb")

# ExecutionSpec field defaults that only one mode actually reads -- named so
# the "unused unless this mode" check below and the field declaration can't
# drift apart the way two copies of the same literal would.
DEFAULT_TOP_K = 1
DEFAULT_SIGMA = 0.1


@dataclass(frozen=True)
class ExecutionSpec:
  """Every field is a compile-time choice (static jit args, see
  `Policy.inference_fn`) -- state, observations and keys stay dynamic.

  `top_k` (`mode='topk'` only) and `sigma` (`mode='perturb'` only) are
  rejected at their non-default value under every other mode, same reason
  `num_candidates` is rejected under `'default'`: a value silently unused by
  the selected mode is a config mistake, not a no-op to allow."""

  mode: str = "default"
  num_candidates: int = 1
  top_k: int = DEFAULT_TOP_K
  sigma: float = DEFAULT_SIGMA

  def __post_init__(self):
    if self.mode not in MODES:
      raise ValueError(f"execution mode must be one of {MODES}, got {self.mode!r}")
    if isinstance(self.num_candidates, bool) or not isinstance(self.num_candidates, int) \
       or self.num_candidates < 1:
      raise ValueError(f"num_candidates must be a positive int, got {self.num_candidates!r}")
    if self.mode == "default" and self.num_candidates != 1:
      raise ValueError("mode 'default' takes exactly one candidate (num_candidates=1)")

    if self.mode == "topk":
      if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) \
         or not (1 <= self.top_k <= self.num_candidates):
        raise ValueError(f"top_k must be a positive int <= num_candidates "
                         f"({self.num_candidates}), got {self.top_k!r}")
    elif self.top_k != DEFAULT_TOP_K:
      raise ValueError("top_k only applies to mode='topk'")

    if self.mode == "perturb":
      if isinstance(self.sigma, bool) or not isinstance(self.sigma, (int, float)) \
         or not self.sigma > 0:
        raise ValueError(f"sigma must be a positive number, got {self.sigma!r}")
    elif self.sigma != DEFAULT_SIGMA:
      raise ValueError("sigma only applies to mode='perturb'")


def validate_execution_spec(spec: ExecutionSpec, *, policy) -> list[str]:
  """Capability/action-space checks that need a constructed policy. Syntax
  (an unknown mode, a non-positive N) is already rejected by `ExecutionSpec`
  itself and by `config.run.validate_run` before any policy exists."""
  problems = []
  # 'topk' only actually blends when top_k > 1 -- top_k=1 is a single winner,
  # exactly like 'bon', valid for a discrete action space too.
  blends_elementwise = spec.mode in ("mean", "median", "perturb") \
    or (spec.mode == "topk" and spec.top_k > 1)
  if blends_elementwise and getattr(policy, "action_discrete", False):
    problems.append(f"mode={spec.mode!r} blends continuous box actions elementwise; "
                    f"{type(policy).__name__} has a discrete action space")
  if spec.mode in SCORE_GUIDED_MODES and not getattr(policy, "supports_bon", False):
    problems.append(f"mode={spec.mode!r} needs a critic-backed score() -- "
                    f"{type(policy).__name__} does not implement one")
  return problems


def _candidate_keys(key, n: int):
  """n=1 reuses the caller's key exactly -- bit-identical to Default rather
  than a size-1 split, which would change the sample. n>1 splits into n
  independent keys."""
  return key[None] if n == 1 else jax.random.split(key, n)


def _generate(infer_step: Callable, observation, key, n: int):
  """`(N, batch, chunk, act_dim)` candidate plans and their extras.
  `infer_step` is vmapped over N independent keys with the observation shared
  (broadcast, not batched again), so normalization/BatchNorm running stats
  are read once per call, not perturbed per candidate."""
  keys = _candidate_keys(key, n)
  return jax.vmap(infer_step, in_axes=(None, 0))(observation, keys)


def _gather_topk(x, idx):
  """`x[idx[b, k], b, ...]` for every batch row `b` and slot `k` -- `idx` has
  shape `(batch, k)`, `x` has a leading candidate axis and `batch` next.
  Returns `(batch, k) + x.shape[2:]`."""
  x_bn = jnp.moveaxis(x, 0, 1)                                    # (batch, n, ...)
  idx = idx.reshape(idx.shape + (1,) * (x_bn.ndim - 2))
  idx = jnp.broadcast_to(idx, idx.shape[:2] + x_bn.shape[2:])
  return jnp.take_along_axis(x_bn, idx, axis=1)


def _gather_winner(x, winner):
  """`x[winner[b], b, ...]` for every batch row `b` -- `winner` has shape
  `(batch,)`, `x` has a leading candidate axis and `batch` next. The k=1 case
  of `_gather_topk`, with that trivial slot squeezed back out."""
  return _gather_topk(x, winner[:, None])[:, 0]


def _score_candidates(score_fn: Callable, observation, actions):
  """Scores every `(n, batch)` candidate once as an ordinary `(n*batch)`-row
  forward pass -- see `act`'s BoN branch for why folding beats vmapping.
  Returns `(scores, finite)`, both `(n, batch)`; a caller decides how to treat
  a non-finite score rather than having it silently dropped here."""
  n, batch = actions.shape[0], actions.shape[1]
  tiled_observation = jnp.broadcast_to(observation[None], (n,) + observation.shape)
  tiled_observation = tiled_observation.reshape((n * batch,) + observation.shape[1:])
  flat_actions = actions.reshape((n * batch,) + actions.shape[2:])
  scores = score_fn(tiled_observation, flat_actions).reshape(n, batch)
  return scores, jnp.isfinite(scores)


def act(infer_step: Callable, score_fn: Optional[Callable], observation, key,
       spec: ExecutionSpec):
  """`infer_step`: pure `(observation, key) -> (action_chunk, extras)`.
  `score_fn`: pure `(observation, action_chunk) -> score`, required for
  `mode='bon'`. Returns `(action_chunk, extras)`; `action_chunk` is always
  `(batch, chunk, act_dim)`, whatever `spec.mode` is."""
  if spec.mode == "default":
    return infer_step(observation, key)

  if spec.mode == "axis_shuffle":
    actions, _ = _generate(infer_step, observation, key, spec.num_candidates)
    # Folded rather than split from `key`: generation still sees `key`
    # itself, unmodified -- keeping n=1 bit-identical to default like the
    # other reducers -- while axis selection still draws independent
    # randomness, uncorrelated with which candidates got sampled. Every
    # (batch, chunk, act_dim) entry gets its own independently-chosen
    # candidate index, so the result is not any single candidate's action but
    # a per-axis recombination of all N -- like `mean`, no single candidate's
    # extras describe it.
    select_key = jax.random.fold_in(key, 1)
    winner = jax.random.randint(select_key, actions.shape[1:], 0, spec.num_candidates)
    return jnp.take_along_axis(actions, winner[None], axis=0)[0], {}

  if spec.mode == "perturb":
    # `num_candidates - 1` local Gaussian nudges around the *one* base action
    # -- not `num_candidates` independent fresh draws (BoN's approach). Handy
    # when the policy is close to deterministic (annealed entropy, or late in
    # training), where independent resampling barely moves the action and BoN
    # ends up scoring near-duplicates; a local perturbation search still finds
    # genuinely different neighbors to compare. The base action is always one
    # of the candidates, so the winner can never score worse than not
    # perturbing at all -- and it is the *only* candidate at num_candidates=1,
    # bit-identical to default like every other reducer here.
    base_action, _ = infer_step(observation, key)
    noise_key = jax.random.fold_in(key, 2)
    noise = spec.sigma * jax.random.normal(
      noise_key, (spec.num_candidates - 1,) + base_action.shape, base_action.dtype)
    perturbed = sanitize_action(base_action[None] + noise, xp=jnp)
    candidates = jnp.concatenate([base_action[None], perturbed], axis=0)
    scores, finite = _score_candidates(score_fn, observation, candidates)
    safe_scores = jnp.where(finite, scores, -jnp.inf)
    winner = jnp.argmax(safe_scores, axis=0)
    return _gather_winner(candidates, winner), {
      "selected_score": _gather_winner(scores, winner),
      "num_invalid": jnp.sum(~finite, axis=0),
    }

  actions, extras = _generate(infer_step, observation, key, spec.num_candidates)

  if spec.mode in ("mean", "median"):
    # Only the candidate axis is reduced -- never seeds/envs/time. Extras from
    # N distinct samples do not describe the reduced action (whose induced
    # distribution differs from the original policy's), so none are kept
    # here; a caller wanting extras uses 'default' or one of the score-guided
    # modes. Median over mean where a single wild candidate should not move
    # the result the way it would move an average.
    reduce = jnp.median if spec.mode == "median" else jnp.mean
    return reduce(actions, axis=0), {}

  if spec.mode == "topk":
    # BoN generalized: average the K best-scoring candidates instead of
    # hard-selecting the single best -- softens BoN's failure mode of
    # exploiting a miscalibrated critic's noise on one winner, while staying
    # quality-directed (unlike plain 'mean', which is blind to the score).
    # top_k == num_candidates degenerates to averaging every candidate, same
    # as 'mean'; top_k == 1 degenerates to the single best, same as 'bon'.
    scores, finite = _score_candidates(score_fn, observation, actions)
    safe_scores = jnp.where(finite, scores, -jnp.inf)
    _, top_idx = jax.lax.top_k(safe_scores.T, spec.top_k)                   # (batch, top_k)
    return jnp.mean(_gather_topk(actions, top_idx), axis=1), {}

  # mode == "bon"
  #
  # score_fn is called once over an (N*batch) leading axis rather than
  # vmapped over the candidate axis: score_fn is a bound Policy.score, and an
  # nnx critic ensemble (networks/ensemble.py) already vmaps internally over
  # its members -- an outer jax.vmap around that nesting breaks nnx's trace
  # tracking. Folding N into the batch is an ordinary larger forward pass,
  # not a second nesting level, and is exactly as parallel on-device.
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
