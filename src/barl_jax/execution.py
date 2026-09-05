"""Policy-independent action selection at inference time: Default, Best-of-N
(BoN), Mean-of-N. `act()` is a pure transformation over a candidate-generation
callable (`infer_step`, any policy's already-pure `inference_step`) and an
optional scoring callable (`score_fn`) -- it never branches on SAC, AMDP,
DIME, SDAC or NCLQL. The one policy-family-specific piece BoN needs -- how to
normalize observations and flatten an action plan for a critic, and which
critic/reduction to trust -- lives on the policy itself (`Policy.score`,
`Policy.supports_bon`; see `policies/rl.py`'s `RLPolicy.score`), not here.

Bound through `Policy.inference_fn(spec=...)` and `StackedPolicy` (the
`execution_spec` constructor kwarg) so collection and evaluation share this
one operation rather than each policy class reimplementing it."""

from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp

MODES = ("default", "bon", "mean")


@dataclass(frozen=True)
class ExecutionSpec:
  """`mode` and `num_candidates` are compile-time choices (static jit args,
  see `Policy.inference_fn`) -- state, observations and keys stay dynamic."""

  mode: str = "default"
  num_candidates: int = 1

  def __post_init__(self):
    if self.mode not in MODES:
      raise ValueError(f"execution mode must be one of {MODES}, got {self.mode!r}")
    if isinstance(self.num_candidates, bool) or not isinstance(self.num_candidates, int) \
       or self.num_candidates < 1:
      raise ValueError(f"num_candidates must be a positive int, got {self.num_candidates!r}")
    if self.mode == "default" and self.num_candidates != 1:
      raise ValueError("mode 'default' takes exactly one candidate (num_candidates=1)")


def validate_execution_spec(spec: ExecutionSpec, *, policy) -> list[str]:
  """Capability/action-space checks that need a constructed policy. Syntax
  (an unknown mode, a non-positive N) is already rejected by `ExecutionSpec`
  itself and by `config.run.validate_run` before any policy exists."""
  problems = []
  if spec.mode == "mean" and getattr(policy, "action_discrete", False):
    problems.append(f"mode='mean' averages continuous box actions elementwise; "
                    f"{type(policy).__name__} has a discrete action space")
  if spec.mode == "bon" and not getattr(policy, "supports_bon", False):
    problems.append(f"mode='bon' needs a critic-backed score() -- "
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


def _gather_winner(x, winner):
  """`x[winner[b], b, ...]` for every batch row `b` -- `winner` has shape
  `(batch,)`, `x` has a leading candidate axis and `batch` next."""
  idx = winner.reshape((1, -1) + (1,) * (x.ndim - 2))
  idx = jnp.broadcast_to(idx, (1,) + x.shape[1:])
  return jnp.take_along_axis(x, idx, axis=0)[0]


def act(infer_step: Callable, score_fn: Optional[Callable], observation, key,
       spec: ExecutionSpec):
  """`infer_step`: pure `(observation, key) -> (action_chunk, extras)`.
  `score_fn`: pure `(observation, action_chunk) -> score`, required for
  `mode='bon'`. Returns `(action_chunk, extras)`; `action_chunk` is always
  `(batch, chunk, act_dim)`, whatever `spec.mode` is."""
  if spec.mode == "default":
    return infer_step(observation, key)

  actions, extras = _generate(infer_step, observation, key, spec.num_candidates)

  if spec.mode == "mean":
    # Only the candidate axis is averaged -- never seeds/envs/time. Extras
    # from N distinct samples do not describe the averaged action (whose
    # induced distribution differs from the original policy's), so none are
    # kept here; a caller wanting extras uses 'default' or 'bon'.
    return jnp.mean(actions, axis=0), {}

  # mode == "bon"
  #
  # score_fn is called once over an (N*batch) leading axis rather than
  # vmapped over the candidate axis: score_fn is a bound Policy.score, and an
  # nnx critic ensemble (networks/ensemble.py) already vmaps internally over
  # its members -- an outer jax.vmap around that nesting breaks nnx's trace
  # tracking. Folding N into the batch is an ordinary larger forward pass,
  # not a second nesting level, and is exactly as parallel on-device.
  n, batch = actions.shape[0], actions.shape[1]
  tiled_observation = jnp.broadcast_to(observation[None], (n,) + observation.shape)
  tiled_observation = tiled_observation.reshape((n * batch,) + observation.shape[1:])
  flat_actions = actions.reshape((n * batch,) + actions.shape[2:])
  scores = score_fn(tiled_observation, flat_actions).reshape(n, batch)
  finite = jnp.isfinite(scores)
  safe_scores = jnp.where(finite, scores, -jnp.inf)
  # All candidates invalid for a row: argmax on an all -inf row deterministically
  # picks index 0 -- a defined fallback, not an accident of the encoding.
  winner = jnp.argmax(safe_scores, axis=0)                                  # (batch,)

  selected_action = _gather_winner(actions, winner)
  selected_extras = jax.tree.map(lambda x: _gather_winner(x, winner), extras)
  selected_extras["selected_score"] = _gather_winner(scores, winner)
  selected_extras["num_invalid"] = jnp.sum(~finite, axis=0)
  return selected_action, selected_extras
