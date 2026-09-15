"""`ActionExecutor` -- the abstract base every action-selection strategy
implements, plus the shared candidate-generation/reduction helpers every
multi-candidate executor (`executors.scored`, `executors.blend`) builds on.

Deliberately not a `Component` (`barl_jax.components`): an executor has no
`env_stats`, no trainable/nnx state, no metrics sink -- it is a pure
transform over a policy's already-pure `inference_step`/`score` callables,
sitting between candidate generation and the env, identical in shape whether
called during training collection or evaluation (see `Policy.inference_fn`,
`multiseed.StackedPolicy`). `config_type` reuses `Component`'s attribute name
for consistency, not its contract."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class ExecutorConfig:
  """Every concrete executor's config redeclares `class_name` with its own
  default (`StochasticActorConfig`/`DDPMActorConfig`'s pattern, not
  `ScheduleConfig`'s one-shape-fits-all) -- this base exists only so
  `RunConfig.collection_executor`/`eval_executor` have one type to be typed
  as, not to carry shared tunables no executor is required to have."""

  class_name: str


class ActionExecutor(ABC):
  """`select()` is the operation: `(observation, key) -> (action_chunk,
  extras)` via a candidate-generating `infer_step` and, for executors that
  need one, a critic-backed `score_fn` (see `needs_score`). Every field of
  `config` is a compile-time choice -- state, observations and keys stay
  dynamic (see `executor_from_config`'s jit-boundary use)."""

  config_type: type = None

  #: True for BestOfN/TopK/Perturb -- `score_fn` must be real, not None, and
  #: `validate()` rejects a policy without `Policy.supports_bon`.
  needs_score: bool = False

  #: True only on `DirectExecutor` -- callers skip `select()` entirely on
  #: this flag to stay on the cheap, no-execution-strategy jit path (see
  #: `Policy.inference_fn`), not because `select()` would compute anything
  #: different.
  is_direct: bool = False

  def __init__(self, config):
    self.config = config

  @abstractmethod
  def select(self, infer_step: Callable, score_fn: Optional[Callable],
            observation, key):
    """`infer_step`: pure `(observation, key) -> (action_chunk, extras)`.
    `score_fn`: pure `(observation, action_chunk) -> score`, non-None iff
    `needs_score`. Returns `(action_chunk, extras)`; `action_chunk` is always
    `(batch, chunk, act_dim)`."""
    ...

  @property
  def blends_elementwise(self) -> bool:
    """True for a reducer whose result is not any single candidate's action
    (Mean/Median/Perturb always; TopK only once `top_k > 1` averages more
    than the winner) -- such a result is meaningless for a discrete action
    space, see `validate()`."""
    return False

  def validate(self, *, policy) -> list[str]:
    """Capability/action-space checks that need a constructed policy. Syntax
    (an unknown class name, a non-positive N) is already rejected by the
    config dataclass itself and by `config.run.validate_run` before any
    policy exists."""
    problems = []
    if self.blends_elementwise and getattr(policy, "action_discrete", False):
      problems.append(f"{type(self).__name__} blends continuous box actions elementwise; "
                      f"{type(policy).__name__} has a discrete action space")
    if self.needs_score and not getattr(policy, "supports_bon", False):
      problems.append(f"{type(self).__name__} needs a critic-backed score() -- "
                      f"{type(policy).__name__} does not implement one")
    return problems


def _validate_num_candidates(num_candidates) -> None:
  """Shared by every multi-candidate executor config's own `__post_init__`
  (`DirectExecutorConfig` has no `num_candidates` field at all, so it never
  calls this) -- one implementation, not a shared dataclass field bag."""
  if isinstance(num_candidates, bool) or not isinstance(num_candidates, int) or num_candidates < 1:
    raise ValueError(f"num_candidates must be a positive int, got {num_candidates!r}")


def executor_from_config(config) -> ActionExecutor:
  """The `class_name`-tagged config -> a fresh, stateless executor instance.
  Cheap to call every jit trace (not every call -- see `_jitted_infer_exec`):
  `config` alone crosses the jit boundary as the static arg, not the
  executor object, so no executor ever needs `__eq__`/`__hash__`."""
  from barl_jax.registration import get_executor
  return get_executor(config.class_name)(config)


# ---------------------------------------------------------------------------
# Shared candidate generation/reduction helpers -- used by both
# `executors.blend` (no scorer) and `executors.scored` (scorer-guided).
# ---------------------------------------------------------------------------

def _candidate_keys(key, n: int):
  """n=1 reuses the caller's key exactly -- bit-identical to Direct rather
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
  forward pass -- see the BoN/TopK executors for why folding beats vmapping.
  Returns `(scores, finite)`, both `(n, batch)`; a caller decides how to treat
  a non-finite score rather than having it silently dropped here."""
  n, batch = actions.shape[0], actions.shape[1]
  tiled_observation = jnp.broadcast_to(observation[None], (n,) + observation.shape)
  tiled_observation = tiled_observation.reshape((n * batch,) + observation.shape[1:])
  flat_actions = actions.reshape((n * batch,) + actions.shape[2:])
  scores = score_fn(tiled_observation, flat_actions).reshape(n, batch)
  return scores, jnp.isfinite(scores)
