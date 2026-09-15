from dataclasses import dataclass

from barl_jax.registration import register_executor

from .base import ActionExecutor, ExecutorConfig


@dataclass(frozen=True)
class DirectExecutorConfig(ExecutorConfig):
  class_name: str = "DirectExecutor"


@register_executor
class DirectExecutor(ActionExecutor):
  """The policy's own action, unmodified -- one candidate, no scorer, no
  key-splitting. The default everywhere (`RunConfig.collection_executor`/
  `eval_executor`); `is_direct` lets callers skip `select()` and stay on
  the cheap no-execution-strategy jit path entirely (see
  `Policy.inference_fn`), so this method exists for completeness and for
  callers that construct a `DirectExecutor` explicitly (e.g.
  `scripts/evaluate.py`'s uniform per-executor loop)."""

  config_type = DirectExecutorConfig
  is_direct = True

  def select(self, infer_step, score_fn, observation, key):
    return infer_step(observation, key)
