"""Stacked state and batched inference across independent seeds."""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import barl_jax.executors as executors
from barl_jax.components import interned_static


def stack(pairs):
  """[(policy, optimizer), ...] -> (graphdef, stacked state)."""
  graphdefs, states = zip(*(nnx.split(pair) for pair in pairs))

  reference = jax.tree.structure(states[0])
  for i, state in enumerate(states[1:], start=1):
    if jax.tree.structure(state) != reference:
      raise ValueError(
        f"seed {i} has a different state structure than seed 0 -- every seed on a "
        f"vmapped axis must share one architecture.")
  shapes = [tuple(leaf.shape) for leaf in jax.tree.leaves(states[0])]
  for i, state in enumerate(states[1:], start=1):
    if [tuple(leaf.shape) for leaf in jax.tree.leaves(state)] != shapes:
      raise ValueError(f"seed {i}'s parameter shapes differ from seed 0's.")

  return graphdefs[0], jax.tree.map(lambda *xs: jnp.stack(xs), *states)


def unstack(graphdef, state, index: int):
  """One seed's (policy, optimizer), for evaluation and checkpointing."""
  return nnx.merge(graphdef, jax.tree.map(lambda x: x[index], state))


class SeedActor:
  """Presents one seed's `Policy` to `run_episodes()` with an action executor
  bound at construction -- the single-env, non-stacked analog of
  `StackedPolicy`, for evaluating (or rendering) one seed on its own instead
  of every seed in one vmapped pass."""

  def __init__(self, policy, executor: "executors.ActionExecutor" = None):
    self.policy, self.executor = policy, executor
    self.chunk_size, self.action_dim = policy.chunk_size, policy.action_dim

  def inference_fn(self):
    return self.policy.inference_fn(executor=self.executor)


class StackedPolicy:
  """Presents N stacked policies to `run_episodes()` as if they were one."""

  def __init__(self, graphdef, state, num_seeds: int, num_envs: int, reference,
              *, executor: "executors.ActionExecutor" = None):
    self._graphdef = graphdef
    self._state    = state
    self.num_seeds = num_seeds
    self.num_envs  = num_envs
    self.discount = reference.discount
    self.chunk_size = reference.chunk_size
    self.action_dim = reference.action_dim
    self.action_discrete = reference.action_discrete
    self._bounds = reference.action_bounds_numpy()
    self._executor = executor

  def action_bounds_numpy(self):
    return self._bounds

  def inference_fn(self):
    is_direct = self._executor is None or self._executor.is_direct
    executor_config = None if is_direct else self._executor.config
    return partial(_stacked_infer, interned_static(_eval_graphdef(self._graphdef, self._state)),
                   self.num_seeds, self.num_envs, executor_config, self._state)


_EVAL_GRAPHDEF: dict = {}


def _eval_graphdef(graphdef, state):
  cached = _EVAL_GRAPHDEF.get(graphdef)
  if cached is None:
    policy, opt = nnx.merge(graphdef, jax.tree.map(lambda x: x[0], state))
    clone = nnx.clone(policy)
    clone.eval()
    cached, _ = nnx.split((clone, opt))
    _EVAL_GRAPHDEF[graphdef] = cached
  return cached


@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def _stacked_infer(graphdef, num_seeds, num_envs, executor_config, state, observation, key):
  obs = observation.reshape(num_seeds, num_envs, *observation.shape[1:])
  keys = jax.random.split(key, num_seeds)

  def one(state, obs, key):
    policy, _ = nnx.merge(graphdef.value, state)
    if executor_config is None:
      return policy.inference_step(obs, key=key)
    executor = executors.executor_from_config(executor_config)
    infer_step = lambda o, k: policy.inference_step(o, key=k)
    score_fn = policy.score if executor.needs_score else None
    return executor.select(infer_step, score_fn, obs, key)

  action, extras = jax.vmap(one)(state, obs, keys)
  flat = lambda x: x.reshape(num_seeds * num_envs, *x.shape[2:])
  return flat(action), jax.tree.map(flat, extras)


def split_metrics(metrics: dict, num_seeds: int) -> tuple[dict, list[dict]]:
  """A vmapped step returns every metric with a leading seed axis."""
  aggregate, per_seed = {}, [{} for _ in range(num_seeds)]
  for key, value in metrics.items():
    value = np.asarray(value)
    if value.ndim == 0 or value.shape[0] != num_seeds:
      aggregate[key] = value.reshape(()) if value.ndim == 0 else value
      for seed in per_seed:
        seed[key] = aggregate[key]
      continue
    aggregate[key] = value.mean()
    for i in range(num_seeds):
      per_seed[i][key] = value[i]
  return aggregate, per_seed


