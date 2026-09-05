"""Stacked state and batched inference across independent seeds."""
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

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


class StackedPolicy:
  """Presents N stacked policies to `run_episodes()` as if they were one."""

  def __init__(self, graphdef, state, num_seeds: int, num_envs: int, reference):
    self._graphdef = graphdef
    self._state    = state
    self.num_seeds = num_seeds
    self.num_envs  = num_envs
    self.discount = reference.discount
    self.chunk_size = reference.chunk_size
    self.action_dim = reference.action_dim
    self.action_discrete = reference.action_discrete
    self._bounds = reference.action_bounds_numpy()

  def action_bounds_numpy(self):
    return self._bounds

  def inference_fn(self):
    return partial(_stacked_infer, _eval_graphdef(self._graphdef, self._state),
                   self.num_seeds, self.num_envs, self._state)


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


@partial(jax.jit, static_argnums=(0, 1, 2))
def _stacked_infer(graphdef, num_seeds, num_envs, state, observation, key):
  obs = observation.reshape(num_seeds, num_envs, *observation.shape[1:])
  keys = jax.random.split(key, num_seeds)

  def one(state, obs, key):
    policy, _ = nnx.merge(graphdef, state)
    return policy.inference_step(obs, key=key)

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


