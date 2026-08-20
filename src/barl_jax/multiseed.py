"""Running N seeds of the same {model, env} as one vmapped process.

`train_step` is a pure function of `(graphdef, state, batch, tiers)`, so N
independently-initialized policies whose states are stacked along a leading axis
step together under a single `jax.vmap`. A seed run this way is bit-identical to
the same seed run alone -- vmap batches the arithmetic, it does not change it.

The policies are built normally, one per seed, and only their *states* are
stacked. Building them under a vmapped constructor would trace all the
Python-level construction (config resolution, the `float()` of a buffer that
fixes AMDP's erf_k, DIME's stability asserts), none of which survives a tracer.

What this does **not** do is vary the configuration across the axis. Everything
stacked here shares one graphdef, and barl's tunables are Python scalars living
in that graphdef rather than in state -- `rho` differing per seed would be a
different graphdef, not a different array. Sweeping a hyperparameter is still
scripts/jax/run_sweep.py's job; this axis is seeds.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from barl_jax.rollout import ReplayBuffer, summarize_rollout


def stack(pairs):
  """[(policy, optimizer), ...] -> (graphdef, stacked state).

  Structures are compared rather than graphdefs: two policies of the same
  architecture carry equal-but-not-identical static metadata (an `_Static`
  holder hashes by identity), so their graphdefs compare unequal while being
  interchangeable. What has to match is the state tree."""
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
  """Presents N stacked policies to `ReplayBuffer.rollout()` as if they were
  one, so the rollout loop is the same proven code the single-seed trainer uses.

  Sub-env slot `j` belongs to seed `j // num_envs`, so the flat
  `(num_seeds * num_envs, obs)` batch the vector env hands back reshapes
  straight onto the seed axis and the vmapped inference needs no gather."""

  def __init__(self, graphdef, state, num_seeds: int, num_envs: int, reference):
    self._graphdef = graphdef
    self._state    = state
    self.num_seeds = num_seeds
    self.num_envs  = num_envs
    self.action_discrete = reference.action_discrete
    self._bounds = reference.action_bounds_numpy()

  def action_bounds_numpy(self):
    return self._bounds

  def inference_fn(self):
    return partial(_stacked_infer, self._graphdef, self.num_seeds, self.num_envs,
                   self._state)


@partial(jax.jit, static_argnums=(0, 1, 2))
def _stacked_infer(graphdef, num_seeds, num_envs, state, observation, key):
  """The compiled body behind `StackedPolicy.inference_fn`. Module-level for the
  same reason as `policy._jitted_infer`: built inside the method it was a fresh
  closure per call, so every rollout recompiled the whole vmapped inference."""
  obs = observation.reshape(num_seeds, num_envs, *observation.shape[1:])
  keys = jax.random.split(key, num_seeds)

  def one(state, obs, key):
    policy, _ = nnx.merge(graphdef, state)
    return policy.inference_step(obs, key=key)

  action, extras = jax.vmap(one)(state, obs, keys)
  flat = lambda x: x.reshape(num_seeds * num_envs, *x.shape[2:])
  return flat(action), jax.tree.map(flat, extras)


class MultiSeedReplay(ReplayBuffer):
  """One ReplayBuffer per seed behind a single rollout loop.

  `ReplayBuffer.rollout()` already tags each chunk with the sub-env slot it came
  from (`Trajectory.id`), so routing is a one-line override of
  `save_trajectory()` and the loop itself -- chunking, episode continuation
  across calls, per-slot resets -- is reused unchanged."""

  def __init__(self, num_seeds: int, num_envs: int, max_transitions: int, traj_len: int):
    super().__init__(max_transitions, traj_len)
    self.num_seeds = num_seeds
    self.num_envs  = num_envs
    self.seeds = [ReplayBuffer(max_transitions, traj_len) for _ in range(num_seeds)]

  def save_trajectory(self, traj):
    self.seeds[traj.id // self.num_envs].save_trajectory(traj)

  def last_rollout_per_seed(self) -> list[dict]:
    """The statistics of the most recent rollout(), split per seed: seed i's
    own numbers over the block of sub-env slots its policy drove. The blend
    rollout() returns reads as a per-seed number once each seed logs to its own
    wandb run, but isn't one."""
    acc = self._rollout_per_env
    return [summarize_rollout(acc, slice(i * self.num_envs, (i + 1) * self.num_envs))
            for i in range(self.num_seeds)]

  @property
  def n_transitions(self) -> int:
    return sum(buffer.n_transitions for buffer in self.seeds)

  def min_transitions(self) -> int:
    """The shortest seed's buffer -- what actually bounds sampling."""
    return min(buffer.n_transitions for buffer in self.seeds)

  def transitions_per_seed(self) -> list[int]:
    return [buffer.n_transitions for buffer in self.seeds]

  def n_chunks(self) -> int:
    return sum(len(buffer.buffer) for buffer in self.seeds)

  def sample(self, batch_size: int) -> dict:
    """`(num_seeds, batch_size, ...)` -- each seed draws from its own data, which
    is what keeps the seeds independent rather than merely differently
    initialized."""
    per_seed = [buffer.sample_numpy(batch_size) for buffer in self.seeds]
    # Stacked on the host and handed over in **one** device_put for the whole
    # dict: going through ReplayBuffer.sample() instead would cross the bus
    # num_seeds x num_fields times every gradient step, and even one jnp.asarray
    # per field costs ~80 us of dispatch apiece regardless of the few KB it
    # carries -- five of them were 455 us of host time per grad step.
    return jax.device_put({key: np.stack([s[key] for s in per_seed])
                           for key in per_seed[0]})


def split_metrics(metrics: dict, num_seeds: int) -> tuple[dict, list[dict]]:
  """A vmapped step returns every metric with a leading seed axis. Returns
  (aggregate, per_seed): the mean, for the terminal panel, and one dict per
  seed under the *plain* key names, for that seed's own wandb run.

  Keys arriving without a seed axis (a scalar the step did not vmap) are
  broadcast to every seed, so each run's history is complete on its own."""
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
