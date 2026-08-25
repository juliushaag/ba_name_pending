import gymnasium as gym
import numpy as np
import pytest

from barl_jax.multiseed import MultiSeedReplay
from barl_jax.rollout import RandomPolicy, ReplayBuffer


def _stacked_fixture(num_seeds: int):
  """`num_seeds` independently-initialized SAC policies, stacked the way the
  trainer stacks them. SAC because it is the cheapest registered policy to
  build; the routing under test is the Policy-agnostic part."""
  from flax import nnx
  from barl_jax.env import Environment
  from barl_jax.multiseed import stack
  from barl_jax.policy import TRAINABLE, Policy

  environment = Environment.make("HalfCheetah")
  env = environment.get_env()
  stats = environment.get_env_stats(env)
  policies = [Policy.make("SAC", env, env_stats=stats, seed=i) for i in range(num_seeds)]
  optimizers = [nnx.Optimizer(p, p.build_optimizer(lr_schedule=None), wrt=TRAINABLE) for p in policies]
  graphdef, state = stack(list(zip(policies, optimizers)))
  return graphdef, state, policies[0], stats["obs_dim"]


def rows(n, start=0, dim=3, act=2):
  """n transitions whose every field encodes its own index, so a sampled row can
  be checked for self-consistency across fields."""
  i = np.arange(start, start + n, dtype=np.float32)
  return dict(
    obs=np.stack([i] * dim, axis=-1),
    next_obs=np.stack([i + 0.5] * dim, axis=-1),
    act=np.stack([-i] * act, axis=-1),
    rew=i.copy(),
    term=np.zeros(n, np.float32),
  )


def test_add_and_size():
  b = ReplayBuffer(100)
  assert b.n_transitions == 0
  b.add(**rows(30))
  assert b.n_transitions == 30
  b.add(**rows(30, start=30))
  assert b.n_transitions == 60


def test_size_saturates_at_capacity():
  b = ReplayBuffer(50)
  for k in range(10):
    b.add(**rows(10, start=10 * k))
  assert b.n_transitions == 50


def test_oldest_first_eviction():
  b = ReplayBuffer(50)
  for k in range(10):
    b.add(**rows(10, start=10 * k))
  # Only the last 50 indices (50..99) survive.
  assert set(np.unique(b._fields["rew"]).astype(int)) == set(range(50, 100))


def test_wraparound_keeps_rows_intact():
  """A batch straddling the ring's end must not tear: every field of a row has
  to land at the same index."""
  b = ReplayBuffer(10)
  b.add(**rows(7))
  b.add(**rows(6, start=7))          # wraps 3 rows past the end
  batch = b.sample_numpy(500)
  assert np.allclose(batch["next_obs"][:, 0], batch["obs"][:, 0] + 0.5)
  assert np.allclose(batch["act"][:, 0], -batch["obs"][:, 0])
  assert np.allclose(batch["rew"], batch["obs"][:, 0])


def test_sample_shapes_and_replacement():
  b = ReplayBuffer(100)
  b.add(**rows(40))
  batch = b.sample_numpy(256)
  assert batch["obs"].shape == (256, 3)
  assert batch["act"].shape == (256, 2)
  assert batch["rew"].shape == (256,)
  assert batch["term"].shape == (256,)


def test_sample_never_returns_unwritten_rows():
  b = ReplayBuffer(1000)
  b.add(**rows(5))
  batch = b.sample_numpy(2000)
  assert set(np.unique(batch["rew"]).astype(int)) <= set(range(5))


def test_sample_empty_raises():
  with pytest.raises(ValueError):
    ReplayBuffer(10).sample_numpy(4)


def test_field_set_must_stay_stable():
  b = ReplayBuffer(10)
  b.add(**rows(2))
  with pytest.raises(KeyError):
    b.add(**rows(2), extra=np.zeros(2, np.float32))


def test_oversized_batch_raises():
  b = ReplayBuffer(4)
  with pytest.raises(ValueError):
    b.add(**rows(5))


def test_multiseed_routes_slots_to_seeds():
  """Slot j belongs to seed j // num_envs -- each seed must see only its own
  block, which is what keeps the seeds independent rather than merely
  differently initialized."""
  ms = MultiSeedReplay(num_seeds=3, num_envs=2, max_transitions=100)
  # 6 slots: seed 0 -> rows 0,1 | seed 1 -> 2,3 | seed 2 -> 4,5
  ms.add(**rows(6))
  for seed in range(3):
    got = set(np.unique(ms.seeds[seed]._fields["rew"][:2]).astype(int))
    assert got == {2 * seed, 2 * seed + 1}
  assert ms.transitions_per_seed() == [2, 2, 2]
  assert ms.n_transitions == 6
  assert ms.min_transitions() == 2


def test_multiseed_sample_has_seed_axis():
  ms = MultiSeedReplay(num_seeds=3, num_envs=2, max_transitions=100)
  ms.add(**rows(6))
  batch = ms.sample(8)
  assert batch["obs"].shape == (3, 8, 3)
  assert batch["rew"].shape == (3, 8)


class _Counting(gym.Env):
  """obs = [t]; terminates at `term_at`, truncates at t == 5 -- so one rollout
  crosses both kinds of boundary and the stored transition can be checked
  against each."""

  observation_space = gym.spaces.Box(-1e9, 1e9, (1,), np.float32)
  action_space      = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)

  def __init__(self, term_at):
    self.t, self.term_at = 0, term_at

  def reset(self, seed=None, options=None):
    self.t = 0
    return np.array([0.0], np.float32), {}

  def step(self, action):
    self.t += 1
    return (np.array([float(self.t)], np.float32), float(self.t),
            self.t == self.term_at, self.t == 5, {})


def _rollout_fields(term_at, n_steps=7):
  env = gym.vector.SyncVectorEnv(
    [lambda: _Counting(term_at), lambda: _Counting(99)],
    autoreset_mode=gym.vector.AutoresetMode.DISABLED)
  buffer = ReplayBuffer(1000)
  buffer.rollout(RandomPolicy(env.single_action_space, seed=0), env,
                 n_steps=n_steps, seed=0)
  n = buffer.n_transitions
  return {k: v[:n] for k, v in buffer._fields.items()}


def test_rollout_next_obs_is_pre_reset():
  """The observation the env returned before its reset is the one stored -- a
  truncated step has to bootstrap from where it actually left off."""
  f = _rollout_fields(term_at=3)
  assert np.allclose(f["next_obs"][:, 0], f["obs"][:, 0] + 1)
  # slot 1 truncates at t == 5: its 5th transition keeps next_obs == 5, not 0.
  assert f["next_obs"][1::2][4, 0] == 5.0


def test_rollout_term_excludes_truncation():
  f = _rollout_fields(term_at=3)
  slot0, slot1 = f["term"][0::2], f["term"][1::2]
  # slot 0 terminates at t == 3, resets, and reaches t == 3 again within 7 steps
  assert list(np.flatnonzero(slot0)) == [2, 5]
  # slot 1 only ever truncates, so it carries no terminal flag at all
  assert slot1.sum() == 0.0


def test_rollout_restarts_episode_after_boundary():
  f = _rollout_fields(term_at=3)
  assert f["obs"][0::2][3, 0] == 0.0     # after termination
  assert f["obs"][1::2][5, 0] == 0.0     # after truncation


def test_eval_slot_blocks_map_to_seeds():
  """`evaluate()` slices one vmapped pass as all_runs[i*eval_runs:(i+1)*eval_runs],
  which is only correct if slot j is driven by seed j // eval_runs. Same
  convention as the rollout, and the thing that silently mislabels every seed's
  eval curve if it ever changes."""
  import jax
  import jax.numpy as jnp
  from barl_jax.multiseed import StackedPolicy, unstack

  num_seeds, per_run = 3, 4
  graphdef, state, reference, obs_dim = _stacked_fixture(num_seeds)

  obs = np.random.RandomState(0).randn(num_seeds * per_run, obs_dim).astype(np.float32)
  key = jax.random.PRNGKey(0)

  stacked = StackedPolicy(graphdef, state, num_seeds, per_run, reference).inference_fn()
  actions = np.asarray(stacked(jnp.asarray(obs), key)[0])

  for i, seed_key in enumerate(jax.random.split(key, num_seeds)):
    policy, _ = unstack(graphdef, state, i)
    block = jnp.asarray(obs[i * per_run:(i + 1) * per_run])
    own = np.asarray(policy.inference_fn()(block, seed_key)[0])
    assert np.allclose(actions[i * per_run:(i + 1) * per_run], own, atol=1e-5)

  # vacuous unless the seeds really are different policies
  assert not np.allclose(actions[:per_run], actions[per_run:2 * per_run], atol=1e-5)
