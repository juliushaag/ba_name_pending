"""Evaluation's own RNG ownership: `derive_seed` is deterministic and
distinguishes its parts, `run_episodes` derives both its policy-action key
and every wave's `env.reset(seed=...)` from the caller's `seed` (never an
unseeded env.reset(), never the hardcoded key 0 it used to pass), and
distinct waves within one call get distinct reset seeds."""

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from barl_jax.evaluation import run_episodes
from barl_jax.seeding import derive_seed

ACT_DIM, OBS_DIM = 2, 3


def test_derive_seed_is_deterministic():
  assert derive_seed(7, "eval", 3) == derive_seed(7, "eval", 3)


@pytest.mark.parametrize("other", [(8, "eval", 3), (7, "train", 3), (7, "eval", 4)])
def test_derive_seed_distinguishes_each_part(other):
  assert derive_seed(7, "eval", 3) != derive_seed(*other)


class SpyVectorEnv(gym.vector.VectorEnv):
  """Records every `seed=` reset() receives; two "episodes" (EPISODE_LEN
  steps) per wave so `run_episodes` needs more than one wave for n > num_envs."""

  EPISODE_LEN = 2

  def __init__(self, num_envs):
    self.num_envs = num_envs
    self.single_observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
    self.single_action_space = gym.spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)
    self.reset_seeds = []
    self._t = np.zeros(num_envs, dtype=int)

  def reset(self, seed=None, options=None):
    self.reset_seeds.append(seed)
    mask = np.ones(self.num_envs, bool) if options is None else np.asarray(options["reset_mask"])
    self._t[mask] = 0
    return np.zeros((self.num_envs, OBS_DIM), np.float32), {}

  def step(self, action):
    self._t += 1
    term = self._t >= self.EPISODE_LEN
    return (np.zeros((self.num_envs, OBS_DIM), np.float32), np.ones(self.num_envs, np.float32),
            term, np.zeros(self.num_envs, bool), {})

  def call(self, name, *a, **kw):
    return [np.zeros((2, 2, 3), np.uint8) for _ in range(self.num_envs)]


class Actor:
  action_dim = ACT_DIM
  chunk_size = 1

  def inference_fn(self):
    def infer(obs, key):
      return jnp.zeros((obs.shape[0], 1, ACT_DIM)), {}
    return infer


def test_reset_is_always_seeded_never_left_to_ambient_entropy():
  env = SpyVectorEnv(2)
  run_episodes(Actor(), env, n=2, render=False, seed=123)
  assert env.reset_seeds and all(s is not None for s in env.reset_seeds)


def test_two_waves_in_one_call_get_different_reset_seeds():
  env = SpyVectorEnv(2)
  run_episodes(Actor(), env, n=4, render=False, seed=123)  # 2 waves of 2
  assert len(env.reset_seeds) == 2
  assert env.reset_seeds[0] != env.reset_seeds[1]


def test_same_seed_reproduces_the_same_reset_sequence():
  seeds_a = SpyVectorEnv(2)
  run_episodes(Actor(), seeds_a, n=4, render=False, seed=123)
  seeds_b = SpyVectorEnv(2)
  run_episodes(Actor(), seeds_b, n=4, render=False, seed=123)
  assert seeds_a.reset_seeds == seeds_b.reset_seeds


def test_different_seeds_reproduce_different_reset_sequences():
  seeds_a = SpyVectorEnv(2)
  run_episodes(Actor(), seeds_a, n=2, render=False, seed=123)
  seeds_b = SpyVectorEnv(2)
  run_episodes(Actor(), seeds_b, n=2, render=False, seed=456)
  assert seeds_a.reset_seeds != seeds_b.reset_seeds


def test_default_seed_is_not_the_old_hardcoded_zero_key_only():
  """Regression guard: run_episodes used to hardcode `jax.random.PRNGKey(0)`
  for policy actions and pass env.reset() no seed at all. `seed=` must now
  actually reach the env even when the caller passes none explicitly (the
  default still derives a real reset seed, not None)."""
  env = SpyVectorEnv(2)
  run_episodes(Actor(), env, n=2, render=False)
  assert env.reset_seeds[0] is not None
  assert env.reset_seeds[0] == derive_seed(0, "wave", 0)


def test_policy_action_key_is_derived_from_seed_not_hardcoded():
  seen_keys = []

  class KeySpyActor(Actor):
    def inference_fn(self):
      def infer(obs, key):
        seen_keys.append(np.asarray(key))
        return jnp.zeros((obs.shape[0], 1, ACT_DIM)), {}
      return infer

  run_episodes(KeySpyActor(), SpyVectorEnv(2), n=1, render=False, seed=999)
  _, expected_subkey = jax.random.split(jax.random.PRNGKey(999))
  np.testing.assert_array_equal(seen_keys[0], np.asarray(expected_subkey))
