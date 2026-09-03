"""What `run_episodes` is allowed to hand `env.step()`.

The rollout has always sanitized (rollout.py); the eval path did not, and
`Policy.scale_action` -- the method whose docstring promises the guard -- had no
call sites at all. A NaN action therefore reached `mj_step`, which only warns and
poisons qpos: seven evals returned quietly wrong numbers and the eighth, the
first one to render, aborted the process inside mjr_render with no python
traceback (sweep 2026-08-27_13-23-15, AMDP on both DMC envs and SDAC on DogRun).

So the contract is tested from the env's side: whatever the actor emits, what
arrives at step() is finite and inside the env's own bounds.
"""

import gymnasium as gym
import jax
import numpy as np
import pytest

from barl_jax.policy import run_episodes

ACT_DIM, OBS_DIM, EPISODE_LEN = 3, 4, 5
ACT_SCALE, ACT_BIAS = 2.0, 0.5


class RecordingVectorEnv(gym.vector.VectorEnv):
    """Records every action it is stepped with. Terminates each slot after
    `EPISODE_LEN` steps so a wave finishes; autoreset is DISABLED, as
    `run_episodes` requires of a real one."""

    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.single_observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.single_action_space = gym.spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)
        self.seen = []
        self._t = np.zeros(num_envs, dtype=int)

    def reset(self, seed=None, options=None):
        mask = (np.ones(self.num_envs, bool) if options is None
                else np.asarray(options["reset_mask"]))
        self._t[mask] = 0
        return np.zeros((self.num_envs, OBS_DIM), np.float32), {}

    def step(self, action):
        self.seen.append(np.array(action, copy=True))
        self._t += 1
        term = self._t >= EPISODE_LEN
        return (np.zeros((self.num_envs, OBS_DIM), np.float32),
                np.ones(self.num_envs, np.float32), term,
                np.zeros(self.num_envs, bool), {})

    def call(self, name, *args, **kwargs):
        return [np.zeros((2, 2, 3), np.uint8) for _ in range(self.num_envs)]


class Actor:
    """Emits whatever `actions` says, in the shape the eval loop expects."""

    def __init__(self, actions):
        self._actions = np.asarray(actions, np.float32)

    def action_bounds_numpy(self):
        return np.float32(ACT_SCALE), np.float32(ACT_BIAS)

    def inference_fn(self):
        def infer(obs, key):
            return jax.numpy.asarray(
                np.broadcast_to(self._actions, (obs.shape[0], ACT_DIM))), {}
        return infer


POISON = [
    pytest.param([np.nan, np.nan, np.nan], id="all-nan"),
    pytest.param([np.nan, 0.5, -0.5], id="one-nan"),
    pytest.param([np.inf, -np.inf, 0.0], id="infinities"),
    pytest.param([50.0, -50.0, 1.5], id="out-of-range"),
]


@pytest.mark.parametrize("emitted", POISON)
def test_what_reaches_the_env_is_finite_and_in_bounds(emitted):
    env = RecordingVectorEnv(2)
    run_episodes(Actor(emitted), env, n=2, render=False)

    seen = np.concatenate(env.seen)
    assert seen.size > 0
    assert np.isfinite(seen).all()
    assert (np.abs(seen - ACT_BIAS) <= ACT_SCALE + 1e-6).all()


def test_nan_becomes_the_neutral_action_not_a_bound():
    """`sanitize_action` maps NaN to 0 and infinities to the bound they were
    heading for -- a bound is a decision, 0 is the absence of one."""
    env = RecordingVectorEnv(1)
    run_episodes(Actor([np.nan, np.inf, -np.inf]), env, n=1, render=False)

    seen = np.concatenate(env.seen)
    assert np.allclose(seen[:, 0], ACT_BIAS)
    assert np.allclose(seen[:, 1], ACT_SCALE + ACT_BIAS)
    assert np.allclose(seen[:, 2], -ACT_SCALE + ACT_BIAS)


def test_the_clamp_is_counted_not_hidden():
    """Clamping keeps the run alive, so the count is the only thing left that
    says this eval's numbers came from a diverged actor. One component of three
    over every step of the episode."""
    env = RecordingVectorEnv(2)
    records = run_episodes(Actor([np.nan, 0.5, -0.5]), env, n=2, render=False)

    assert len(records) == 2
    for r in records:
        assert r["n_act"] == EPISODE_LEN * ACT_DIM
        assert r["act_nonfinite"] == EPISODE_LEN


def test_a_finite_action_is_counted_as_clean():
    env = RecordingVectorEnv(2)
    records = run_episodes(Actor([0.5, -0.5, 0.0]), env, n=2, render=False)

    assert all(r["act_nonfinite"] == 0 for r in records)
    assert all(r["n_act"] == EPISODE_LEN * ACT_DIM for r in records)
