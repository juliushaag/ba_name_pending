"""barl_jax.exact_resume: the opt-in capture/restore mechanism, verified
against a concrete adapter for Pendulum-v1 (examples.cpu_quickstart's
TinyPendulum) -- the actual acceptance bar the plan sets: uninterrupted
stepping versus save/interrupt/restore/continue must produce identical
observations, rewards, and termination/truncation for the same action
sequence, not merely "the mechanism runs"."""

import importlib.util
import pathlib

import numpy as np

from barl_jax.exact_resume import ExactResumeMixin


def _load_tiny_pendulum():
  path = pathlib.Path(__file__).resolve().parent.parent / "examples" / "cpu_quickstart.py"
  spec = importlib.util.spec_from_file_location("cpu_quickstart", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module.TinyPendulum


TinyPendulum = _load_tiny_pendulum()


class TinyPendulumExactResume(ExactResumeMixin, TinyPendulum):
  """Pendulum-v1's step() is a deterministic function of (state, action) --
  no RNG involved (see gymnasium.envs.classic_control.pendulum.PendulumEnv.step);
  `np_random` is only ever read in reset(). So the whole state is: `state`
  (theta, thetadot) and `last_u` (render-only, included for completeness) on
  the unwrapped env, and the outer `TimeLimit` wrapper's own elapsed-step
  counter (gym.make() wraps every classic_control env in one) -- captured
  separately since it lives above `env.unwrapped`, not on it."""

  UNWRAPPED_ATTRS = ("state", "last_u")
  OUTER_ATTRS = ("_elapsed_steps",)


def _rollout(env, actions):
  obs_log, reward_log, done_log = [], [], []
  for a in actions:
    obs, reward, term, trunc, _ = env.step(np.asarray([a], np.float32))
    obs_log.append(obs)
    reward_log.append(reward)
    done_log.append((term, trunc))
  return obs_log, reward_log, done_log


def test_uninterrupted_matches_save_restore_continuation():
  environment = TinyPendulumExactResume()

  # Reference: one uninterrupted run.
  reference_env = environment.get_env()
  reference_env.reset(seed=7)
  warmup = [0.5, -0.3, 1.0, -1.5, 0.2]
  continuation = [0.1, -0.9, 1.3, 0.0, -0.4, 0.8]
  _rollout(reference_env, warmup)
  ref_obs, ref_reward, ref_done = _rollout(reference_env, continuation)
  reference_env.close()

  # Save/restore: same warmup on a fresh env, capture, build a second fresh
  # env, restore into it, then continue on the RESTORED one -- not the
  # original -- with the identical continuation actions.
  live_env = environment.get_env()
  live_env.reset(seed=7)
  _rollout(live_env, warmup)
  captured = environment.capture_env_state(live_env)
  live_env.close()

  restored_env = environment.get_env()
  restored_env.reset(seed=0)  # deliberately different seed: state must come from restore, not reset
  environment.restore_env_state(restored_env, captured)
  restored_obs, restored_reward, restored_done = _rollout(restored_env, continuation)
  restored_env.close()

  for a, b in zip(ref_obs, restored_obs):
    np.testing.assert_array_equal(a, b)
  assert ref_reward == restored_reward
  assert ref_done == restored_done


def test_restore_does_not_read_a_stale_capture_across_further_steps():
  """A capture taken mid-episode, then the source env stepped further,
  must not have silently mutated -- restore_env_state must reproduce the
  state as of the *capture* point, not whatever the source env holds now."""
  environment = TinyPendulumExactResume()
  env = environment.get_env()
  env.reset(seed=3)
  _rollout(env, [0.5, -0.2])
  captured = environment.capture_env_state(env)
  _rollout(env, [1.0, -1.0, 0.3])  # source env keeps moving after the capture

  restored_env = environment.get_env()
  restored_env.reset(seed=0)
  environment.restore_env_state(restored_env, captured)

  # capture() again immediately after restore: must match the pre-further-steps capture.
  recaptured = environment.capture_env_state(restored_env)
  np.testing.assert_array_equal(recaptured["unwrapped"]["state"], captured["unwrapped"]["state"])
  assert recaptured["outer"]["_elapsed_steps"] == captured["outer"]["_elapsed_steps"]

  env.close()
  restored_env.close()


def test_supports_exact_resume_is_false_by_default_for_ordinary_environments():
  assert TinyPendulum.supports_exact_resume is False
  assert TinyPendulumExactResume.supports_exact_resume is True
