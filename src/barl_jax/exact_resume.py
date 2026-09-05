"""Opt-in exact environment-state capture/restore, for environments with
demonstrably complete state capture only. `Environment.supports_exact_resume`
is `False` for every environment by default (env.py); ordinary resume
(episodes restart, pending action chunks discarded -- see
training_checkpoint.py) is every environment's behavior unless it declares
otherwise here, explicitly, per environment family. A generic wrapper that
copies only the base env's RNG is NOT sufficient -- gymnasium's own
`TimeLimit` wrapper carries its own elapsed-step counter outside the
unwrapped env entirely, and simulator internals (mujoco qpos/qvel and
beyond) are a different capture surface again.

This module is the mechanism, not a claim that every environment supports
it: only `examples`/`tests` declare a concrete adapter today (Pendulum-v1's
classic_control physics), verified by a real save/restore continuation test
(tests/test_exact_resume.py), not merely "it runs". Not wired into
trainer.py's resume flow (which still restarts episodes for every
environment) -- see that gap called out in the README."""


def _copy(value):
  import numpy as np
  return np.array(value, copy=True) if isinstance(value, np.ndarray) else value


class ExactResumeMixin:
  """Mix into an `Environment` subclass whose wrapped gym env's full state
  is exactly: its own `np_random` bit-generator state, a fixed list of
  attributes on `env.unwrapped` (`UNWRAPPED_ATTRS`), and a fixed list of
  attributes on the env object `get_env()` itself returns
  (`OUTER_ATTRS` -- gymnasium wrappers like `TimeLimit` keep state at this
  level, not on `env.unwrapped`). Declaring this mixin is an explicit claim
  that those three things really are the whole state for stepping and
  resetting identically going forward; an incomplete attribute list is a
  silent correctness bug, not a missing feature -- keep both lists
  conservative and prove completeness with a continuation test before
  trusting this for anything beyond debugging/comparison."""

  supports_exact_resume = True
  OUTER_ATTRS: tuple = ()
  UNWRAPPED_ATTRS: tuple = ()

  def capture_env_state(self, env) -> dict:
    base = env.unwrapped
    return {
      "rng": base.np_random.bit_generator.state,
      "outer": {name: _copy(getattr(env, name)) for name in self.OUTER_ATTRS},
      "unwrapped": {name: _copy(getattr(base, name)) for name in self.UNWRAPPED_ATTRS},
    }

  def restore_env_state(self, env, state: dict) -> None:
    base = env.unwrapped
    base.np_random.bit_generator.state = state["rng"]
    for name, value in state["outer"].items():
      setattr(env, name, _copy(value))
    for name, value in state["unwrapped"].items():
      setattr(base, name, _copy(value))
