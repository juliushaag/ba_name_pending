import multiprocessing as mp
from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np


class EnvStats(dict):

  @classmethod
  def from_env(cls, env: gym.Env, **declared) -> "EnvStats":
    obs_space, act_space = env.observation_space, env.action_space
    discrete = isinstance(act_space, gym.spaces.Discrete)

    stats = cls(
      obs_shape    = tuple(obs_space.shape),
      obs_dim      = int(np.prod(obs_space.shape)),
      obs_low      = np.asarray(obs_space.low,  dtype=np.float32),
      obs_high     = np.asarray(obs_space.high, dtype=np.float32),
      act_discrete = discrete,
      act_dim      = int(act_space.n) if discrete else int(np.prod(act_space.shape)),
      act_low      = None if discrete else np.asarray(act_space.low,  dtype=np.float32),
      act_high     = None if discrete else np.asarray(act_space.high, dtype=np.float32),
      max_episode_steps = None,
      return_min   = None,
      return_max   = None,
    )
    stats.update(declared)
    return stats

  def derive(self, **overrides) -> "EnvStats":
    """Shallow copy with some keys replaced, for a component that feeds its
    inner network something other than the raw env dims."""
    return EnvStats({**self, **overrides})

  def static(self) -> dict:
    """Everything in this stats dict -- the run-config view, safe to log once."""
    return dict(self)


class Environment(ABC):

  _classes = dict()

  # SyncVectorEnv runs N sub-envs sequentially in one process, so mujoco's
  # GIL-releasing C step never actually overlaps -- override to True for
  # AsyncVectorEnv (one subprocess per sub-env) once per-step physics cost
  # dominates IPC overhead.
  use_async_vector_env = False

  # True only for a family that mixes in ExactResumeMixin (below) and
  # has verified its state-capture attribute lists are actually complete
  # (see that module). Every environment defaults to ordinary resume
  # semantics: episodes restart, pending action chunks are discarded.
  supports_exact_resume = False

  # Env facts no gym space can report -- `return_min`/`return_max` bound the
  # achievable episode return, which is what a categorical/histogram critic's
  # fixed support needs -- and per-env run-key overrides. Both are class
  # attributes merged down the MRO (a family base's applies to every env under
  # it, one env's own overrides a single key), declared in code because they
  # are properties of the benchmark, not tunables.
  stats: dict = {}
  defaults: dict = {}

  @classmethod
  def merged_stats(cls) -> dict:
    out = {}
    for klass in reversed(cls.__mro__):
      out.update(getattr(klass, "stats", {}))
    return out

  @classmethod
  def merged_defaults(cls) -> dict:
    out = {}
    for klass in reversed(cls.__mro__):
      out.update(getattr(klass, "defaults", {}))
    return out

  @abstractmethod
  def get_env(self) -> gym.Env:
    ...

  @abstractmethod
  def get_dataset(self):
    """Only implemented by offline-dataset-backed environments."""
    ...

  def get_vector_env(self, num_envs: int, asynchronous: bool = None) -> gym.vector.VectorEnv:
    """autoreset_mode=DISABLED is mandatory -- Collector.collect() resets
    finished sub-envs itself via reset_mask. `asynchronous` overrides
    `use_async_vector_env` for a caller the class-level choice doesn't suit --
    a rendering env has to be synchronous, since a forked worker's render()
    comes back as an EOFError (MuJoCo's EGL context does not survive the
    fork) and its frames would go through the pipe at ~900 KB a step."""
    use_async = self.use_async_vector_env if asynchronous is None else asynchronous
    if not use_async:
      return gym.vector.SyncVectorEnv([self.get_env for _ in range(num_envs)],
                                      autoreset_mode=gym.vector.AutoresetMode.DISABLED)
    # Not the default `fork`: JAX is multithreaded by the time this is called,
    # and forking a multithreaded parent is what python warns will "likely lead
    # to a deadlock" -- a worker can inherit a lock held by a thread that does
    # not exist on its side. forkserver's workers fork from a clean
    # single-threaded server process instead.
    methods = mp.get_all_start_methods()
    context = "forkserver" if "forkserver" in methods else "spawn"
    return gym.vector.AsyncVectorEnv([self.get_env for _ in range(num_envs)],
                                     autoreset_mode=gym.vector.AutoresetMode.DISABLED,
                                     context=context)

  @staticmethod
  def make(name : str) -> "Environment":
    return Environment.class_for(name)()


  @staticmethod
  def class_for(name: str):
    from barl_jax.registration import available_names, import_type, load_builtins, resolve_named
    load_builtins("environments")
    if ":" in name:
      cls = import_type(name)
    else:
      cls = resolve_named("environments", name, Environment._classes, Environment)
      if cls is None:
        raise KeyError(f"{name!r} is not a registered Environment or plugin "
                       f"({', '.join(available_names('environments', Environment._classes))})")
    if not issubclass(cls, Environment):
      raise TypeError(f"{name!r} must implement Environment")
    return cls

  @staticmethod
  def available() -> list[str]:
    from barl_jax.registration import available_names, load_builtins
    load_builtins("environments")
    return available_names("environments", Environment._classes)


  def get_env_stats(self, env: gym.Env) -> EnvStats:
    """The env-derived facts every component is built with -- the spaces, plus
    this class's `stats` declared down the MRO. Pass the same `env` that goes
    to Policy.make()."""
    return EnvStats.from_env(env, max_episode_steps=self.get_max_traj_length(env),
                             **self.merged_stats())


  def get_max_traj_length(self, env, max_length = 1000):
    spec = getattr(env, "spec", None)
    if spec is not None:
        max_steps = getattr(spec, "max_episode_steps", None)
        if max_steps is not None:
            return max_steps

    max_steps = getattr(env, "_max_episode_steps", None)
    if max_steps is not None:
        return max_steps

    # Walk down the wrapper stack -- a TimeLimit further inside may not be reachable above.
    current = env
    while hasattr(current, "env"):
        current = current.env

        spec = getattr(current, "spec", None)
        if spec is not None:
            max_steps = getattr(spec, "max_episode_steps", None)
            if max_steps is not None:
                return max_steps

        max_steps = getattr(current, "_max_episode_steps", None)
        if max_steps is not None:
            return max_steps

    return max_length

def register_environment(cls : type):

  name = cls.__name__
  if name in Environment._classes:
    raise ValueError(f"Environment {name} already registered")

  Environment._classes[name] = cls

  return cls


# ---------------------------------------------------------------------------
# Exact state capture -- opt-in, for environments with demonstrably complete
# state capture only. `Environment.supports_exact_resume` is `False` for
# every environment by default (above); ordinary resume (episodes restart,
# pending action chunks discarded -- see training_checkpoint.py) is every
# environment's behavior unless it declares otherwise here, explicitly, per
# environment family. A generic wrapper that copies only the base env's RNG
# is NOT sufficient -- gymnasium's own `TimeLimit` wrapper carries its own
# elapsed-step counter outside the unwrapped env entirely, and simulator
# internals (mujoco qpos/qvel and beyond) are a different capture surface
# again.
#
# This is the mechanism, not a claim that every environment supports it:
# only `examples`/`tests` declare a concrete adapter today (Pendulum-v1's
# classic_control physics), verified by a real save/restore continuation
# test (tests/test_exact_resume.py), not merely "it runs". Not wired into
# trainer.py's resume flow (which still restarts episodes for every
# environment) -- see that gap called out in the README.
# ---------------------------------------------------------------------------


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
