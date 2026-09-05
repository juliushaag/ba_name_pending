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
    from barl_jax.registration import import_type, load_builtins
    load_builtins("environments")
    cls = import_type(name) if ":" in name else Environment._classes[name]
    if not issubclass(cls, Environment):
      raise TypeError(f"{name!r} must implement Environment")
    return cls

  @staticmethod
  def available() -> list[str]:
    from barl_jax.registration import load_builtins
    load_builtins("environments")
    return list(Environment._classes.keys())


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
