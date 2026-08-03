import enum
from abc import ABC, abstractmethod

from barl_jax.env_stats import EnvStats
import gymnasium as gym


class Environment(ABC):

  _classes = dict()

  # SyncVectorEnv runs N sub-envs sequentially in one process, so mujoco's
  # GIL-releasing C step never actually overlaps -- override to True for
  # AsyncVectorEnv (one subprocess per sub-env) once per-step physics cost
  # dominates IPC overhead.
  use_async_vector_env = False

  # Env facts no gym space can report -- `return_min`/`return_max` bound the
  # achievable episode return, which is what HLGaussCritic's fixed support needs.
  # Merged down the MRO, so a family base can set a default and one subclass
  # override a single key.
  stats: dict = {}

  @abstractmethod
  def get_env(self) -> gym.Env:
    ...

  @abstractmethod
  def get_dataset(self):
    """Only implemented by offline-dataset-backed environments."""
    ...

  def get_vector_env(self, num_envs: int, asynchronous: bool = None) -> gym.vector.VectorEnv:
    """autoreset_mode=DISABLED is mandatory -- ReplayBuffer.rollout() resets
    finished sub-envs itself via reset_mask. `asynchronous` overrides
    `use_async_vector_env` for a caller the class-level choice doesn't suit --
    a rendering env has to be synchronous, since a forked worker's render()
    comes back as an EOFError (MuJoCo's EGL context does not survive the
    fork) and its frames would go through the pipe at ~900 KB a step."""
    use_async = self.use_async_vector_env if asynchronous is None else asynchronous
    cls = gym.vector.AsyncVectorEnv if use_async else gym.vector.SyncVectorEnv
    return cls([self.get_env for _ in range(num_envs)], autoreset_mode=gym.vector.AutoresetMode.DISABLED)

  @staticmethod
  def make(name : str) -> "Environment":
    return Environment._classes[name]()


  @staticmethod
  def available() -> list[str]:
    return list(Environment._classes.keys())


  def get_env_stats(self, env: gym.Env) -> EnvStats:
    """The env-derived facts every component is built with -- spaces plus this
    class's declared `stats`. Pass the same `env` that goes to Policy.make()."""
    declared = {}
    for klass in reversed(type(self).__mro__):
      declared.update(klass.__dict__.get("stats") or {})
    return EnvStats.from_env(env, max_episode_steps=self.get_max_traj_length(env), **declared)


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
