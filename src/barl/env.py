import enum
from abc import ABC, abstractmethod

from barl.dataset import SingleStepDataset
from wrapt import decorator
import gymnasium as gym

"""
I need a universal and easy method to call environments and datasets (if they are provided)

"""


class Environment(ABC):

  _classes = dict()

  # Physics-heavy envs (MuJoCo Gym, DeepMind Control Suite) benefit from
  # real OS-level parallelism across CPU cores: mujoco's C step releases the
  # GIL, but SyncVectorEnv runs N sub-envs sequentially in one process, so it
  # never actually overlaps that compute. Override to True to vectorize via
  # AsyncVectorEnv (one subprocess per sub-env) instead -- worth it once
  # per-step physics cost dominates per-process IPC overhead; leave False
  # (default) for cheap/lightweight envs, or ones with expensive
  # per-process setup (e.g. loading large offline-dataset assets) where
  # subprocess startup would outweigh the benefit.
  use_async_vector_env = False

  @abstractmethod
  def get_env(self) -> gym.Env:
    ...

  @abstractmethod
  def get_dataset(self) -> SingleStepDataset:
    ...

  def get_vector_env(self, num_envs: int) -> gym.vector.VectorEnv:
    """N-way vectorized env for online rollout collection (see
    ReplayBuffer.rollout). autoreset_mode=DISABLED is mandatory -- rollout()
    resets finished sub-envs itself via reset_mask so a fresh Trajectory
    chunk always starts from a real initial observation."""
    cls = gym.vector.AsyncVectorEnv if self.use_async_vector_env else gym.vector.SyncVectorEnv
    return cls([self.get_env for _ in range(num_envs)], autoreset_mode=gym.vector.AutoresetMode.DISABLED)

  @staticmethod
  def make(name : str) -> "Environment":
    return Environment._classes[name]()


  @staticmethod
  def available() -> list[str]:
    return list(Environment._classes.keys())


def register_environment(cls : type):

  name = cls.__name__
  if name in Environment._classes:
    raise ValueError(f"Environment {name} already registered")

  Environment._classes[name] = cls

  return cls
