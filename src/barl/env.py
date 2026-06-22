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

  @abstractmethod
  def get_env(self) -> gym.Env:
    ...
  
  @abstractmethod
  def get_dataset(self) -> SingleStepDataset:
    ...

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
