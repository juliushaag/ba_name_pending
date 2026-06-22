from abc import abstractmethod

from barl.dataset import SingleStepDataset
import ogbench

from barl.env import register_environment, Environment

import gymnasium as gym


class OGBenchEnvironment(Environment):

  @abstractmethod
  def ogbench_name(self):
    ...

  def get_env(self) -> gym.Env:
    return ogbench.make_env_and_datasets(self.ogbench_name(), env_only=True)

  def get_dataset(self) -> SingleStepDataset:
    res = ogbench.make_env_and_datasets(self.ogbench_name(), compact_dataset=True)
    return SingleStepDataset.from_ogbench(res[1])

@register_environment
class CubeSingle(OGBenchEnvironment):
  def ogbench_name(self):
    return "cube-single-play-singletask-task2-v0"

@register_environment
class CubeDouble(OGBenchEnvironment):
  def ogbench_name(self):
    return "cube-double-play-singletask-task2-v0"

  
