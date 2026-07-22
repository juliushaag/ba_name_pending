from barl.dataset import SingleStepDataset
from barl.env import Environment, register_environment

import gymnasium as gym


class MuJoCoGymEnvironment(Environment):
  """Wraps gymnasium's built-in MuJoCo `-v5` envs. Online-only --
  get_dataset() always raises. render_mode="rgb_array" is free for the N
  parallel rollout envs that never render (MujocoRenderer sets up its
  context lazily on the first real render() call)."""

  use_async_vector_env = True

  def mujoco_id(self) -> str:
    raise NotImplementedError

  def get_env(self) -> gym.Env:
    return gym.make(self.mujoco_id(), render_mode="rgb_array")

  def get_dataset(self) -> SingleStepDataset:
    raise NotImplementedError(f"{type(self).__name__} has no offline dataset -- MuJoCo Gym is online-only.")


@register_environment
class HalfCheetah(MuJoCoGymEnvironment):
  def mujoco_id(self): return "HalfCheetah-v5"

@register_environment
class Hopper(MuJoCoGymEnvironment):
  def mujoco_id(self): return "Hopper-v5"

@register_environment
class Walker2d(MuJoCoGymEnvironment):
  def mujoco_id(self): return "Walker2d-v5"

@register_environment
class Ant(MuJoCoGymEnvironment):
  def mujoco_id(self): return "Ant-v5"

@register_environment
class Humanoid(MuJoCoGymEnvironment):
  def mujoco_id(self): return "Humanoid-v5"

@register_environment
class HumanoidStandup(MuJoCoGymEnvironment):
  def mujoco_id(self): return "HumanoidStandup-v5"

@register_environment
class InvertedPendulum(MuJoCoGymEnvironment):
  def mujoco_id(self): return "InvertedPendulum-v5"

@register_environment
class InvertedDoublePendulum(MuJoCoGymEnvironment):
  def mujoco_id(self): return "InvertedDoublePendulum-v5"

@register_environment
class Reacher(MuJoCoGymEnvironment):
  def mujoco_id(self): return "Reacher-v5"

@register_environment
class Swimmer(MuJoCoGymEnvironment):
  def mujoco_id(self): return "Swimmer-v5"

@register_environment
class Pusher(MuJoCoGymEnvironment):
  def mujoco_id(self): return "Pusher-v5"
