from barl.dataset import SingleStepDataset
from barl.env import Environment, register_environment

import gymnasium as gym


class MuJoCoGymEnvironment(Environment):
  """Wraps gymnasium's built-in MuJoCo `-v5` envs. Online-only --
  get_dataset() always raises. render_mode="rgb_array" is free for the N
  parallel rollout envs that never render (MujocoRenderer sets up its
  context lazily on the first real render() call)."""

  # Async pays off once the per-step physics cost clears the pickle/pipe
  # round-trip: measured 2.1x on Humanoid (0.36 ms/env-step) but 0.83x on
  # HalfCheetah (0.06 ms/env-step), where the sub-envs are too cheap to amortize
  # it. Overridden to False per-env for the cheap ones.
  use_async_vector_env = True

  def mujoco_id(self) -> str:
    raise NotImplementedError

  def get_env(self) -> gym.Env:
    return gym.make(self.mujoco_id(), render_mode="rgb_array")

  def get_dataset(self) -> SingleStepDataset:
    raise NotImplementedError(f"{type(self).__name__} has no offline dataset -- MuJoCo Gym is online-only.")

@register_environment
class HalfCheetah(MuJoCoGymEnvironment):
  # 0.06 ms/env-step -- measured 16.1k env-steps/s sync vs 13.4k async.
  use_async_vector_env = False
  stats = {"return_min": -1000.0, "return_max": 16000.0}
  def mujoco_id(self): return "HalfCheetah-v5"

@register_environment
class Hopper(MuJoCoGymEnvironment):
  stats = {"return_min": 0.0, "return_max": 4500.0}
  def mujoco_id(self): return "Hopper-v5"

@register_environment
class Walker2d(MuJoCoGymEnvironment):
  stats = {"return_min": 0.0, "return_max": 7000.0}
  def mujoco_id(self): return "Walker2d-v5"

@register_environment
class Ant(MuJoCoGymEnvironment):
  stats = {"return_min": -2000.0, "return_max": 9000.0}
  def mujoco_id(self): return "Ant-v5"

@register_environment
class Humanoid(MuJoCoGymEnvironment):
  stats = {"return_min": 0.0, "return_max": 12000.0}
  def mujoco_id(self): return "Humanoid-v5"

@register_environment
class HumanoidStandup(MuJoCoGymEnvironment):
  stats = {"return_min": 0.0, "return_max": 400000.0}
  def mujoco_id(self): return "HumanoidStandup-v5"

@register_environment
class InvertedPendulum(MuJoCoGymEnvironment):
  stats = {"return_min": 0.0, "return_max": 1000.0}
  def mujoco_id(self): return "InvertedPendulum-v5"

@register_environment
class InvertedDoublePendulum(MuJoCoGymEnvironment):
  stats = {"return_min": 0.0, "return_max": 10000.0}
  def mujoco_id(self): return "InvertedDoublePendulum-v5"

@register_environment
class Reacher(MuJoCoGymEnvironment):
  stats = {"return_min": -100.0, "return_max": 0.0}
  def mujoco_id(self): return "Reacher-v5"

@register_environment
class Swimmer(MuJoCoGymEnvironment):
  stats = {"return_min": -100.0, "return_max": 400.0}
  def mujoco_id(self): return "Swimmer-v5"

@register_environment
class Pusher(MuJoCoGymEnvironment):
  stats = {"return_min": -300.0, "return_max": 0.0}
  def mujoco_id(self): return "Pusher-v5"
