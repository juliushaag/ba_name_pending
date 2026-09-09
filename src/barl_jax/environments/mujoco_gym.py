import gymnasium as gym

from barl_jax.env import Environment, register_environment


class MuJoCoGymEnvironment(Environment):
  """Wraps gymnasium's built-in MuJoCo `-v5` envs. Online-only --
  get_dataset() always raises. Unlike HumanoidBench's Task, gymnasium's own
  MujocoRenderer sets up its GL/EGL context lazily on the first real
  render() call, so render_mode="rgb_array" would be free even for the N
  parallel rollout envs that never render -- gated on `render` anyway, to
  keep every Environment family to the same contract."""

  use_async_vector_env = True

  def mujoco_id(self) -> str:
    raise NotImplementedError

  def get_env(self, *, render: bool = False) -> gym.Env:
    return gym.make(self.mujoco_id(), render_mode="rgb_array" if render else None)

  def get_dataset(self):
    raise NotImplementedError(f"{type(self).__name__} has no offline dataset -- MuJoCo Gym is online-only.")

@register_environment
class HalfCheetah(MuJoCoGymEnvironment):
  # 0.06 ms/env-step -- measured 16.1k env-steps/s sync vs 13.4k async.
  use_async_vector_env = False
  stats = {"return_min": -3200.0, "return_max": 3200.0}
  def mujoco_id(self): return "HalfCheetah-v5"

@register_environment
class Hopper(MuJoCoGymEnvironment):
  stats = {"return_min": -900.0, "return_max": 900.0}
  def mujoco_id(self): return "Hopper-v5"

@register_environment
class Walker2d(MuJoCoGymEnvironment):
  stats = {"return_min": -1400.0, "return_max": 1400.0}
  def mujoco_id(self): return "Walker2d-v5"

@register_environment
class Ant(MuJoCoGymEnvironment):
  stats = {"return_min": -1600.0, "return_max": 1600.0}
  def mujoco_id(self): return "Ant-v5"

@register_environment
class Humanoid(MuJoCoGymEnvironment):
  stats = {"return_min": -1600.0, "return_max": 1600.0}
  def mujoco_id(self): return "Humanoid-v5"

@register_environment
class HumanoidStandup(MuJoCoGymEnvironment):
  stats = {"return_min": -80000.0, "return_max": 80000.0}
  def mujoco_id(self): return "HumanoidStandup-v5"

@register_environment
class InvertedPendulum(MuJoCoGymEnvironment):
  stats = {"return_min": -200.0, "return_max": 200.0}
  def mujoco_id(self): return "InvertedPendulum-v5"

@register_environment
class InvertedDoublePendulum(MuJoCoGymEnvironment):
  stats = {"return_min": -2000.0, "return_max": 2000.0}
  def mujoco_id(self): return "InvertedDoublePendulum-v5"

@register_environment
class Reacher(MuJoCoGymEnvironment):
  stats = {"return_min": -160.0, "return_max": 160.0}
  def mujoco_id(self): return "Reacher-v5"

@register_environment
class Swimmer(MuJoCoGymEnvironment):
  stats = {"return_min": -80.0, "return_max": 80.0}
  def mujoco_id(self): return "Swimmer-v5"

@register_environment
class Pusher(MuJoCoGymEnvironment):
  stats = {"return_min": -380.0, "return_max": 380.0}
  def mujoco_id(self): return "Pusher-v5"
