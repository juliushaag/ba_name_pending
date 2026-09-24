import gymnasium as gym

from barl_jax.env import Environment, register_tasks


class MuJoCoGymEnvironment(Environment):
  """Wraps gymnasium's built-in MuJoCo `-v5` envs. Online-only. Unlike
  HumanoidBench's Task, gymnasium's own MujocoRenderer sets up its GL/EGL context
  lazily on the first real render() call, so render_mode="rgb_array" would be free
  even for the N parallel rollout envs that never render -- gated on `render`
  anyway, to keep every Environment family to the same contract."""

  use_async_vector_env = True

  def mujoco_id(self) -> str:
    raise NotImplementedError

  def get_env(self, *, render: bool = False) -> gym.Env:
    return gym.make(self.mujoco_id(), render_mode="rgb_array" if render else None)


# name -> gymnasium id
_TASKS = {name: f"{name}-v5" for name in (
  "HalfCheetah", "Hopper", "Walker2d", "Ant", "Humanoid", "HumanoidStandup",
  "InvertedPendulum", "InvertedDoublePendulum", "Reacher", "Swimmer", "Pusher")}

# Symmetric return bounds. Humanoid and Ant are dime_am's own values.
_RETURN_BOUNDS = {
  "HalfCheetah": 3200.0, "Hopper": 900.0, "Walker2d": 1400.0, "Ant": 1600.0,
  "Humanoid": 1600.0, "HumanoidStandup": 80000.0, "InvertedPendulum": 200.0,
  "InvertedDoublePendulum": 2000.0, "Reacher": 160.0, "Swimmer": 80.0, "Pusher": 380.0,
}

register_tasks(MuJoCoGymEnvironment, "mujoco_id", _TASKS, into=globals(), bounds=_RETURN_BOUNDS,
               # 0.06 ms/env-step: 16.1k env-steps/s sync vs 13.4k async.
               extra={"HalfCheetah": {"use_async_vector_env": False}})
