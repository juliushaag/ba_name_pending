"""CPU MuJoCo MyoSuite tasks, with seeded resets and episode success metrics.

Import MyoSuite only when a task is constructed: discovering environments must
also work in installations without this simulator or its model assets.
"""

import gymnasium as gym
import numpy as np

from barl_jax.env import Environment, register_tasks


class MyoSuiteAdapter(gym.Wrapper):
  """Adapt MyoSuite's seeding, success and offscreen rendering to our contract.

  MyoSuite's evaluate_success counts an episode as successful after more than
  five solved steps (not necessarily consecutive). Keep that count across the
  episode so the evaluator can read a single final info['success'] flag.
  """

  def __init__(self, env: gym.Env, *, render: bool = False):
    super().__init__(env)
    self._render_mode = "rgb_array" if render else None
    self.metadata = {"render_modes": ["rgb_array"],
                     "render_fps": max(1, round(1 / env.unwrapped.dt))}
    self._renderer = None
    self._camera = None
    self._scene_option = None
    self._solved_steps = 0

  @property
  def render_mode(self):
    return self._render_mode

  def reset(self, *, seed=None, options=None):
    # Task resets sample targets and joint positions before the base reset;
    # the base reset passes seed to Robot.reset(), which does not reseed the
    # task RNG. Seed before either stage and keep the robot's noise RNG shared.
    if seed is not None:
      raw = self.env.unwrapped
      raw.seed(seed)
      raw.robot.np_random = raw.np_random
    self._solved_steps = 0
    return self.env.reset(seed=seed, options=options)

  def step(self, action):
    obs, reward, terminated, truncated, info = self.env.step(action)
    self._solved_steps += int(bool(np.asarray(info["solved"]).item()))
    return obs, reward, terminated, truncated, {**info, "success": self._solved_steps > 5}

  def render(self):
    if self.render_mode is None:
      return None
    # Own the renderer so close() releases EGL resources: MyoSuite's env
    # inherits gym.Env.close(), and its renderer's close only closes a window.
    # No graphics context is allocated for collection or return-only eval.
    import mujoco

    raw = self.env.unwrapped
    if self._renderer is None:
      self._renderer = mujoco.Renderer(raw.mj_model, height=480, width=640)
      self._camera = mujoco.MjvCamera()
      self._scene_option = mujoco.MjvOption()
      # Preserve the task's configured camera and muscle visualization. The
      # helpers belong to the pinned MyoSuite release and allocate no GL state.
      raw.mj_renderer._update_camera_properties(self._camera)
      raw.mj_renderer._update_renderer_settings(self._scene_option)
    self._renderer.update_scene(raw.mj_data, camera=self._camera,
                                scene_option=self._scene_option)
    return self._renderer.render()

  def close(self):
    try:
      if self._renderer is not None:
        self._renderer.close()
        self._renderer = None
    finally:
      self.env.close()


class MyoSuiteEnvironment(Environment):
  """One standard MyoBase task; muscle actions retain MyoSuite's normalization."""

  # Start small tasks synchronously; callers can opt into CPU subprocesses
  # through get_vector_env(..., asynchronous=True) for expensive leg physics.
  use_async_vector_env = False

  def myosuite_id(self) -> str:
    raise NotImplementedError

  def get_env(self, *, render: bool = False) -> gym.Env:
    try:
      import myosuite  # noqa: F401 -- registers Gymnasium task IDs
    except ModuleNotFoundError as exc:
      if exc.name != "myosuite":
        raise
      raise ModuleNotFoundError(
        "MyoSuite tasks require myosuite==2.12.2 on Python 3.11–3.13; "
        "run `uv sync` in the repository to install it.") from exc

    # MyoSuite does not implement Gymnasium's render interface itself. Check
    # our complete adapter rather than its inner environment.
    env = gym.make(self.myosuite_id(), disable_env_checker=True)
    return gym.wrappers.PassiveEnvChecker(MyoSuiteAdapter(env, render=render))


_TASKS = {f"Myo{name}": f"myo{name}-v0" for name in (
  "ElbowPose1D6MFixed", "ElbowPose1D6MRandom",
  "FingerPoseFixed", "FingerPoseRandom", "FingerReachFixed", "FingerReachRandom",
  "HandPoseFixed", "HandPoseRandom", "HandReachFixed", "HandReachRandom",
  "LegStandRandom", "LegWalk",
)}

# Discounted Q supports at the default gamma=.99, with symmetric 2x headroom.
# Pose/reach: at most 8 reward/step for 100 steps (~507 discounted), plus
# room for negative distance/activation terms and the one-off -50 failure.
# Standing: at most 18/step for 150 steps (~1401 discounted), plus failure.
# Walking: at most 25/step for 1000 steps (~2500 discounted); the negative
# hip-error shaping and one-off -100 failure fit the symmetric margin in
# normal operation. These are practical supports, not global physics bounds;
# changed rewards/gamma may need explicit critic v_min/v_max overrides.
_RETURN_BOUNDS = {name: 1200.0 for name in _TASKS}
_RETURN_BOUNDS.update(MyoLegStandRandom=3000.0, MyoLegWalk=5000.0)

register_tasks(MyoSuiteEnvironment, "myosuite_id", _TASKS, into=globals(),
               bounds=_RETURN_BOUNDS, extra={
                 "MyoLegStandRandom": {"use_async_vector_env": True},
                 "MyoLegWalk": {"use_async_vector_env": True},
               })
