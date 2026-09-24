import contextlib
import os

import gymnasium as gym

from barl_jax.env import Environment, register_tasks


@contextlib.contextmanager
def _without_mujoco_egl_device_id():
  """MUJOCO_EGL_DEVICE_ID (scripts/train.py) pins mujoco.egl's own device
  enumeration to the node-wide physical GPU index SLURM granted -- needed
  there because mujoco.egl's enumeration ignores cgroup device isolation.
  dm_control's own, separately-implemented EGL renderer
  (dm_control/_render/pyopengl/egl_renderer.py) reads the same env var but
  enumerates devices in its own order, so the same index can name a GPU this
  job's cgroup has no access to -- and unlike mujoco.egl's silent
  wrong-GPU/black-frame failure, it hard-fails: `ImportError: Cannot
  initialize a headless EGL display.` Unset, its device loop tries every
  enumerated device and keeps the first one that actually initializes, which
  is necessarily one this job can access. Scoped to this import rather than
  cleared globally, since a HumanoidBench/MuJoCo-Gym env sharing the process
  still needs the pin for mujoco.egl."""
  saved = os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
  try:
    yield
  finally:
    if saved is not None:
      os.environ["MUJOCO_EGL_DEVICE_ID"] = saved


class DMControlEnvironment(Environment):
  """Wraps a DeepMind Control Suite domain/task pair via shimmy, flattening
  its Dict observation into a single flat Box (every Policy/MLP here assumes
  that shape). Online-only, like MuJoCoGymEnvironment."""

  use_async_vector_env = False

  # dime_am's tuned DMC value, `v_min: -200, v_max: 200` at `n_atoms: 101` --
  # reward in [0, 1] over a 1000-step horizon, so the discounted value is
  # bounded by (1-gamma^T)/(1-gamma) and the support is twice that, symmetric.
  stats = {"return_min": -200.0, "return_max": 200.0}

  def domain_task(self) -> tuple[str, str]:
    raise NotImplementedError

  def get_env(self, *, render: bool = False) -> gym.Env:
    with _without_mujoco_egl_device_id():
      from dm_control import suite
      from shimmy.dm_control_compatibility import DmControlCompatibilityV0
      domain, task = self.domain_task()
      dm_env = suite.load(domain_name=domain, task_name=task)
    env = DmControlCompatibilityV0(dm_env, render_mode="rgb_array" if render else None)
    return gym.wrappers.FlattenObservation(env)


# name -> (domain, task)
_TASKS = {
  "DogRun": ("dog", "run"),
  "DogStand": ("dog", "stand"),
  "DogWalk": ("dog", "walk"),
  "DogTrot": ("dog", "trot"),
  "HumanoidWalk": ("humanoid", "walk"),
  "HumanoidRun": ("humanoid", "run"),
  "HumanoidStand": ("humanoid", "stand"),
  "CartpoleSwingup": ("cartpole", "swingup"),
  "CartpoleBalance": ("cartpole", "balance"),
  "CheetahRun": ("cheetah", "run"),
  "WalkerWalk": ("walker", "walk"),
  "WalkerRun": ("walker", "run"),
  "HopperHop": ("hopper", "hop"),
  "HopperStand": ("hopper", "stand"),
  "QuadrupedWalk": ("quadruped", "walk"),
  "QuadrupedRun": ("quadruped", "run"),
  "FingerSpin": ("finger", "spin"),
  "FishSwim": ("fish", "swim"),
  "AcrobotSwingup": ("acrobot", "swingup"),
  "PendulumSwingup": ("pendulum", "swingup"),
  "BallInCupCatch": ("ball_in_cup", "catch"),
  "ReacherEasy": ("reacher", "easy"),
  "ReacherHard": ("reacher", "hard"),
}

register_tasks(DMControlEnvironment, "domain_task", _TASKS, into=globals())
