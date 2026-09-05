"""ManiSkill3 tabletop manipulation (Tao et al. 2024), on the CPU backend.

ManiSkill's own draw is its GPU-parallel simulator, which steps N environments
at once and hands back torch tensors already on device. That is deliberately
*not* what this wraps: `CPUGymWrapper` turns one `physx_cpu` env into an
ordinary numpy `gym.Env`, which is the shape everything here already speaks --
`ReplayBuffer.rollout`, `run_episodes`, `EnvStats`, the async workers. The GPU
path would need a torch<->jax bridge and a rollout loop that never touches
numpy, and is a separate piece of work.

Two things differ from the MuJoCo/DMC families and are load-bearing:

* **`terminated` means the task succeeded.** ManiSkill sets `terminated =
  info["success"]` (envs/sapien_env.py), where a MuJoCo humanoid terminates by
  falling over. So `eval/success_rate` cannot be inferred from termination the
  way it is for the other families: `policy.episode_success` reads ManiSkill's
  own `info["success"]` where the env reports one and falls back to the survival
  convention where it does not.
* **Horizons are short and per-task** (50-100 steps, against 1000 for DMC), and
  the reward is `normalized_dense`, i.e. in [0, 1] per step. Both feed the
  declared return support; see config/run/envs.yaml.
"""

import gymnasium as gym

from barl_jax.env import Environment, register_environment


class ManiSkillEnvironment(Environment):
  """One ManiSkill task id, CPU-simulated and unbatched.

  A step costs ~2 ms here -- 5x a DMC humanoid's -- so the physics dominates IPC
  by a wide margin and subprocess workers pay for themselves."""

  use_async_vector_env = True

  # reward is `normalized_dense`, in [0, 1] per step; horizon 50 vs DMC's 1000.
  # Same derivation: twice the discounted bound (1-gamma^T)/(1-gamma) at gamma
  # 0.99, symmetric.
  stats = {"return_min": -79.0, "return_max": 79.0}

  def task_id(self) -> str:
    raise NotImplementedError

  def get_env(self) -> gym.Env:
    import mani_skill.envs  # noqa: F401
    from mani_skill.utils.wrappers import CPUGymWrapper
    # `num_envs=1` + physx_cpu is the unbatched backend; CPUGymWrapper is what
    # unwraps ManiSkill's batched torch tensors back to single-env numpy.
    # `reward_mode` is named rather than defaulted because the declared return
    # support is derived from it being in [0, 1] per step.
    env = gym.make(self.task_id(), obs_mode="state", reward_mode="normalized_dense",
                   sim_backend="physx_cpu", render_mode="rgb_array", num_envs=1)
    return CPUGymWrapper(env)

  def get_dataset(self):
    raise NotImplementedError(
      f"{type(self).__name__} has no offline dataset -- ManiSkill is wired here as an "
      f"online-only benchmark, like MuJoCoGymEnvironment and DMControlEnvironment.")


# ---- horizon 50 ------------------------------------------------------------

@register_environment
class PushCube(ManiSkillEnvironment):
  def task_id(self): return "PushCube-v1"


@register_environment
class PickCube(ManiSkillEnvironment):
  def task_id(self): return "PickCube-v1"


@register_environment
class StackCube(ManiSkillEnvironment):
  def task_id(self): return "StackCube-v1"


@register_environment
class PickSingleYCB(ManiSkillEnvironment):
  """The one task here whose assets ship separately -- the YCB object set (85 MB):

      uv run python -m mani_skill.utils.download_asset ycb -y

  Without it `get_env()` raises at construction; interactively ManiSkill prompts
  for the download, which under a sweep surfaces as an EOFError."""
  def task_id(self): return "PickSingleYCB-v1"


# ---- horizon 100 -----------------------------------------------------------

@register_environment
class PegInsertionSide(ManiSkillEnvironment):
  # Twice the discounted bound at its own horizon of 100.
  stats = {"return_min": -126.79, "return_max": 126.79}
  def task_id(self): return "PegInsertionSide-v1"
