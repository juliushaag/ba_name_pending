from barl_jax.env import Environment, register_environment

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

  # Every `stats` below is a symmetric bound on the *discounted* value a critic
  # regresses, not the undiscounted episode return -- at gamma 0.99 over a
  # 1000-step horizon the two differ by 10x. Following dime_am's convention:
  # twice the discounted bound `R*(1 - gamma^T)/(T*(1 - gamma))`, symmetric, for
  # over-estimation headroom. That rule reproduces its DMC ±200 exactly.
  #
  # Humanoid and Ant are dime_am's own numbers (README: `alg.critic.v_min=-1600
  # alg.critic.v_max=1600`, "the same values for both gym environments"). The
  # rest follow the rule -- it has no reference behind it, and Reacher's T=50
  # and Pusher's T=100 make their factor 0.790 and 0.634 rather than 0.100.

  def mujoco_id(self) -> str:
    raise NotImplementedError

  def get_env(self) -> gym.Env:
    return gym.make(self.mujoco_id(), render_mode="rgb_array")

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
