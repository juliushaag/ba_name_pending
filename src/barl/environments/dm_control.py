from barl.dataset import SingleStepDataset
from barl.env import Environment, register_environment

from dm_control import suite
from shimmy.dm_control_compatibility import DmControlCompatibilityV0

import gymnasium as gym


class DMControlEnvironment(Environment):
  """
  Wraps a DeepMind Control Suite (dm_control.suite) domain/task pair into a
  gymnasium.Env via shimmy, then flattens its observation: dm_control tasks
  expose a Dict of named arrays (e.g. {"position":..., "velocity":...}), but
  Policy/MLP throughout this framework assume a single flat Box observation
  (Policy.__init__ reads env.observation_space.shape directly), same as
  every other suite here. Online-only, like MuJoCoGymEnvironment -- no
  offline dataset.

  render_mode="rgb_array" is set unconditionally: shimmy's compatibility
  wrapper only builds a renderer eagerly for render_mode="human" (see
  DmControlCompatibilityV0.__init__); "rgb_array" just calls
  physics.render() lazily on demand, so this is free for envs that never
  render, same reasoning as MuJoCoGymEnvironment.
  """

  use_async_vector_env = True

  def domain_task(self) -> tuple[str, str]:
    raise NotImplementedError

  def get_env(self) -> gym.Env:
    domain, task = self.domain_task()
    dm_env = suite.load(domain_name=domain, task_name=task)
    env = DmControlCompatibilityV0(dm_env, render_mode="rgb_array")
    return gym.wrappers.FlattenObservation(env)

  def get_dataset(self) -> SingleStepDataset:
    raise NotImplementedError(f"{type(self).__name__} has no offline dataset -- DeepMind Control Suite is online-only.")


@register_environment
class CartpoleSwingup(DMControlEnvironment):
  def domain_task(self): return ("cartpole", "swingup")

@register_environment
class CartpoleBalance(DMControlEnvironment):
  def domain_task(self): return ("cartpole", "balance")

@register_environment
class CheetahRun(DMControlEnvironment):
  def domain_task(self): return ("cheetah", "run")

@register_environment
class WalkerWalk(DMControlEnvironment):
  def domain_task(self): return ("walker", "walk")

@register_environment
class WalkerRun(DMControlEnvironment):
  def domain_task(self): return ("walker", "run")

@register_environment
class HopperHop(DMControlEnvironment):
  def domain_task(self): return ("hopper", "hop")

@register_environment
class HopperStand(DMControlEnvironment):
  def domain_task(self): return ("hopper", "stand")

@register_environment
class HumanoidWalk(DMControlEnvironment):
  def domain_task(self): return ("humanoid", "walk")

@register_environment
class HumanoidRun(DMControlEnvironment):
  def domain_task(self): return ("humanoid", "run")

@register_environment
class QuadrupedWalk(DMControlEnvironment):
  def domain_task(self): return ("quadruped", "walk")

@register_environment
class QuadrupedRun(DMControlEnvironment):
  def domain_task(self): return ("quadruped", "run")

@register_environment
class FingerSpin(DMControlEnvironment):
  def domain_task(self): return ("finger", "spin")

@register_environment
class FishSwim(DMControlEnvironment):
  def domain_task(self): return ("fish", "swim")

@register_environment
class AcrobotSwingup(DMControlEnvironment):
  def domain_task(self): return ("acrobot", "swingup")

@register_environment
class PendulumSwingup(DMControlEnvironment):
  def domain_task(self): return ("pendulum", "swingup")

@register_environment
class BallInCupCatch(DMControlEnvironment):
  def domain_task(self): return ("ball_in_cup", "catch")

@register_environment
class ReacherEasy(DMControlEnvironment):
  def domain_task(self): return ("reacher", "easy")

@register_environment
class ReacherHard(DMControlEnvironment):
  def domain_task(self): return ("reacher", "hard")
