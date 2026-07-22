import numpy as np

import gymnasium as gym

from barl.dataset import SingleStepDataset
from barl.env import Environment, register_environment


class _MjxGymEnv(gym.Env):
  """Adapts one mujoco_playground MjxEnv into a plain numpy-in/numpy-out
  gymnasium.Env: one jax.jit'd reset/step call per Env.reset()/step(), a
  jax.random.PRNGKey threaded through (derived from self.np_random, gym's own
  seeded RNG) instead of a numpy seed. Playground ships no first-party
  Gymnasium wrapper (unlike dm_control, which shimmy covers), so this is
  barl-specific glue -- and since it steps one env at a time through Python,
  it never gets MJX's actual advantage (batched simulation across envs in a
  single XLA call); see MjxPlaygroundEnvironment for why AsyncVectorEnv is
  deliberately not used here either.

  jax/mujoco_playground are imported lazily inside __init__, not at module
  scope: this module is imported unconditionally at `barl` package-init time
  (see barl/__init__.py) for env-registration purposes alone, and eagerly
  initializing JAX's runtime in every process regardless of whether a
  Playground env is ever used would make an unrelated run pay for it too.
  """

  metadata = {"render_modes": ["rgb_array"]}

  def __init__(self, playground_id: str, render_mode: str = "rgb_array"):
    super().__init__()

    import jax
    import mujoco_playground as playground

    self._jax = jax
    self._playground = playground

    self._env = playground.registry.load(playground_id)
    self._reset_fn = jax.jit(self._env.reset)
    self._step_fn = jax.jit(self._env.step)
    self._state = None
    self.render_mode = render_mode

    obs_size = self._env.observation_size
    if isinstance(obs_size, dict):
      self.observation_space = gym.spaces.Dict({
        k: gym.spaces.Box(-np.inf, np.inf, shape=self._as_shape(v), dtype=np.float32)
        for k, v in obs_size.items()
      })
    else:
      self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=self._as_shape(obs_size), dtype=np.float32)

    # Real actuator bounds (not an assumed [-1, 1]) -- same convention gymnasium's
    # own MuJoCo envs use for their action_space.
    ctrlrange = self._env.mj_model.actuator_ctrlrange
    self.action_space = gym.spaces.Box(
      low=ctrlrange[:, 0].astype(np.float32),
      high=ctrlrange[:, 1].astype(np.float32),
      dtype=np.float32,
    )

  @staticmethod
  def _as_shape(size) -> tuple:
    return (size,) if isinstance(size, int) else tuple(size)

  @staticmethod
  def _to_numpy(obs):
    if isinstance(obs, dict):
      return {k: np.asarray(v) for k, v in obs.items()}
    return np.asarray(obs)

  def reset(self, *, seed=None, options=None):
    super().reset(seed=seed)
    key = self._jax.random.PRNGKey(int(self.np_random.integers(0, 2**31 - 1)))
    self._state = self._reset_fn(key)
    return self._to_numpy(self._state.obs), {}

  def step(self, action):
    self._state = self._step_fn(self._state, self._jax.numpy.asarray(action))
    obs = self._to_numpy(self._state.obs)
    reward = float(self._state.reward)
    # `state.done` doesn't distinguish termination from time-limit truncation --
    # treated as terminated; truncated is always False.
    terminated = bool(self._state.done)
    return obs, reward, terminated, False, {}

  def render(self):
    return self._playground.render_array(self._env.mj_model, self._state)


class MjxPlaygroundEnvironment(Environment):
  """Wraps a MuJoCo Playground (MJX) env via `_MjxGymEnv`. Online-only --
  get_dataset() always raises.

  use_async_vector_env stays False (unlike MuJoCoGymEnvironment/
  DMControlEnvironment): AsyncVectorEnv forks a subprocess per sub-env, and
  forking a process after JAX's runtime is initialized is unsupported --
  the child can inherit a broken copy of JAX's device/XLA state. Since
  `barl/__init__.py` imports this module (for registration) in every
  process regardless of which env a run actually uses, and importing jax
  itself is deferred to `_MjxGymEnv.__init__` (see its docstring), the
  hazard only bites once some other run in the same process has actually
  constructed a Playground env; SyncVectorEnv sidesteps it entirely by
  never forking at all.
  """

  use_async_vector_env = False

  def playground_id(self) -> str:
    raise NotImplementedError

  def get_env(self) -> gym.Env:
    return _MjxGymEnv(self.playground_id())

  def get_dataset(self) -> SingleStepDataset:
    raise NotImplementedError(f"{type(self).__name__} has no offline dataset -- MuJoCo Playground is online-only.")


# ---- DeepMind Control Suite (MJX) ----
@register_environment
class MJXAcrobotSwingup(MjxPlaygroundEnvironment):
  def playground_id(self): return "AcrobotSwingup"

@register_environment
class MJXAcrobotSwingupSparse(MjxPlaygroundEnvironment):
  def playground_id(self): return "AcrobotSwingupSparse"

@register_environment
class MJXBallInCup(MjxPlaygroundEnvironment):
  def playground_id(self): return "BallInCup"

@register_environment
class MJXCartpoleBalance(MjxPlaygroundEnvironment):
  def playground_id(self): return "CartpoleBalance"

@register_environment
class MJXCartpoleBalanceSparse(MjxPlaygroundEnvironment):
  def playground_id(self): return "CartpoleBalanceSparse"

@register_environment
class MJXCartpoleSwingup(MjxPlaygroundEnvironment):
  def playground_id(self): return "CartpoleSwingup"

@register_environment
class MJXCartpoleSwingupSparse(MjxPlaygroundEnvironment):
  def playground_id(self): return "CartpoleSwingupSparse"

@register_environment
class MJXCheetahRun(MjxPlaygroundEnvironment):
  def playground_id(self): return "CheetahRun"

@register_environment
class MJXFingerSpin(MjxPlaygroundEnvironment):
  def playground_id(self): return "FingerSpin"

@register_environment
class MJXFingerTurnEasy(MjxPlaygroundEnvironment):
  def playground_id(self): return "FingerTurnEasy"

@register_environment
class MJXFingerTurnHard(MjxPlaygroundEnvironment):
  def playground_id(self): return "FingerTurnHard"

@register_environment
class MJXFishSwim(MjxPlaygroundEnvironment):
  def playground_id(self): return "FishSwim"

@register_environment
class MJXHopperHop(MjxPlaygroundEnvironment):
  def playground_id(self): return "HopperHop"

@register_environment
class MJXHopperStand(MjxPlaygroundEnvironment):
  def playground_id(self): return "HopperStand"

@register_environment
class MJXHumanoidStand(MjxPlaygroundEnvironment):
  def playground_id(self): return "HumanoidStand"

@register_environment
class MJXHumanoidWalk(MjxPlaygroundEnvironment):
  def playground_id(self): return "HumanoidWalk"

@register_environment
class MJXHumanoidRun(MjxPlaygroundEnvironment):
  def playground_id(self): return "HumanoidRun"

@register_environment
class MJXPendulumSwingup(MjxPlaygroundEnvironment):
  def playground_id(self): return "PendulumSwingup"

@register_environment
class MJXPointMass(MjxPlaygroundEnvironment):
  def playground_id(self): return "PointMass"

@register_environment
class MJXReacherEasy(MjxPlaygroundEnvironment):
  def playground_id(self): return "ReacherEasy"

@register_environment
class MJXReacherHard(MjxPlaygroundEnvironment):
  def playground_id(self): return "ReacherHard"

@register_environment
class MJXSwimmerSwimmer6(MjxPlaygroundEnvironment):
  def playground_id(self): return "SwimmerSwimmer6"

@register_environment
class MJXWalkerRun(MjxPlaygroundEnvironment):
  def playground_id(self): return "WalkerRun"

@register_environment
class MJXWalkerStand(MjxPlaygroundEnvironment):
  def playground_id(self): return "WalkerStand"

@register_environment
class MJXWalkerWalk(MjxPlaygroundEnvironment):
  def playground_id(self): return "WalkerWalk"

# ---- Locomotion ----
@register_environment
class MJXApolloJoystickFlatTerrain(MjxPlaygroundEnvironment):
  def playground_id(self): return "ApolloJoystickFlatTerrain"

@register_environment
class MJXBarkourJoystick(MjxPlaygroundEnvironment):
  def playground_id(self): return "BarkourJoystick"

@register_environment
class MJXBerkeleyHumanoidJoystickFlatTerrain(MjxPlaygroundEnvironment):
  def playground_id(self): return "BerkeleyHumanoidJoystickFlatTerrain"

@register_environment
class MJXBerkeleyHumanoidJoystickRoughTerrain(MjxPlaygroundEnvironment):
  def playground_id(self): return "BerkeleyHumanoidJoystickRoughTerrain"

@register_environment
class MJXG1JoystickFlatTerrain(MjxPlaygroundEnvironment):
  def playground_id(self): return "G1JoystickFlatTerrain"

@register_environment
class MJXG1JoystickRoughTerrain(MjxPlaygroundEnvironment):
  def playground_id(self): return "G1JoystickRoughTerrain"

@register_environment
class MJXGo1JoystickFlatTerrain(MjxPlaygroundEnvironment):
  def playground_id(self): return "Go1JoystickFlatTerrain"

@register_environment
class MJXGo1JoystickRoughTerrain(MjxPlaygroundEnvironment):
  def playground_id(self): return "Go1JoystickRoughTerrain"

@register_environment
class MJXGo1Getup(MjxPlaygroundEnvironment):
  def playground_id(self): return "Go1Getup"

@register_environment
class MJXGo1Handstand(MjxPlaygroundEnvironment):
  def playground_id(self): return "Go1Handstand"

@register_environment
class MJXGo1Footstand(MjxPlaygroundEnvironment):
  def playground_id(self): return "Go1Footstand"

@register_environment
class MJXH1InplaceGaitTracking(MjxPlaygroundEnvironment):
  def playground_id(self): return "H1InplaceGaitTracking"

@register_environment
class MJXH1JoystickGaitTracking(MjxPlaygroundEnvironment):
  def playground_id(self): return "H1JoystickGaitTracking"

@register_environment
class MJXOp3Joystick(MjxPlaygroundEnvironment):
  def playground_id(self): return "Op3Joystick"

@register_environment
class MJXSpotFlatTerrainJoystick(MjxPlaygroundEnvironment):
  def playground_id(self): return "SpotFlatTerrainJoystick"

@register_environment
class MJXSpotGetup(MjxPlaygroundEnvironment):
  def playground_id(self): return "SpotGetup"

@register_environment
class MJXSpotJoystickGaitTracking(MjxPlaygroundEnvironment):
  def playground_id(self): return "SpotJoystickGaitTracking"

@register_environment
class MJXT1JoystickFlatTerrain(MjxPlaygroundEnvironment):
  def playground_id(self): return "T1JoystickFlatTerrain"

@register_environment
class MJXT1JoystickRoughTerrain(MjxPlaygroundEnvironment):
  def playground_id(self): return "T1JoystickRoughTerrain"

# ---- Manipulation ----
@register_environment
class MJXAlohaHandOver(MjxPlaygroundEnvironment):
  def playground_id(self): return "AlohaHandOver"

@register_environment
class MJXAlohaSinglePegInsertion(MjxPlaygroundEnvironment):
  def playground_id(self): return "AlohaSinglePegInsertion"

@register_environment
class MJXPandaPickCube(MjxPlaygroundEnvironment):
  def playground_id(self): return "PandaPickCube"

@register_environment
class MJXPandaPickCubeOrientation(MjxPlaygroundEnvironment):
  def playground_id(self): return "PandaPickCubeOrientation"

@register_environment
class MJXPandaPickCubeCartesian(MjxPlaygroundEnvironment):
  def playground_id(self): return "PandaPickCubeCartesian"

@register_environment
class MJXPandaOpenCabinet(MjxPlaygroundEnvironment):
  def playground_id(self): return "PandaOpenCabinet"

@register_environment
class MJXPandaRobotiqPushCube(MjxPlaygroundEnvironment):
  def playground_id(self): return "PandaRobotiqPushCube"

@register_environment
class MJXLeapCubeReorient(MjxPlaygroundEnvironment):
  def playground_id(self): return "LeapCubeReorient"

@register_environment
class MJXLeapCubeRotateZAxis(MjxPlaygroundEnvironment):
  def playground_id(self): return "LeapCubeRotateZAxis"

@register_environment
class MJXAeroCubeRotateZAxis(MjxPlaygroundEnvironment):
  def playground_id(self): return "AeroCubeRotateZAxis"
