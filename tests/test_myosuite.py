"""MyoSuite adapter regressions and optional real-simulator integration checks."""

import sys
from contextlib import closing
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from barl_jax.collection import Collector
from barl_jax.env import Environment
from barl_jax.environments.myosuite import _TASKS, MyoSuiteAdapter
from barl_jax.evaluation import episode_success
from barl_jax.replay import ReplayBuffer


class SolvedEnv(gym.Env):
  """Deterministic episode with non-consecutive successful steps."""

  dt = 0.01
  observation_space = gym.spaces.Box(-1, 1, (1,), np.float32)
  action_space = gym.spaces.Box(-1, 1, (1,), np.float32)
  mj_model, mj_data = object(), object()

  def reset(self, *, seed=None, options=None):
    self.t = 0
    return np.zeros(1, np.float32), {}

  def step(self, action):
    self.t += 1
    return (np.zeros(1, np.float32), 0.0, self.t == 12, False,
            {"solved": np.asarray(self.t % 2 == 1)})


def test_success_matches_myosuite_episode_metric_and_resets():
  env = MyoSuiteAdapter(SolvedEnv())
  env.reset()
  for _ in range(10):  # Five solved steps is below MyoSuite's success threshold.
    *_, info = env.step(np.zeros(1))
    assert not info["success"]
  env.step(np.zeros(1))
  _, _, term, _, info = env.step(np.zeros(1))
  assert term and not info["solved"]
  assert episode_success(info, 0, False, term)
  env.reset()
  *_, info = env.step(np.zeros(1))
  assert not info["success"]


def test_rendering_is_lazy_and_closes_its_context(monkeypatch):
  calls = []
  raw = SolvedEnv()
  raw.mj_renderer = SimpleNamespace(
    _update_camera_properties=lambda camera: calls.append(("camera", camera)),
    _update_renderer_settings=lambda option: calls.append(("option", option)))

  class Renderer:
    def __init__(self, model, *, height, width):
      calls.append(("create", model, height, width))

    def update_scene(self, data, *, camera, scene_option):
      calls.append(("update", data, camera, scene_option))

    def render(self):
      return np.ones((480, 640, 3), np.uint8)

    def close(self):
      calls.append(("close",))

  monkeypatch.setitem(sys.modules, "mujoco", SimpleNamespace(
    Renderer=Renderer, MjvCamera=object, MjvOption=object))
  with MyoSuiteAdapter(SolvedEnv()) as env:
    env.reset()
    assert env.render() is None
  assert not calls

  env = MyoSuiteAdapter(raw, render=True)
  assert env.render_mode == "rgb_array" and env.metadata["render_fps"] == 100
  assert not calls
  assert env.render().shape == (480, 640, 3)
  env.render()
  assert sum(c[0] == "create" for c in calls) == 1
  env.close()
  env.close()
  assert sum(c[0] == "close" for c in calls) == 1


def test_discovery_without_myosuite_and_actionable_missing_dependency(monkeypatch):
  monkeypatch.setitem(sys.modules, "myosuite", None)
  assert set(_TASKS) <= set(Environment.available())
  family = Environment.make("MyoElbowPose1D6MRandom")
  with pytest.raises(ModuleNotFoundError, match="uv sync"):
    family.get_env()


@pytest.fixture
def myosuite_installed():
  pytest.importorskip("myosuite", reason="optional MyoSuite simulator is not installed")


@pytest.mark.parametrize("name", sorted(_TASKS))
def test_real_tasks_seeded_resets_spaces_and_steps(myosuite_installed, name):
  family = Environment.make(name)
  with family.get_env() as env:
    first, _ = env.reset(seed=123)
    assert env.observation_space.contains(first)
    assert np.all(env.action_space.low == -1) and np.all(env.action_space.high == 1)
    stats = family.get_env_stats(env)
    assert stats["max_episode_steps"] == env.spec.max_episode_steps
    assert stats["obs_dim"] == first.size
    assert stats["return_min"] < 0 < stats["return_max"]
    action = np.zeros(env.action_space.shape, np.float32)
    first_step = env.step(action)
    assert env.observation_space.contains(first_step[0])
    assert np.isfinite(first_step[1])
    assert isinstance(first_step[4]["success"], bool)
    for _ in range(2):
      env.step(action)
    again, _ = env.reset(seed=123)
    np.testing.assert_array_equal(first, again)
    again_step = env.step(action)
    np.testing.assert_array_equal(first_step[0], again_step[0])
    assert first_step[1:4] == again_step[1:4]
    if "Random" in name:
      different, _ = env.reset(seed=456)
      assert not np.array_equal(first, different)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_vector_collection_crosses_time_limits(myosuite_installed, asynchronous):
  family = Environment.make("MyoElbowPose1D6MRandom")
  with closing(family.get_vector_env(2, asynchronous=asynchronous)) as env:
    buffer = ReplayBuffer(256, seed=7)
    collector = Collector(seed=7)
    # Two 100-step episodes finish during this collection; the collector must
    # reset their slots explicitly and preserve the pre-reset next observation.
    collector.random_fill(env, n_steps=104, seed=7, sink=buffer)
    assert buffer.n_transitions == 208
    fields = buffer.snapshot()["fields"]
    assert np.isfinite(fields["obs"]).all() and np.isfinite(fields["next_obs"]).all()
    assert not fields["term"].any(), "time limits must keep critic bootstrapping"
    assert collector.state is not None
