from contextlib import closing
from types import SimpleNamespace
from unittest.mock import Mock

import gymnasium as gym
import jax.numpy as jnp
import numpy as np
import pytest

from barl_jax import trainer


@pytest.mark.parametrize('num_seeds,episodes', [(1, 3), (3, 1), (3, 4)])
def test_evaluation_reuses_seed_slots_and_groups_rounds(monkeypatch, tmp_path, num_seeds, episodes):
  class Episode(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)

    def __init__(self, index):
      self.index = index
      self.seeds = []

    def reset(self, *, seed=None, options=None):
      super().reset(seed=seed)
      self.seeds.append(seed)
      self.steps = 0
      return np.zeros(1, np.float32), {}

    def step(self, action):
      if self.index == num_seeds - 1 or action[0] != 0:
        assert action[0] == pytest.approx(self.index / num_seeds)
      self.steps += 1
      return np.zeros(1, np.float32), float(self.index + 1), self.steps == self.index + 1, False, {}

  def actor_factory(graphdef, state, seeds, width, reference, **kwargs):
    assert (seeds, width) == (num_seeds, 1)
    return SimpleNamespace(chunk_size=1, action_dim=1, num_seeds=num_seeds,
      inference_fn=lambda: lambda obs, key: (jnp.arange(num_seeds)[:, None] / num_seeds, {}))

  monkeypatch.setattr(trainer, 'StackedPolicy', actor_factory)
  monkeypatch.setattr(trainer, 'unstack', lambda *args: (None, None))
  monkeypatch.setattr(trainer, 'save_policy', Mock())
  run = SimpleNamespace(cfg=SimpleNamespace(eval_runs=episodes, eval_every=10),
                        base_seed=0, eval_count=0, num_seeds=num_seeds,
                        graphdef=None, state=None, reference=None, eval_executor=None)
  log = Mock()
  with closing(gym.vector.SyncVectorEnv([lambda i=i: Episode(i) for i in range(num_seeds)],
                               autoreset_mode=gym.vector.AutoresetMode.DISABLED)) as env:
    best, _ = trainer.evaluate(run, (env, None), log, str(tmp_path),
                               [-np.inf] * num_seeds, want_video=False)
    assert best == [(i + 1) ** 2 for i in range(num_seeds)]
    metrics = log.log_per_seed.call_args.args[0]
    assert [m['eval/avg_length'] for m in metrics] == list(range(1, num_seeds + 1))
    assert [m['eval/avg_cum_reward'] for m in metrics] == best
    # Longest slot completes exactly one episode per round; shorter slots may
    # be reset while waiting, but only their first episode is recorded.
    assert len(env.envs[-1].seeds) == episodes
    assert len(set(env.envs[-1].seeds)) == episodes
  assert run.eval_count == 1


def test_want_video_renders_exactly_one_episode_and_rotates_seeds(monkeypatch, tmp_path):
  """render_env holds one offscreen GL context, not one per seed (see
  evaluate()'s video-block comment) -- so exactly one episode is rendered per
  eval round, and it rotates across training seeds round to round rather than
  always the same one."""
  num_seeds = 3

  class EvalEpisode(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
      super().reset(seed=seed)
      self.steps = 0
      return np.zeros(1, np.float32), {}

    def step(self, action):
      self.steps += 1
      return np.zeros(1, np.float32), 1.0, self.steps == 1, False, {}

  class RenderEpisode(EvalEpisode):
    def render(self):
      return np.zeros((2, 2, 3), dtype=np.uint8)

  def actor_factory(graphdef, state, seeds, width, reference, **kwargs):
    return SimpleNamespace(chunk_size=1, action_dim=1, num_seeds=num_seeds,
      inference_fn=lambda: lambda obs, key: (jnp.zeros((num_seeds, 1)), {}))

  # unstack(graphdef, state, index) -> (policy, optimizer); SeedActor only
  # ever reads chunk_size/action_dim/inference_fn off the policy half.
  seed_policy = SimpleNamespace(chunk_size=1, action_dim=1,
    inference_fn=lambda executor=None: (lambda obs, key: (jnp.zeros((1, 1)), {})))

  monkeypatch.setattr(trainer, 'StackedPolicy', actor_factory)
  monkeypatch.setattr(trainer, 'unstack', lambda *args: (seed_policy, None))
  monkeypatch.setattr(trainer, 'save_policy', Mock())
  run = SimpleNamespace(cfg=SimpleNamespace(eval_runs=1, eval_every=10),
                        base_seed=0, eval_count=0, num_seeds=num_seeds,
                        graphdef=None, state=None, reference=None, eval_executor=None)
  log = Mock()
  # render_env sized 1, matching _train()'s construction (one offscreen GL
  # context total, however many training seeds this run has).
  with closing(gym.vector.SyncVectorEnv([EvalEpisode] * num_seeds,
                                        autoreset_mode=gym.vector.AutoresetMode.DISABLED)) as eval_env, \
       closing(gym.vector.SyncVectorEnv([RenderEpisode],
                                        autoreset_mode=gym.vector.AutoresetMode.DISABLED)) as render_env:
    seed_indices = []
    for _ in range(num_seeds + 1):
      trainer.evaluate(run, (eval_env, render_env), log, str(tmp_path),
                       [-np.inf] * num_seeds, want_video=True)
      seed_indices.append(log.log_video.call_args.kwargs["seed_index"])

  assert log.log_video.call_count == num_seeds + 1
  assert {call.args[0] for call in log.log_video.call_args_list} == {"eval/video"}
  # One eval round in, one video out -- and it cycles 0, 1, 2, 0, ... rather
  # than pinning the same seed or rendering every seed at once.
  assert seed_indices == [0, 1, 2, 0]
