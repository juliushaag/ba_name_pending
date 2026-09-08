"""Public evaluation CLI: discovery and real checkpoint reconstruction."""
import importlib.util
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from barl_jax.checkpoints import save_policy
from barl_jax.env import Environment
from barl_jax.policy import Policy

spec = importlib.util.spec_from_file_location(
  "evaluate", Path(__file__).resolve().parent.parent / "scripts/evaluate.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
checkpoints, environment_name, evaluate = module.checkpoints, module.environment_name, module.evaluate


class ShortPendulum(Environment):
  stats = {"return_min": -100.0, "return_max": 0.0}

  def get_dataset(self):
    raise NotImplementedError

  def get_env(self):
    return gym.make("Pendulum-v1", max_episode_steps=3)


def test_checkpoint_discovery_and_saved_environment(tmp_path):
  run = tmp_path / "arbitrary-run-name"
  best = run / "checkpoints" / "seed7" / "best"
  best.mkdir(parents=True)
  (best / "meta.json").write_text("{}")
  (run / "config.yaml").write_text("runs: [{model: SAC, env: HalfCheetah}]\n")
  assert checkpoints(tmp_path, "best") == [best]
  assert checkpoints(best, "final") == [best]
  assert checkpoints(tmp_path, "final") == []
  assert environment_name(best) == "HalfCheetah"


def test_evaluate_restores_checkpoint_and_pairs_modes(tmp_path, monkeypatch):
  family = ShortPendulum()
  env = family.get_env()
  try:
    policy = Policy.make("SAC", env, env_stats=family.get_env_stats(env), seed=0)
    save_policy(policy, tmp_path / "best")
  finally:
    env.close()
  monkeypatch.setattr(Environment, "make", lambda name: family)
  kwargs = dict(episodes=2, num_envs=2, num_candidates=1, seed=7)
  result = evaluate(tmp_path / "best", "short", ["default", "mean", "bon"], **kwargs)
  assert all(np.isfinite(row["mean_return"]) for row in result.values())
  assert result["default"] == result["mean"] == result["bon"]
  json.dumps(result, allow_nan=False)
