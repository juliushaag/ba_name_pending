"""Public evaluation CLI: discovery and real checkpoint reconstruction."""
import importlib.util
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

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
  # top_k=1 too: at num_candidates=1 every mode here is bit-identical to
  # default (execution.py's own invariant, see test_execution.py), so a real
  # checkpoint evaluated end-to-end through the CLI should agree exactly.
  kwargs = dict(episodes=2, num_envs=2, num_candidates=1, top_k=1, seed=7)
  modes = ["default", "mean", "median", "bon", "topk", "axis_shuffle", "perturb"]
  result = evaluate(tmp_path / "best", "short", modes, **kwargs)
  assert all(np.isfinite(row["mean_return"]) for row in result.values())
  assert len({tuple(result[m].items()) for m in modes}) == 1
  json.dumps(result, allow_nan=False)


# ---------------------------------------------------------------------------
# --config: load_eval_config's merge/validation, evaluate_entry, main() glue
# ---------------------------------------------------------------------------

def test_load_eval_config_merges_defaults_into_each_run(tmp_path):
  cfg = tmp_path / "eval.yaml"
  cfg.write_text("""
output: out.json
defaults:
  episodes: 5
  num_candidates: 3
runs:
  - path: a
    mode: [default, mean]
  - path: b
    episodes: 9
""")
  output, entries = module.load_eval_config(cfg)
  assert output == "out.json"
  assert entries == [
    {"episodes": 5, "num_candidates": 3, "path": "a", "mode": ["default", "mean"]},
    {"episodes": 9, "num_candidates": 3, "path": "b"},
  ]


@pytest.mark.parametrize("body, match", [
  ("output: out.json\n", "runs"),
  ("runs: [{env: HalfCheetah}]\n", "path"),
])
def test_load_eval_config_rejects_missing_runs_or_path(tmp_path, body, match):
  cfg = tmp_path / "eval.yaml"
  cfg.write_text(body)
  with pytest.raises(ValueError, match=match):
    module.load_eval_config(cfg)


def test_evaluate_entry_reads_env_from_saved_config_when_omitted(tmp_path, monkeypatch):
  run = tmp_path / "run"
  best = run / "checkpoints" / "seed0" / "best"
  family = ShortPendulum()
  env = family.get_env()
  try:
    policy = Policy.make("SAC", env, env_stats=family.get_env_stats(env), seed=0)
    save_policy(policy, best)
  finally:
    env.close()
  (run / "config.yaml").write_text("runs: [{model: SAC, env: short}]\n")
  monkeypatch.setattr(Environment, "make", lambda name: family)

  entry = dict(path=str(run), episodes=2, num_envs=2, num_candidates=1, seed=7)
  result = module.evaluate_entry(entry)
  assert list(result) == [str(best)]
  assert np.isfinite(result[str(best)]["default"]["mean_return"])


def test_evaluate_entry_reports_missing_checkpoints_without_raising(tmp_path):
  result = module.evaluate_entry(dict(path=str(tmp_path)))
  assert "error" in next(iter(result.values()))


def test_main_writes_config_results_to_output_json(tmp_path, monkeypatch, capsys):
  run = tmp_path / "run"
  best = run / "checkpoints" / "seed0" / "best"
  family = ShortPendulum()
  env = family.get_env()
  try:
    policy = Policy.make("SAC", env, env_stats=family.get_env_stats(env), seed=0)
    save_policy(policy, best)
  finally:
    env.close()
  monkeypatch.setattr(Environment, "make", lambda name: family)

  cfg = tmp_path / "eval.yaml"
  out = tmp_path / "out" / "results.json"
  cfg.write_text(f"runs:\n  - path: {run}\n    env: short\n    episodes: 2\n"
                 f"    num_envs: 2\n    num_candidates: 1\n")
  monkeypatch.setattr("sys.argv", ["evaluate.py", "--config", str(cfg), "--output", str(out)])
  module.main()

  written = json.loads(out.read_text())
  assert np.isfinite(written[str(best)]["default"]["mean_return"])
  assert "Wrote 1 checkpoint result(s)" in capsys.readouterr().out


def test_main_rejects_single_run_flags_alongside_config(monkeypatch):
  monkeypatch.setattr("sys.argv", ["evaluate.py", "--config", "whatever.yaml", "--env", "HalfCheetah"])
  with pytest.raises(SystemExit):
    module.main()


# ---------------------------------------------------------------------------
# Crash handling: a failure must never cost results that already finished.
# ---------------------------------------------------------------------------

def test_evaluate_isolates_a_crashing_mode_and_keeps_the_rest(tmp_path, monkeypatch):
  """One mode raising inside run_episodes (a real crash, not an
  ExecutionSpec-rejected mode -- see the 'skipped' path) must not lose a
  different mode's already-computed result, and the failure must carry a
  traceback, not just a one-line message."""
  best = tmp_path / "best"
  family = ShortPendulum()
  env = family.get_env()
  try:
    policy = Policy.make("SAC", env, env_stats=family.get_env_stats(env), seed=0)
    save_policy(policy, best)
  finally:
    env.close()
  monkeypatch.setattr(Environment, "make", lambda name: family)

  import barl_jax.evaluation as evaluation_module
  real_run_episodes = evaluation_module.run_episodes

  def flaky_run_episodes(actor, vector, episodes, *, render, seed):
    if actor.spec.mode == "mean":
      raise RuntimeError("boom")
    return real_run_episodes(actor, vector, episodes, render=render, seed=seed)

  monkeypatch.setattr(evaluation_module, "run_episodes", flaky_run_episodes)

  result = module.evaluate(best, "short", ["default", "mean"],
                           episodes=2, num_envs=2, num_candidates=1, seed=7)
  assert np.isfinite(result["default"]["mean_return"])
  assert result["mean"]["error"] == "RuntimeError: boom"
  assert "Traceback" in result["mean"]["traceback"]
  assert "boom" in result["mean"]["traceback"]


def test_main_writes_partial_results_when_one_entry_crashes(tmp_path, monkeypatch):
  """A second entry that raises before evaluate_entry's own per-checkpoint
  try/except (a malformed `path`, here) must not cost the first entry's real,
  already-computed result -- and the file must still get written at all."""
  run = tmp_path / "run"
  best = run / "checkpoints" / "seed0" / "best"
  family = ShortPendulum()
  env = family.get_env()
  try:
    policy = Policy.make("SAC", env, env_stats=family.get_env_stats(env), seed=0)
    save_policy(policy, best)
  finally:
    env.close()
  monkeypatch.setattr(Environment, "make", lambda name: family)

  cfg = tmp_path / "eval.yaml"
  out = tmp_path / "out" / "results.json"
  cfg.write_text(f"runs:\n  - path: {run}\n    env: short\n    episodes: 2\n"
                 f"    num_envs: 2\n    num_candidates: 1\n  - path: null\n")
  monkeypatch.setattr("sys.argv", ["evaluate.py", "--config", str(cfg), "--output", str(out)])
  with pytest.raises(SystemExit):
    module.main()

  written = json.loads(out.read_text())
  assert np.isfinite(written[str(best)]["default"]["mean_return"])
  crashed = [row for row in written.values() if "error" in row]
  assert len(crashed) == 1
  assert "Traceback" in crashed[0]["traceback"]


def test_main_writes_nan_results_instead_of_losing_the_file(tmp_path, monkeypatch):
  monkeypatch.setattr(module, "evaluate_entry",
                      lambda entry: {"x": {"default": {"mean_return": float("nan")}}})
  cfg = tmp_path / "eval.yaml"
  out = tmp_path / "out.json"
  cfg.write_text("runs:\n  - path: whatever\n")
  monkeypatch.setattr("sys.argv", ["evaluate.py", "--config", str(cfg), "--output", str(out)])
  module.main()
  # NaN is not valid JSON, but Python's json module writes it as a bare
  # literal rather than the strict write silently losing the whole file.
  assert "NaN" in out.read_text()
