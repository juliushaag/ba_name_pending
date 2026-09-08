"""Evaluate saved policies from a checkpoint, run directory, or sweep.

    uv run python scripts/evaluate.py sweeps/<submission> --mode default mean
    uv run python scripts/evaluate.py <checkpoint> --env HalfCheetah --json
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import yaml


def checkpoints(path: Path, checkpoint: str) -> list[Path]:
  if (path / "meta.json").is_file():
    return [path]
  return sorted(p.parent for p in path.rglob("meta.json")
                if p.parent.name == checkpoint and p.parent.parent.name.startswith("seed"))


def environment_name(path: Path) -> str | None:
  for parent in path.parents:
    config = parent / "config.yaml"
    if config.is_file():
      data = yaml.safe_load(config.read_text()) or {}
      runs = data.get("runs", [])
      if len(runs) == 1:
        return runs[0].get("env", data.get("defaults", {}).get("env"))
  return None


class EvaluationActor:
  def __init__(self, policy, spec):
    self.policy, self.spec = policy, spec
    self.chunk_size, self.action_dim = policy.chunk_size, policy.action_dim

  def inference_fn(self):
    return self.policy.inference_fn(spec=self.spec)


def evaluate(path, env_name, modes, *, episodes, num_envs, num_candidates, seed):
  from barl_jax.checkpoints import load_policy
  from barl_jax.env import Environment
  from barl_jax.evaluation import run_episodes
  from barl_jax.execution import ExecutionSpec, validate_execution_spec

  environment = Environment.make(env_name)
  single = environment.get_env()
  try:
    policy = load_policy(path, single, env_stats=environment.get_env_stats(single))
    vector = environment.get_vector_env(min(episodes, num_envs))
    try:
      results = {}
      for mode in modes:
        spec = ExecutionSpec(mode, 1 if mode == "default" else num_candidates)
        problems = validate_execution_spec(spec, policy=policy)
        if problems:
          results[mode] = {"error": "; ".join(problems)}
          continue
        records = run_episodes(EvaluationActor(policy, spec), vector, episodes,
                               render=False, seed=seed)
        returns = np.asarray([sum(r["rewards"]) for r in records])
        results[mode] = {"mean_return": float(returns.mean()),
                         "std_return": float(returns.std()),
                         "success_rate": float(np.mean([r["success"] for r in records]))}
      return results
    finally:
      vector.close()
  finally:
    single.close()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("path", type=Path)
  parser.add_argument("--env", help="Environment name; defaults to the saved run config")
  parser.add_argument("--checkpoint", choices=("best", "final"), default="best")
  parser.add_argument("--mode", nargs="+", choices=("default", "mean", "bon"), default=["default"])
  parser.add_argument("--episodes", type=int, default=10)
  parser.add_argument("--num-envs", type=int, default=8)
  parser.add_argument("--num-candidates", type=int, default=8)
  parser.add_argument("--seed", type=int, default=0, help="Shared evaluation seed for paired comparisons")
  parser.add_argument("--json", action="store_true")
  args = parser.parse_args()
  if min(args.episodes, args.num_envs, args.num_candidates) < 1:
    parser.error("episodes, num-envs and num-candidates must be positive")
  paths = checkpoints(args.path.resolve(), args.checkpoint)
  if not paths:
    parser.error(f"no {args.checkpoint} policy checkpoints under {args.path}")
  results = {}
  for path in paths:
    try:
      env_name = args.env or environment_name(path)
      if not env_name:
        raise ValueError("no saved environment found; pass --env")
      results[str(path)] = evaluate(path, env_name, args.mode, episodes=args.episodes,
                                   num_envs=args.num_envs, num_candidates=args.num_candidates,
                                   seed=args.seed)
    except Exception as exc:
      results[str(path)] = {"error": f"{type(exc).__name__}: {exc}"}
  if args.json:
    print(json.dumps(results, indent=2, allow_nan=False))
  else:
    for path, modes in results.items():
      print(path)
      for mode, result in modes.items():
        print(f"  {mode}: {result}")
  if any("error" in modes or any("error" in row for row in modes.values())
         for modes in results.values()):
    raise SystemExit(1)


if __name__ == "__main__":
  main()
