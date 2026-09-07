"""Compare Direct vs Mean-of-N execution on a sweep's already-trained checkpoints.

`barl_jax.execution.ExecutionSpec` makes action selection independent of the
policy that produced the candidates (see `execution.py`): "Direct" is
`mode='default'` (one sample, the policy's usual inference), "Mean-of-N" is
`mode='mean'` (N independent samples, averaged elementwise). This script loads
each seed's `best` checkpoint from a finished sweep run and re-evaluates it
under both, with no retraining and no policy-specific code -- the same
`run_episodes` used during training, wrapped so the execution mode is a
property of the wrapper rather than of the policy.

Only runs whose `objective.json` says `status: "finished"` are evaluated -- a
run still training, or one whose checkpoints never made it back from cluster
scratch, has nothing comparable to report. `--chunk-size` further filters to
one axis value of a chunk_size grid sweep (see `dogrun_chunk4_debug_sweep.yaml`).

    uv run python scripts/eval_execution_modes.py sweeps/2026-09-06_13-08-15 \\
        --chunk-size 1 --episodes 10 --num-candidates 8
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

_RUN_DIR_RE = re.compile(r"^(?P<model>[A-Za-z0-9]+)_.*chunk_size=(?P<chunk_size>\d+)$")


def _finished_runs(sweep_dir: Path, chunk_size: int):
  """(model, run_dir, objective) for every run directory whose tag matches
  chunk_size and whose objective.json reports status == "finished"."""
  runs = []
  for run_dir in sorted(p for p in sweep_dir.iterdir() if p.is_dir()):
    m = _RUN_DIR_RE.match(run_dir.name)
    if not m or int(m["chunk_size"]) != chunk_size:
      continue
    objective_path = run_dir / "objective.json"
    if not objective_path.exists():
      continue
    objective = json.loads(objective_path.read_text())
    if objective.get("status") != "finished":
      continue
    runs.append((m["model"], run_dir, objective))
  return runs


def _seed_checkpoints(run_dir: Path, num_seeds: int):
  out = []
  for seed in range(num_seeds):
    best = run_dir / "checkpoints" / f"seed{seed}" / "best"
    if best.exists():
      out.append((seed, best))
  return out


def _patch_missing_params(meta: dict) -> tuple[dict, list[str]]:
  """Backfill `params.<key>` and `components.<slot>.<key>` entries a snapshot
  predates with today's live default for that key, so a checkpoint saved
  before a param was added (e.g. `params.alpha_granularity`, `entropy.tune`
  -- both added after these sweep checkpoints) still passes
  `ConfigTree.resolve_params`/`resolve_slot`'s strict snapshot-completeness
  check. Only fills keys the snapshot lacks -- every key it does have, and
  every slot's chosen `class`, is untouched. Returns the patched meta (a
  copy) and the list of dotted keys filled."""
  import copy

  from barl_jax.config.tree import ConfigTree

  # `alpha_granularity` is a mode switch, not a plain additive knob: its
  # *docstring* says "per_chunk" is the default ("one alpha for the whole
  # chunked plan, exactly as before"), but the live `Config` declaration on
  # MaxEntRLPolicy is actually "per_action" -- seemingly a bug in that
  # in-progress work, not something to load-bearing-ly rely on here. Backfill
  # with "per_chunk" regardless of the live default: that is the value that
  # reproduces every pre-this-feature checkpoint's actual saved shapes (a
  # scalar log_alpha/target, not a (chunk_size,) one) -- using the live
  # default here would give AutotuneRegulator a differently-shaped log_alpha
  # than what was trained and restore would fail (or silently mis-restore).
  LEGACY_PARAM_DEFAULTS = {"alpha_granularity": "per_chunk"}

  meta = copy.deepcopy(meta)
  resolved = meta.setdefault("snapshot", {}).setdefault("resolved", {})
  live_cfg = ConfigTree()
  model_name, profile = meta["model"], meta.get("profile")
  filled = []

  def backfill(prefix: str, live: dict, overrides: dict = {}):
    replayed_keys = {k[len(prefix):] for k in resolved if k.startswith(prefix)}
    for key, value in live.items():
      if key not in replayed_keys:
        resolved[f"{prefix}{key}"] = overrides.get(key, value)
        filled.append(f"{prefix}{key}")

  backfill("params.", live_cfg.resolve_params(model_name, profile), LEGACY_PARAM_DEFAULTS)
  for slot in live_cfg.slots(model_name):
    _, live_slot = live_cfg.resolve_slot(model_name, slot, profile)
    backfill(f"components.{slot}.", live_slot)

  return meta, filled


def _load_policy_compat(path: Path, env, env_stats):
  """`checkpoints.load_policy`, but through `_patch_missing_params` first --
  this script only ever reads checkpoints, so patching the in-memory meta
  (never the file on disk) to tolerate a newer code's added params is safe
  where retraining code should instead be updated to migrate the snapshot."""
  import json
  from dataclasses import replace

  import orbax.checkpoint as ocp
  from flax import nnx

  from barl_jax.config.model import ModelSpec
  from barl_jax.config.tree import ConfigTree
  from barl_jax.policy import Policy

  path = path.absolute()
  meta = json.loads((path / "meta.json").read_text())
  meta, filled = _patch_missing_params(meta)
  if filled:
    print(f"  (backfilled params not yet in this checkpoint's snapshot: {filled})")
  config = ConfigTree(snapshot=meta.get("snapshot"), overrides=meta.get("overrides"))
  spec = replace(ModelSpec.resolve(meta["model"], config, meta.get("profile")),
                policy_class=meta["policy_type"])
  policy = Policy.make(spec.name, env, spec=spec, env_stats=env_stats, **meta.get("kwargs", {}))
  target = nnx.state(policy)
  with ocp.StandardCheckpointer() as ckpt:
    restored = ckpt.restore(path / "state", target)
  nnx.update(policy, restored)
  return policy


class _SpecBoundActor:
  """Adapts a restored Policy's `inference_fn(spec=...)` to the no-arg
  `actor.inference_fn()` that `evaluation.run_episodes` calls -- the execution
  mode lives on this wrapper, not on the policy or on run_episodes."""

  def __init__(self, policy, spec):
    self._policy = policy
    self._spec = spec
    self.chunk_size = policy.chunk_size
    self.action_dim = policy.action_dim

  def inference_fn(self):
    return self._policy.inference_fn(spec=self._spec)


def _episode_returns(runs: list[dict]) -> np.ndarray:
  return np.array([sum(r["rewards"]) for r in runs], dtype=np.float64)


def evaluate_checkpoint(ckpt_path: Path, single_env, vector_env, env_stats, *, seed: int,
                        episodes: int, num_candidates: int) -> dict:
  from barl_jax.evaluation import run_episodes
  from barl_jax.execution import ExecutionSpec, validate_execution_spec

  policy = _load_policy_compat(ckpt_path, single_env, env_stats=env_stats)
  out = {}
  for label, spec in (("direct", ExecutionSpec(mode="default")),
                      ("mean_of_n", ExecutionSpec(mode="mean", num_candidates=num_candidates))):
    problems = validate_execution_spec(spec, policy=policy)
    if problems:
      out[label] = {"error": "; ".join(problems)}
      continue
    actor = _SpecBoundActor(policy, spec)
    # Same eval_seed for both modes on this checkpoint: any difference between
    # them is then attributable to the execution mode, not to different resets.
    episode_data = run_episodes(actor, vector_env, episodes, render=False, seed=seed)
    returns = _episode_returns(episode_data)
    out[label] = {
      "mean_return": float(returns.mean()),
      "std_return": float(returns.std()),
      "success_rate": float(np.mean([r["success"] for r in episode_data])),
    }
  return out


def main():
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("sweep_dir", help="e.g. sweeps/2026-09-06_13-08-15")
  parser.add_argument("--chunk-size", type=int, default=1)
  parser.add_argument("--episodes", type=int, default=10, help="per seed, per mode")
  parser.add_argument("--num-candidates", type=int, default=8, help="Mean-of-N's N")
  parser.add_argument("--json", action="store_true")
  args = parser.parse_args()

  from barl_jax.env import Environment
  from barl_jax.seeding import derive_seed

  sweep_dir = Path(args.sweep_dir)
  runs = _finished_runs(sweep_dir, args.chunk_size)
  if not runs:
    raise SystemExit(f"no finished chunk_size={args.chunk_size} runs under {sweep_dir}")

  results = {}
  for model, run_dir, objective in runs:
    num_seeds = objective["num_seeds"]
    checkpoints = _seed_checkpoints(run_dir, num_seeds)
    if not checkpoints:
      print(f"{model}: finished but no checkpoints under {run_dir}/checkpoints -- skipping")
      continue

    environment = Environment.make(objective["env"])
    single_env = environment.get_env()
    env_stats = environment.get_env_stats(single_env)
    vector_env = environment.get_vector_env(min(args.episodes, 8))
    try:
      per_seed = {}
      for seed, ckpt_path in checkpoints:
        eval_seed = derive_seed(run_dir.name, seed, "eval_execution_modes")
        per_seed[seed] = evaluate_checkpoint(
          ckpt_path, single_env, vector_env, env_stats,
          seed=eval_seed, episodes=args.episodes, num_candidates=args.num_candidates)
        print(f"{model} seed{seed}: "
              f"direct={per_seed[seed]['direct'].get('mean_return'):.1f}  "
              f"mean_of_n={per_seed[seed]['mean_of_n'].get('mean_return'):.1f}")
    finally:
      vector_env.close()
      single_env.close()
    results[model] = {"env": objective["env"], "chunk_size": args.chunk_size,
                      "num_candidates": args.num_candidates, "per_seed": per_seed}

  if args.json:
    print(json.dumps(results, indent=2))
    return

  print(f"\n{'model':<8}{'direct':>12}{'mean_of_n':>12}{'delta':>10}")
  for model, r in results.items():
    direct = np.mean([s["direct"]["mean_return"] for s in r["per_seed"].values()
                      if "mean_return" in s["direct"]])
    mean_n = np.mean([s["mean_of_n"]["mean_return"] for s in r["per_seed"].values()
                      if "mean_return" in s["mean_of_n"]])
    print(f"{model:<8}{direct:>12.1f}{mean_n:>12.1f}{mean_n - direct:>+10.1f}")


if __name__ == "__main__":
  main()
