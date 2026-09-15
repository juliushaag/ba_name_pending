"""Evaluate saved policies from a checkpoint, run directory, or sweep.

    uv run python scripts/evaluate.py sweeps/<submission> --mode DirectExecutor MeanExecutor
    uv run python scripts/evaluate.py <checkpoint> --env HalfCheetah --json
    uv run python scripts/evaluate.py --config experiments/eval.yaml

--config replays a whole batch of evaluations from one YAML file (see
experiments/eval.yaml for the schema) instead of the single path this CLI
otherwise takes -- every other flag but --output/--json is then read from
that file, not the command line."""
import argparse
import json
import os
import time
import traceback
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

# See train.py's identical guard: on a shared multi-GPU node,
# mujoco.egl's device enumeration ignores cgroup isolation and can silently
# land on a GPU this job can't actually render on, producing a black video
# with no error. Pin it to the node-wide index SLURM actually granted.
if "MUJOCO_EGL_DEVICE_ID" not in os.environ:
  slurm_gpus = os.environ.get("SLURM_STEP_GPUS", os.environ.get("SLURM_JOB_GPUS"))
  if slurm_gpus:
    os.environ["MUJOCO_EGL_DEVICE_ID"] = slurm_gpus.split(",")[0]

import numpy as np
import yaml


def _log_failure(label: str, exc: Exception) -> str:
  """Prints a one-line summary plus the full traceback (indented so it reads
  as attached to `label`, not blended into whatever prints next) and returns
  that same traceback for the caller's own `error`/`traceback` result fields
  -- one implementation shared by evaluate()'s per-mode, evaluate_entry()'s
  per-checkpoint and main()'s per-entry catch sites, rather than a one-line
  summary with no traceback anywhere but the JSON file."""
  tb = traceback.format_exc()
  print(f"{label}: FAILED -- {type(exc).__name__}: {exc}")
  print("\n".join(f"    {line}" for line in tb.rstrip().splitlines()))
  return tb


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


def evaluate(path, env_name, modes, *, episodes, num_envs, num_candidates, seed, top_k=2, sigma=0.1):
  import dataclasses

  from barl_jax.checkpoints import load_policy
  from barl_jax.env import Environment
  from barl_jax.evaluation import run_episodes
  from barl_jax.multiseed import SeedActor
  from barl_jax.registration import get_executor

  environment = Environment.make(env_name)
  single = environment.get_env()
  try:
    policy = load_policy(path, single, env_stats=environment.get_env_stats(single))
    vector = environment.get_vector_env(min(episodes, num_envs))
    try:
      results = {}
      print(f"  {path}  env={env_name}  {episodes} episode(s) x {len(modes)} mode(s) {modes}")
      for mode in modes:
        t0 = time.perf_counter()
        executor_cls = get_executor(mode)
        config_type = executor_cls.config_type
        # Only pass the knobs this executor's config actually declares --
        # DirectExecutorConfig has neither num_candidates/top_k/sigma, so no
        # per-mode `if mode == ...` branch is needed to keep it from
        # rejecting them.
        declared = {f.name for f in dataclasses.fields(config_type)}
        kwargs = {}
        if "num_candidates" in declared:
          kwargs["num_candidates"] = num_candidates
        if "top_k" in declared:
          kwargs["top_k"] = top_k
        if "sigma" in declared:
          kwargs["sigma"] = sigma
        executor = executor_cls(config_type(**kwargs))
        problems = executor.validate(policy=policy)
        if problems:
          results[mode] = {"error": "; ".join(problems)}
          print(f"    [{mode}] skipped -- {'; '.join(problems)}")
          continue
        # Caught per mode, not left to propagate: one mode dying (a NaN action
        # poisoning mj_step, an OOM at a large num_candidates, ...) must not
        # cost every mode that already finished for this checkpoint -- the
        # caller only ever sees whatever `evaluate()` returns, not what it
        # built up to before raising.
        try:
          records = run_episodes(SeedActor(policy, executor), vector, episodes,
                                 render=False, seed=seed)
          returns = np.asarray([sum(r["rewards"]) for r in records])
          results[mode] = {"mean_return": float(returns.mean()),
                           "std_return": float(returns.std()),
                           "success_rate": float(np.mean([r["success"] for r in records]))}
          print(f"    [{mode}] mean_return={results[mode]['mean_return']:.4g} "
                f"success_rate={results[mode]['success_rate']:.2f}  ({time.perf_counter() - t0:.1f}s)")
        except Exception as exc:
          tb = _log_failure(f"    [{mode}]", exc)
          results[mode] = {"error": f"{type(exc).__name__}: {exc}", "traceback": tb}
      return results
    finally:
      vector.close()
  finally:
    single.close()


# One source of truth for a run's optional keys, shared by argparse's own
# `default=`, a `--config` entry's fallback when it omits a key, and the
# `--config`/single-path mutual-exclusion check in main() -- see there.
DEFAULTS = dict(checkpoint="best", mode=["DirectExecutor"], episodes=10, num_envs=8,
                num_candidates=8, top_k=2, sigma=0.1, seed=0)


def load_eval_config(path) -> tuple[str | None, list[dict]]:
  """`--config`'s schema (see experiments/eval.yaml): `output` (a JSON path;
  `--output` on the command line wins if both are given), `defaults` (any
  key `evaluate_entry` reads, applied under every run that doesn't set its
  own), and `runs` (a list of per-run dicts; `path` is required, `env` is
  optional -- omitted, it is read from that checkpoint's own saved run
  config, exactly like the plain CLI's `--env`)."""
  data = yaml.safe_load(Path(path).read_text()) or {}
  runs = data.get("runs")
  if not runs:
    raise ValueError(f"{path}: no `runs` entries")
  defaults = data.get("defaults") or {}
  for i, run in enumerate(runs):
    if "path" not in run:
      raise ValueError(f"{path}: runs[{i}] has no `path`")
  return data.get("output"), [{**defaults, **run} for run in runs]


def evaluate_entry(entry: dict) -> dict:
  """One fully-resolved run (a `--config` entry merged with its `defaults:`,
  or the single path/flags the plain CLI was given) -> `{checkpoint_path:
  {mode: result}}` -- `entry["path"]` may itself be one policy checkpoint, a
  run directory, or a whole sweep, exactly like the plain CLI (`checkpoints()`
  discovers every seed's checkpoint under it)."""
  root = Path(entry["path"]).resolve()
  checkpoint = entry.get("checkpoint", DEFAULTS["checkpoint"])
  paths = checkpoints(root, checkpoint)
  if not paths:
    print(f"{root}: no {checkpoint} checkpoints found")
    return {str(root): {"error": f"no {checkpoint} policy checkpoints under {root}"}}
  print(f"{root}: {len(paths)} {checkpoint} checkpoint(s)")
  out = {}
  for path in paths:
    try:
      env_name = entry.get("env") or environment_name(path)
      if not env_name:
        raise ValueError("no saved environment found; set `env` for this run")
      out[str(path)] = evaluate(
        path, env_name, entry.get("mode", DEFAULTS["mode"]),
        episodes=entry.get("episodes", DEFAULTS["episodes"]),
        num_envs=entry.get("num_envs", DEFAULTS["num_envs"]),
        num_candidates=entry.get("num_candidates", DEFAULTS["num_candidates"]),
        top_k=entry.get("top_k", DEFAULTS["top_k"]), sigma=entry.get("sigma", DEFAULTS["sigma"]),
        seed=entry.get("seed", DEFAULTS["seed"]))
    except Exception as exc:
      tb = _log_failure(f"  {path}", exc)
      out[str(path)] = {"error": f"{type(exc).__name__}: {exc}", "traceback": tb}
  return out


def main():
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("path", type=Path, nargs="?", default=None,
                      help="Checkpoint/run/sweep path; omit when using --config")
  parser.add_argument("--config", help="Replay a batch of evaluations from a YAML file "
                                       "(see experiments/eval.yaml) instead of --path/flags")
  parser.add_argument("--output", help="Write the combined JSON results to this path "
                                       "(a --config file's own `output:` is the default)")
  parser.add_argument("--env", help="Environment name; defaults to the saved run config")
  parser.add_argument("--checkpoint", choices=("best", "final"), default=DEFAULTS["checkpoint"])
  parser.add_argument("--mode", nargs="+",
                      choices=("DirectExecutor", "MeanExecutor", "MedianExecutor", "BestOfNExecutor",
                               "TopKExecutor", "AxisShuffleExecutor", "PerturbExecutor"),
                      default=DEFAULTS["mode"])
  parser.add_argument("--episodes", type=int, default=DEFAULTS["episodes"])
  parser.add_argument("--num-envs", type=int, default=DEFAULTS["num_envs"])
  parser.add_argument("--num-candidates", type=int, default=DEFAULTS["num_candidates"])
  parser.add_argument("--top-k", type=int, default=DEFAULTS["top_k"], help="Only used by --mode TopKExecutor")
  parser.add_argument("--sigma", type=float, default=DEFAULTS["sigma"],
                      help="Only used by --mode PerturbExecutor")
  parser.add_argument("--seed", type=int, default=DEFAULTS["seed"],
                      help="Shared evaluation seed for paired comparisons")
  parser.add_argument("--json", action="store_true", help="Print the combined results as JSON too")
  args = parser.parse_args()

  if args.config:
    single_run_flags = (args.path is not None or args.env is not None
                        or any(getattr(args, key) != default for key, default in DEFAULTS.items()))
    if single_run_flags:
      parser.error("--config replays a whole eval config; only --output/--json may accompany it")
    output, entries = load_eval_config(args.config)
    output = args.output or output
  else:
    if args.path is None:
      parser.error("path is required unless --config is given")
    if min(args.episodes, args.num_envs, args.num_candidates) < 1:
      parser.error("episodes, num-envs and num-candidates must be positive")
    entries = [dict(path=str(args.path), env=args.env, checkpoint=args.checkpoint,
                    mode=args.mode, episodes=args.episodes, num_envs=args.num_envs,
                    num_candidates=args.num_candidates, top_k=args.top_k, sigma=args.sigma,
                    seed=args.seed)]
    output = args.output

  print(f"Evaluating {len(entries)} entr{'y' if len(entries) == 1 else 'ies'}"
       + (f" from {args.config}" if args.config else ""))
  t0 = time.perf_counter()
  results = {}
  # try/finally around the whole loop, not just each entry's own try/except
  # below: whatever crashes -- a bug in that per-entry handling itself, a
  # KeyboardInterrupt, anything else `evaluate_entry` doesn't already turn
  # into an `error` entry -- every prior entry's real results still get
  # written instead of the file simply never appearing.
  try:
    for i, entry in enumerate(entries):
      print(f"[{i + 1}/{len(entries)}]")
      try:
        results.update(evaluate_entry(entry))
      except Exception as exc:
        key = entry.get("path", f"entry[{i}]")
        tb = _log_failure(f"  {key}", exc)
        results[str(key)] = {"error": f"{type(exc).__name__}: {exc}", "traceback": tb}
  finally:
    print(f"Done in {time.perf_counter() - t0:.1f}s")
    if output:
      out = Path(output)
      out.parent.mkdir(parents=True, exist_ok=True)
      try:
        out.write_text(json.dumps(results, indent=2, allow_nan=False))
      except ValueError:
        # A legitimately-computed NaN/Inf return (a diverged policy) must not
        # cost every other entry's results -- write it anyway rather than
        # lose the whole file over one value.
        out.write_text(json.dumps(results, indent=2, allow_nan=True))
      print(f"Wrote {len(results)} checkpoint result(s) to {out}")

  if args.json:
    try:
      print(json.dumps(results, indent=2, allow_nan=False))
    except ValueError:
      print(json.dumps(results, indent=2, allow_nan=True))
  elif not output:
    for path, modes in results.items():
      print(path)
      for mode, result in modes.items():
        print(f"  {mode}: {result}")

  if any("error" in modes or any("error" in row for row in modes.values())
         for modes in results.values()):
    raise SystemExit(1)


if __name__ == "__main__":
  main()
