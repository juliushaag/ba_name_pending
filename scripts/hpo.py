#!/usr/bin/env python3

"""Hyperparameter search over the existing experiment pipeline.

A search trial is structurally the same thing as one grid combo, so this script
samples the axes (scripts/search.py) and hands the results to the *same*
materialize / check / executor stack submit.py uses -- which knows nothing about
search, and does not need to. What a search adds over a grid is only the
generator (sampled points instead of a cross product) and a results channel to
drive the next round with; both live here, beside the pipeline rather than in
it.

    uv run python scripts/hpo.py --search src/barl_jax/defaults/search/demo.yaml --n-trials 20
    uv run python scripts/hpo.py --search src/barl_jax/defaults/search/demo.yaml --executor local --seeds 1
    uv run python scripts/hpo.py --report sweeps/<timestamp>

Batch by default: `n_trials` are sampled, materialized as one fully-expanded
sweep (each trial's own resolved snapshot, the same config-check and the same
slurm/tmux/local executors), and submitted once -- which keeps the
"fully expanded before anything is submitted" invariant, and means a search
sweep is indistinguishable on the cluster from a grid sweep of the same size.

`--report <dir>` is the readback half: every run's trainer now writes
`objective.json` beside its config (see barl_jax.trainer._write_objective), so
a finished sweep can be ranked locally without touching wandb. That file is
also what an *adaptive* driver -- successive-halving / TPE over waves, the one
mode this script deliberately does not build -- would poll to decide the next
wave. Random search over one expanded batch is the honest baseline; an Optuna
or ASHA driver would slot in by replacing search.sample_trial and re-running
the materialize/submit below per wave.
"""

import argparse
import datetime
import json
import os
import random
import re
from pathlib import Path

import grid
import yaml
from executors import EXECUTORS, SWEEP_NAME_FORMAT, SWEEPS_DIR, resolve
from search import check_searchability, load_search, sample_trials


def trial_tag(trial: dict, index: int, num_seeds: int) -> str:
  """`algo_env[_Nseeds]_tNNN`, fs-safe -- search runs are distinguished by trial
  index rather than by every varied value, which for a sampled point is all of
  them."""
  parts = [str(trial["algo"]), str(trial["env"])]
  parts.append(f"{num_seeds}seeds")
  parts.append(f"t{index:03d}")
  return re.sub(r"[^A-Za-z0-9.=_-]", "-", "_".join(parts))


def report(sweep_dir: str) -> None:
  """Rank a finished (or running) sweep by each run's objective.json."""
  root = Path(sweep_dir)
  if not root.is_dir():
    raise SystemExit(f"{root}: not a directory")
  rows = []
  for run_dir in sorted(root.iterdir()):
    obj_path = run_dir / "objective.json"
    if not obj_path.exists():
      continue
    with open(obj_path) as f:
      data = json.load(f)
    rows.append((run_dir.name, data))
  if not rows:
    print(f"No objective.json found under {root} -- runs write it into their own "
          f"directory on every eval (see trainer._write_objective).")
    return
  # Non-finite objectives (no eval yet, or a crash before one) rank last.
  rows.sort(key=lambda r: r[1]["objective"], reverse=True)
  print(f"{'rank':>4}  {'objective':>10}  {'status':<9}  tag")
  for i, (tag, data) in enumerate(rows):
    obj = data["objective"]
    obj_s = f"{obj:10.3f}" if obj != float("-inf") else f"{'-inf':>10}"
    print(f"{i + 1:>4}  {obj_s}  {data['status']:<9}  {tag}")


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("--search", default=None,
                  help="path to a search spec YAML (src/barl_jax/defaults/search/*.yaml)")
  ap.add_argument("--n-trials", type=int, default=None,
                  help="trials to sample (default: the spec's n_trials)")
  ap.add_argument("--seeds", type=int, default=None,
                  help="seeds per trial, vmapped like --seeds under submit.py "
                       "(default: config/run/defaults.yaml's num_seeds)")
  ap.add_argument("--executor", choices=("auto", *EXECUTORS), default="auto")
  ap.add_argument("--max-parallel", type=int, default=None)
  ap.add_argument("--sweep-dir", default=None,
                  help="where this search's sweep tree goes (default: sweeps/<timestamp>)")
  ap.add_argument("--report", default=None, metavar="SWEEP_DIR",
                  help="rank a finished sweep by objective.json and exit")
  ap.add_argument("--dry-run", action="store_true",
                  help="print the sampled trials and exit without writing or submitting")
  ap.add_argument("--skip-check", action="store_true")
  ap.add_argument("--validate", action="store_true",
                  help="smoke-test every distinct {algo x env} instead of training")
  args = ap.parse_args()

  if args.report:
    report(args.report)
    return
  if not args.search:
    ap.error("--search is required unless --report is given")

  spec = load_search(args.search)
  searchable = check_searchability(spec)
  if searchable:
    print("\nRefusing to run -- a search axis targets a non-search Config, not a HParam:")
    for problem in searchable:
      print(f"  - {problem}")
    print("\n  A search is over explicit optimization knobs (HParam). Config keys "
          "(activation, norm, dtype, tier, lr, ...) are varied with --set or a grid axis, "
          "not per-trial.")
    raise SystemExit(1)

  n_trials = args.n_trials or spec.get("n_trials")
  if not n_trials:
    ap.error("spec has no n_trials and --n-trials was not given")

  rng = random.Random(int(spec.get("sampler_seed", 0)))
  trials = sample_trials(spec["axes"], n_trials, rng)
  if not trials:
    raise SystemExit("search space produced no trials")

  num_seeds = args.seeds or spec.get("seeds_per_trial") or grid.default_num_seeds()

  print()
  print(f"Search:  {Path(args.search).name} ({len(trials)} trials, "
        f"{num_seeds} seed{'s' if num_seeds != 1 else ''} each)")
  print("Space:   " + ", ".join(f"{k}:{v['type']}" for k, v in spec["axes"].items()))
  if args.validate:
    print("Validate: one rollout chunk + one train step per distinct {algo x env}")
  print()
  for index, trial in enumerate(trials):
    shown = "  ".join(f"{k}={v}" for k, v in trial.items())
    print(f"  [{index:03d}] {shown}")

  if args.dry_run:
    print("\nDry run: nothing written, nothing submitted.")
    return

  if not args.skip_check:
    from submit import check_config
    print("\nChecking model configs ...", flush=True)
    problems = check_config(sorted({t["algo"] for t in trials}),
                            sorted({t["env"] for t in trials}))
    if problems:
      print("\nRefusing to submit -- unusable config:")
      for problem in problems:
        print(f"  {problem}")
      raise SystemExit(1)
    print("  OK")

  base_config_path = spec.get("base") or grid.DEFAULT_BASE_CONFIG
  with open(base_config_path) as f:
    defaults = dict((yaml.safe_load(f) or {}).get("defaults") or {})
  if args.validate:
    defaults["validate"] = True

  sweep_name = datetime.datetime.now().strftime(SWEEP_NAME_FORMAT)
  sweep_dir = Path(args.sweep_dir or (SWEEPS_DIR / sweep_name)).absolute()

  tags = [trial_tag(t, i, num_seeds) for i, t in enumerate(trials)]
  defaults["sweep_tag"] = sweep_name
  sweep_dir.mkdir(parents=True, exist_ok=True)
  runs = grid.materialize(trials, tags, defaults=defaults, sweep_dir=str(sweep_dir),
                          num_seeds=args.seeds or spec.get("seeds_per_trial"))

  head = {
    "sweep_tag": sweep_name, "created": sweep_name, "command": grid.sweep_command(),
    "kind": "search", "search": str(Path(args.search).resolve()),
    "sampler": spec.get("sampler", "random"), "n_trials": len(trials),
    "seeds": num_seeds, "objective": spec.get("objective", "eval/avg_cum_reward"),
    "base_config": os.path.abspath(base_config_path), "executor": resolve(args.executor).name,
  }
  grid.write_sweep_yaml(str(sweep_dir), head, runs)

  executor = resolve(args.executor)
  max_parallel = args.max_parallel or executor.default_parallelism
  print(f"\nDir:      {sweep_dir}")
  print(f"Executor: {executor.name} (parallel {max_parallel})")
  exec_report = executor.submit(runs, sweep_name, sweep_dir,
                                max_parallel=max_parallel, debug=False,
                                stream=args.validate)
  if executor.blocking:
    grid.write_sweep_yaml(str(sweep_dir), head, runs)
  print()
  width = max(len(label) for label in exec_report)
  for label, value in exec_report.items():
    print(f"  {label + ':':<{width + 1}} {value}")
  if executor.blocking and any(r.exit_code for r in runs):
    raise SystemExit(1)


if __name__ == "__main__":
  main()
