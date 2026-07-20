import argparse
import datetime
import itertools
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import yaml

"""
Launches a grid of {algo x env x seed} training runs as separate, truly
concurrent OS processes (each run gets its own Python interpreter and CUDA
context, unlike the in-process `runs:` series in scripts/online_learning.py
et al. -- that one runs a list sequentially in a single process; this one
fans a grid out in parallel).

  uv run python scripts/run_sweep.py \\
      --algos SACPolicy PPOPolicy \\
      --envs HalfCheetah CheetahRun \\
      --seeds 3

launches 2*2*3 = 12 runs (seeds 0, 1, 2 per algo/env pair), up to
--max_parallel at a time. Each run's stdout/stderr goes to its own log file
under sweeps/<timestamp>/ so concurrent output doesn't interleave.
"""

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(SCRIPT_DIR, "..", "config")


def run_one(script: str, model: str, env: str, seed: int, defaults: dict, sweep_dir: str) -> int:
  tag = f"{model}_{env}_seed{seed}"

  run_config_path = os.path.join(sweep_dir, f"{tag}.yaml")
  with open(run_config_path, "w") as f:
    yaml.safe_dump({"defaults": defaults, "runs": [{"model": model, "env": env, "seed": seed}]}, f)

  script_path = os.path.join(SCRIPT_DIR, f"{script}.py")
  log_path = os.path.join(sweep_dir, f"{tag}.log")

  print(f"[{tag}] starting -- log: {log_path}")
  with open(log_path, "w") as logf:
    proc = subprocess.run(
      [sys.executable, script_path, "--config", run_config_path],
      stdout=logf, stderr=subprocess.STDOUT,
    )

  status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
  print(f"[{tag}] {status}")
  return proc.returncode


if __name__ == "__main__":

  args = argparse.ArgumentParser(description="Launch a grid of {algo x env x seed} training runs in parallel.")
  args.add_argument("--algos", nargs="+", required=True, help="Policy names, e.g. SACPolicy PPOPolicy")
  args.add_argument("--envs", nargs="+", required=True, help="Environment names, e.g. HalfCheetah CheetahRun")
  args.add_argument("--seeds", type=int, default=1, help="Number of seeds per (algo, env) pair -- runs seed 0..seeds-1")
  args.add_argument("--script", default="online_learning",
                     choices=["online_learning", "offline_learning", "offline_to_online"],
                     help="Which training script to launch each run with")
  args.add_argument("--config", default=None,
                     help="Base config providing `defaults:` for every run (default: config/<script>.yaml)")
  args.add_argument("--max_parallel", type=int, default=8, help="Max concurrent runs (0 = unlimited)")

  kwargs = args.parse_args()

  base_config_path = kwargs.config or os.path.join(CONFIG_DIR, f"{kwargs.script}.yaml")
  with open(base_config_path) as f:
    defaults = (yaml.safe_load(f) or {}).get("defaults", {})

  combos = list(itertools.product(kwargs.algos, kwargs.envs, range(kwargs.seeds)))

  now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  sweep_dir = os.path.join(SCRIPT_DIR, "..", "sweeps", now)
  os.makedirs(sweep_dir, exist_ok=True)

  # One tag per sweep invocation, threaded through `defaults` so every run
  # in the grid picks it up and can be filtered together in wandb/the logger.
  sweep_tag = "sweep-" + datetime.datetime.now().strftime("%m-%d-%H-%M")
  defaults = {**defaults, "sweep_tag": sweep_tag}

  max_workers = kwargs.max_parallel if kwargs.max_parallel > 0 else len(combos)
  print(f"Launching {len(combos)} runs ({len(kwargs.algos)} algos x {len(kwargs.envs)} envs x {kwargs.seeds} seeds), "
        f"max_parallel={max_workers}, logs under {sweep_dir}")

  with ThreadPoolExecutor(max_workers=max_workers) as pool:
    futures = [pool.submit(run_one, kwargs.script, algo, env, seed, defaults, sweep_dir)
               for algo, env, seed in combos]
    results = [f.result() for f in futures]

  failed = sum(1 for r in results if r != 0)
  print(f"Done: {len(results) - failed}/{len(results)} succeeded"
        + (f", {failed} failed -- see logs in {sweep_dir}" if failed else ""))
