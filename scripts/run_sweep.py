import argparse
import datetime
import itertools
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import yaml

# Launches a grid of {algo x env x seed} training runs as separate, truly
# concurrent OS processes (own interpreter + CUDA context each), unlike
# scripts/online_learning.py's in-process sequential `runs:` list.

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(SCRIPT_DIR, "..", "config")


def run_one(script: str, model: str, env: str, seed: int, defaults: dict, sweep_dir: str, stream: bool = False) -> int:
  tag = f"{model}_{env}_seed{seed}"

  run_config_path = os.path.join(sweep_dir, f"{tag}.yaml")
  with open(run_config_path, "w") as f:
    yaml.safe_dump({"defaults": defaults, "runs": [{"model": model, "env": env, "seed": seed}]}, f)

  script_path = os.path.join(SCRIPT_DIR, f"{script}.py")
  log_path = os.path.join(sweep_dir, f"{tag}.log")
  cmd = [sys.executable, script_path, "--config", run_config_path]

  print(f"[{tag}] starting -- log: {log_path}\n[{tag}] $ {shlex.join(cmd)}")
  with open(log_path, "w") as logf:
    if stream:
      # --validate runs: tee each line (tag-prefixed) to the terminal as well as the log file.
      proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
      )
      for line in proc.stdout:
        print(f"[{tag}] {line}", end="")
        logf.write(line)
      proc.wait()
    else:
      proc = subprocess.run(
        cmd, stdout=logf, stderr=subprocess.STDOUT,
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
  args.add_argument("--max_parallel", type=int, default=3, help="Max concurrent runs (0 = unlimited)")
  args.add_argument("--validate", action="store_true",
                     help="Smoke-test every {algo x env} combination instead of running a full "
                          "sweep: one rollout chunk + one train_step + optimizer step each, no "
                          "eval/checkpointing/wandb logging (see online_learning.validate_run). "
                          "Ignores --seeds -- exactly one run per combination.")

  kwargs = args.parse_args()

  base_config_path = kwargs.config or os.path.join(CONFIG_DIR, f"{kwargs.script}.yaml")
  with open(base_config_path) as f:
    defaults = (yaml.safe_load(f) or {}).get("defaults", {})

  if kwargs.validate:
    defaults = {**defaults, "validate": True}
    combos = list(itertools.product([0], kwargs.envs, kwargs.algos))
  else:
    combos = list(itertools.product(range(kwargs.seeds), kwargs.envs, kwargs.algos))

  now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  sweep_dir = os.path.join(SCRIPT_DIR, "..", "sweeps", now)
  os.makedirs(sweep_dir, exist_ok=True)

  # Threaded through `defaults` so every run in the grid can be filtered together in wandb.
  sweep_tag = "sweep-" + datetime.datetime.now().strftime("%m-%d-%H-%M")
  defaults = {**defaults, "sweep_tag": sweep_tag}

  max_workers = kwargs.max_parallel if kwargs.max_parallel > 0 else len(combos)
  desc = (f"{len(combos)} validate run(s) ({len(kwargs.algos)} algos x {len(kwargs.envs)} envs)" if kwargs.validate
          else f"{len(combos)} runs ({len(kwargs.algos)} algos x {len(kwargs.envs)} envs x {kwargs.seeds} seeds)")
  print(f"Launching {desc}, max_parallel={max_workers}, logs under {sweep_dir}")

  with ThreadPoolExecutor(max_workers=max_workers) as pool:
    futures = [pool.submit(run_one, kwargs.script, algo, env, seed, defaults, sweep_dir, kwargs.validate)
               for seed, env, algo  in combos]
    results = [f.result() for f in futures]

  failed = sum(1 for r in results if r != 0)
  print(f"Done: {len(results) - failed}/{len(results)} succeeded"
        + (f", {failed} failed -- see logs in {sweep_dir}" if failed else ""))
