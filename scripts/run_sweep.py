import argparse
import datetime
import itertools
import os
import shlex
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import yaml

# Launches a grid of {algo x env x seed} training runs as separate, truly
# concurrent OS processes (own interpreter + CUDA context each), unlike
# scripts/online_learning.py's in-process sequential `runs:` list.

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(SCRIPT_DIR, "..", "config")


def _resolve_model_snapshot(model_name: str) -> dict:
  """Self-contained mirror of barl.model_config.snapshot_for() -- duplicated
  (not imported) so this lightweight subprocess-launcher script never has to
  import the `barl` package (and therefore torch/mujoco) just to read a YAML
  file. Embedded into each run's per-run YAML under `models:` so the run
  stays exactly recreatable even if config/models.yaml changes later."""
  models_path = os.path.join(CONFIG_DIR, "models.yaml")
  with open(models_path) as f:
    models = yaml.safe_load(f) or {}

  out: dict = {}
  alias = models.get("aliases", {}).get(model_name)
  policy_name = alias["policy"] if alias else model_name
  if alias:
    out["aliases"] = {model_name: alias}
  if policy_name in models:
    out[policy_name] = models[policy_name]
  return out


def detect_gpus() -> list[str]:
  """Best-effort GPU index detection via nvidia-smi, without depending on torch. Returns [] if
  nvidia-smi isn't available/fails, meaning: don't restrict CUDA_VISIBLE_DEVICES at all."""
  try:
    out = subprocess.run(
      ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
      capture_output=True, text=True, check=True,
    )
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]
  except (OSError, subprocess.CalledProcessError):
    return []


def run_one(script: str, model: str, env: str, seed: int, defaults: dict, sweep_dir: str,
            gpu_id: str | None = None, stream: bool = False) -> int:
  tag = f"{model}_{env}_seed{seed}"

  run_config_path = os.path.join(sweep_dir, f"{tag}.yaml")
  with open(run_config_path, "w") as f:
    yaml.safe_dump({
      "defaults": defaults,
      "runs": [{"model": model, "env": env, "seed": seed}],
      "models": _resolve_model_snapshot(model),
    }, f)

  script_path = os.path.join(SCRIPT_DIR, f"{script}.py")
  log_path = os.path.join(sweep_dir, f"{tag}.log")
  cmd = [sys.executable, script_path, "--config", run_config_path]

  env_vars = os.environ.copy()
  if gpu_id is not None:
    env_vars["CUDA_VISIBLE_DEVICES"] = gpu_id
    tag = f"{tag}_gpu{gpu_id}"

  print(f"[{tag}] starting -- log: {log_path}\n[{tag}] $ {shlex.join(cmd)}")
  with open(log_path, "w") as logf:
    if stream:
      # --validate runs: tee each line (tag-prefixed) to the terminal as well as the log file.
      proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env_vars,
      )
      for line in proc.stdout:
        print(f"[{tag}] {line}", end="")
        logf.write(line)
      proc.wait()
    else:
      proc = subprocess.run(
        cmd, stdout=logf, stderr=subprocess.STDOUT, env=env_vars,
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
  args.add_argument("--max_parallel", type=int, default=3,
                     help="Max concurrent runs per GPU (0 = unlimited per GPU). Total concurrency "
                          "is this times the number of GPUs in --gpus / detected via nvidia-smi.")
  args.add_argument("--gpus", nargs="+", default=None,
                     help="GPU indices to spread runs across, e.g. --gpus 0 1 2 3 -- each run gets "
                          "CUDA_VISIBLE_DEVICES set to exactly one of these. Default: autodetect "
                          "via nvidia-smi; if that finds none, CUDA_VISIBLE_DEVICES is left untouched.")
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

  gpus = kwargs.gpus if kwargs.gpus is not None else detect_gpus()
  per_gpu_limit = kwargs.max_parallel if kwargs.max_parallel > 0 else None

  if gpus:
    semaphores = {g: threading.Semaphore(per_gpu_limit) for g in gpus} if per_gpu_limit else None
    max_workers = (per_gpu_limit * len(gpus)) if per_gpu_limit else len(combos)
    gpu_desc = f", gpus={gpus}, max_parallel_per_gpu={per_gpu_limit or 'unlimited'}"
  else:
    semaphores = None
    max_workers = per_gpu_limit or len(combos)
    gpu_desc = ""

  desc = (f"{len(combos)} validate run(s) ({len(kwargs.algos)} algos x {len(kwargs.envs)} envs)" if kwargs.validate
          else f"{len(combos)} runs ({len(kwargs.algos)} algos x {len(kwargs.envs)} envs x {kwargs.seeds} seeds)")
  print(f"Launching {desc}, max_parallel={max_workers}{gpu_desc}, logs under {sweep_dir}")

  def run_one_gated(script, model, env, seed, defaults, sweep_dir, gpu_id, stream):
    sem = semaphores[gpu_id] if semaphores else None
    if sem is not None:
      sem.acquire()
    try:
      return run_one(script, model, env, seed, defaults, sweep_dir, gpu_id=gpu_id, stream=stream)
    finally:
      if sem is not None:
        sem.release()

  with ThreadPoolExecutor(max_workers=max_workers) as pool:
    futures = [
      pool.submit(run_one_gated, kwargs.script, algo, env, seed, defaults, sweep_dir,
                  gpus[i % len(gpus)] if gpus else None, kwargs.validate)
      for i, (seed, env, algo) in enumerate(combos)
    ]
    results = [f.result() for f in futures]

  failed = sum(1 for r in results if r != 0)
  print(f"Done: {len(results) - failed}/{len(results)} succeeded"
        + (f", {failed} failed -- see logs in {sweep_dir}" if failed else ""))
