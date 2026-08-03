import argparse
import datetime
import itertools
import os
import re
import shlex
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import yaml

# Launches a grid of training runs as separate, truly concurrent OS processes
# (own interpreter + CUDA context each), unlike scripts/online_learning.py's
# in-process sequential `runs:` list. The grid is a cross product over named
# axes: `algo`/`env`/`seed`, plus any number of `model.<dotted>` axes (which
# become that run's model_overrides, see barl.model_config.set_overrides) and
# `run.<key>` axes (which override the run config's own keys). Nothing has to
# be edited in place to sweep a tunable.

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(SCRIPT_DIR, "..", "config")

RESERVED_AXES = ("algo", "env", "seed")


def _resolve_model_snapshot(model_name: str, overrides: dict | None = None) -> dict:
  """Self-contained mirror of barl.model_config.snapshot_for() -- duplicated
  (not imported) so this lightweight subprocess-launcher script never has to
  import the `barl` package (and therefore torch/mujoco) just to read a YAML
  file. Embedded into each run's per-run YAML under `models:` so the run
  stays exactly recreatable even if config/models.yaml or
  config/components.yaml changes later."""
  with open(os.path.join(CONFIG_DIR, "models.yaml")) as f:
    models = yaml.safe_load(f) or {}
  with open(os.path.join(CONFIG_DIR, "components.yaml")) as f:
    components = yaml.safe_load(f) or {}

  out: dict = {}
  alias = models.get("aliases", {}).get(model_name)
  policy_name = alias["policy"] if alias else model_name
  if alias:
    out["aliases"] = {model_name: alias}
  if policy_name in models:
    out[policy_name] = models[policy_name]

  # Baselines for exactly the component classes this run uses -- the ones this
  # policy's slots name, plus any an override swaps in. Without them the
  # snapshot would resolve against whatever components.yaml says at replay
  # time (see model_config.COMPONENT_DEFAULTS_KEY).
  named = [slot.get("class")
           for slot in (models.get(policy_name) or {}).get("components", {}).values()]
  named += [v for k, v in (overrides or {}).items() if k.endswith(".class")]
  used = {cls_name: components[cls_name] for cls_name in named if cls_name in components}
  if used:
    out["component_defaults"] = used

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


def build_axes(kwargs) -> tuple[str | None, dict, list[dict]]:
  """The sweep's axes from either a --grid file or the --algos/--envs/--seeds
  shorthand, plus any --set values as single-valued axes. Returns
  (base_config_path, axes, exclude)."""
  base, axes, exclude = None, {}, []

  if kwargs.grid:
    with open(kwargs.grid) as f:
      grid = yaml.safe_load(f) or {}
    base = grid.get("base")
    axes = dict(grid.get("axes") or {})
    exclude = list(grid.get("exclude") or [])

  if kwargs.algos:
    axes["algo"] = kwargs.algos
  if kwargs.envs:
    axes["env"] = kwargs.envs
  if kwargs.seeds is not None:
    axes["seed"] = list(i + 42 for i in range(kwargs.seeds))

  for assignment in kwargs.set or []:
    key, sep, value = assignment.partition("=")
    if not sep:
      raise SystemExit(f"--set {assignment!r} is not of the form key=value")
    axes[key.strip()] = [yaml.safe_load(value)]

  for name, values in axes.items():
    if not isinstance(values, list) or not values:
      raise SystemExit(f"axis {name!r} must be a non-empty list of values, got {values!r}")
    if name not in RESERVED_AXES and not name.startswith(("model.", "run.")):
      raise SystemExit(f"axis {name!r} must be one of {RESERVED_AXES}, or be prefixed "
                       f"'model.' (a dotted model override) or 'run.' (a run-config key)")
    if name in ("run.model", "run.env", "run.seed"):
      raise SystemExit(f"axis {name!r} would fight the {name.partition('.')[2]!r} the run entry "
                       f"already gets -- use the {name.partition('.')[2].replace('model', 'algo')!r} "
                       f"axis instead")

  for required in ("algo", "env"):
    if not axes.get(required):
      raise SystemExit(f"no {required!r} axis -- pass --{required}s or declare one in --grid")
  axes.setdefault("seed", [0])

  # Checked only once the CLI axes are in, since an exclude may well name an
  # axis the grid file leaves to --algos/--envs/--seeds.
  unknown = [k for ex in exclude for k in ex if k not in axes]
  if unknown:
    raise SystemExit(f"{kwargs.grid}: exclude names non-axis key(s) {unknown} -- it would never "
                     f"match anything. Axes are: {', '.join(axes)}")

  return base, axes, exclude


def build_combos(axes: dict, exclude: list[dict]) -> list[dict]:
  combos = [dict(zip(axes, values)) for values in itertools.product(*axes.values())]
  return [c for c in combos
          if not any(all(c.get(k) == v for k, v in ex.items()) for ex in exclude)]


def name_combos(combos: list[dict], axes: dict) -> list[str]:
  """One filesystem-safe tag per combo: algo_env_seedN, plus key=value for
  every *varying* non-reserved axis (a fixed --set value is the same in every
  run, so it only clutters the name -- it's recorded in the per-run YAML)."""
  varying = [k for k, v in axes.items() if k not in RESERVED_AXES and len(v) > 1]

  short = {k: k.rpartition(".")[2] for k in varying}
  for k, s in list(short.items()):
    if list(short.values()).count(s) > 1:
      short[k] = k.partition(".")[2].replace(".", "-")

  tags = []
  for combo in combos:
    parts = [str(combo["algo"]), str(combo["env"]), f"seed{combo['seed']}"]
    parts += [f"{short[k]}={combo[k]}" for k in varying]
    tags.append(re.sub(r"[^A-Za-z0-9.=_-]", "-", "_".join(parts)))
  return tags


def run_one(script: str, combo: dict, tag: str, defaults: dict, env_defaults: dict, sweep_dir: str,
            gpu_id: str | None = None, stream: bool = False) -> int:
  model, env = combo["algo"], combo["env"]

  run_entry = {"model": model, "env": env, "seed": combo["seed"]}
  run_entry.update({k.partition(".")[2]: v for k, v in combo.items() if k.startswith("run.")})

  overrides = {k.partition(".")[2]: v for k, v in combo.items() if k.startswith("model.")}
  if overrides:
    run_entry["model_overrides"] = overrides

  run_config = {"defaults": defaults, "runs": [run_entry],
                "models": _resolve_model_snapshot(model, overrides)}
  if env_defaults:
    run_config["env_defaults"] = env_defaults

  run_config_path = os.path.join(sweep_dir, f"{tag}.yaml")
  with open(run_config_path, "w") as f:
    yaml.safe_dump(run_config, f)

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

  args = argparse.ArgumentParser(
    description="Launch a grid of training runs in parallel, one OS process each.",
    epilog="Axes: algo/env/seed, plus model.<dotted path> (a model override, e.g. "
           "model.components.critic.rho) and run.<key> (a run-config key, e.g. run.utd_ratio).")
  args.add_argument("--grid", default=None,
                     help="YAML sweep spec: {base: <run config>, axes: {name: [values]}, "
                          "exclude: [{axis: value, ...}]}. Combined with any --algos/--envs/"
                          "--seeds/--set given, which win per axis.")
  args.add_argument("--algos", nargs="+", help="`algo` axis, e.g. SACPolicy PPOPolicy")
  args.add_argument("--envs", nargs="+", help="`env` axis, e.g. HalfCheetah CheetahRun")
  args.add_argument("--seeds", type=int, default=None,
                     help="`seed` axis as a count -- runs seed 0..seeds-1 (default: seed 0 only)")
  args.add_argument("--set", action="append", metavar="AXIS=VALUE",
                     help="Pin one axis to a single value across the whole sweep, e.g. "
                          "--set model.components.critic.rho=0.0 --set run.batch_size=512. "
                          "Repeatable.")
  args.add_argument("--script", default="online_learning",
                     choices=["online_learning", "offline_learning", "offline_to_online"],
                     help="Which training script to launch each run with")
  args.add_argument("--config", default=None,
                     help="Base config providing `defaults:`/`env_defaults:` for every run "
                          "(default: --grid's `base:`, else config/<script>.yaml)")
  args.add_argument("--max_parallel", type=int, default=3,
                     help="Max concurrent runs per GPU (0 = unlimited per GPU). Total concurrency "
                          "is this times the number of GPUs in --gpus / detected via nvidia-smi.")
  args.add_argument("--gpus", nargs="+", default=None,
                     help="GPU indices to spread runs across, e.g. --gpus 0 1 2 3 -- each run gets "
                          "CUDA_VISIBLE_DEVICES set to exactly one of these. Default: autodetect "
                          "via nvidia-smi; if that finds none, CUDA_VISIBLE_DEVICES is left untouched.")
  args.add_argument("--dry_run", action="store_true",
                     help="Print the runs the grid expands to and exit, without launching anything.")
  args.add_argument("--validate", action="store_true",
                     help="Smoke-test every {algo x env} combination instead of running a full "
                          "sweep: one rollout chunk + one train_step + optimizer step each, no "
                          "eval/checkpointing/wandb logging (see online_learning.validate_run). "
                          "Collapses every other axis to its first value -- exactly one run per "
                          "{algo, env} combination.")

  kwargs = args.parse_args()

  grid_base, axes, exclude = build_axes(kwargs)

  if kwargs.validate:
    axes = {k: (v if k in ("algo", "env") else v[:1]) for k, v in axes.items()}

  combos = build_combos(axes, exclude)
  if not combos:
    raise SystemExit("grid is empty -- every combination was excluded")
  tags = name_combos(combos, axes)

  base_config_path = kwargs.config or grid_base or os.path.join(CONFIG_DIR, f"{kwargs.script}.yaml")
  with open(base_config_path) as f:
    base_config = yaml.safe_load(f) or {}
  defaults = base_config.get("defaults", {})
  env_defaults = base_config.get("env_defaults") or {}

  if kwargs.validate:
    defaults = {**defaults, "validate": True}

  now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  sweep_dir = os.path.join(SCRIPT_DIR, "..", "sweeps", now)

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

  axis_desc = " x ".join(f"{len(v)} {k}" for k, v in axes.items() if len(v) > 1) or "1 run"
  kind = "validate run(s)" if kwargs.validate else "runs"
  print(f"Launching {len(combos)} {kind} ({axis_desc}), max_parallel={max_workers}{gpu_desc}, "
        f"logs under {sweep_dir}")

  if kwargs.dry_run:
    for combo, tag in zip(combos, tags):
      print(f"  {tag}: {combo}")
    raise SystemExit(0)

  os.makedirs(sweep_dir, exist_ok=True)

  def run_one_gated(script, combo, tag, defaults, env_defaults, sweep_dir, gpu_id, stream):
    sem = semaphores[gpu_id] if semaphores else None
    if sem is not None:
      sem.acquire()
    try:
      return run_one(script, combo, tag, defaults, env_defaults, sweep_dir,
                     gpu_id=gpu_id, stream=stream)
    finally:
      if sem is not None:
        sem.release()

  with ThreadPoolExecutor(max_workers=max_workers) as pool:
    futures = [
      pool.submit(run_one_gated, kwargs.script, combo, tag, defaults, env_defaults,
                  sweep_dir, gpus[i % len(gpus)] if gpus else None, kwargs.validate)
      for i, (combo, tag) in enumerate(zip(combos, tags))
    ]
    results = [f.result() for f in futures]

  failed = sum(1 for r in results if r != 0)
  print(f"Done: {len(results) - failed}/{len(results)} succeeded"
        + (f", {failed} failed -- see logs in {sweep_dir}" if failed else ""))
