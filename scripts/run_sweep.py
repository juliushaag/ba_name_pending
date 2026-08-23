import argparse
import datetime
import itertools
import os
import queue
import re
import shlex
import subprocess
import sys
import threading

import yaml

# Launches a grid of training runs as **one sequential worker per GPU**: each
# worker pulls the next combo off a shared queue, runs it as its own OS process
# (own interpreter + CUDA context) pinned to that worker's GPU via
# CUDA_VISIBLE_DEVICES, and only then takes the next one -- so a run never
# shares a GPU, and N GPUs finish the grid ~N times faster. With no --gpus it's
# a single worker, i.e. the whole sweep strictly one at a time. The grid is a
# cross product over named axes: `algo`/`env`/`seed`, plus any number of
# `model.<dotted>` axes (which become that run's model_overrides, see
# barl_jax.model_config.set_overrides) and `run.<key>` axes (which override the
# run config's own keys). Nothing has to be edited in place to sweep a tunable.
#
# Seeds are *not* an axis: `--seeds N` runs N of them vmapped inside each
# process (see barl_jax/multiseed.py), which is both cheaper than N processes
# and bit-identical to running each alone. The `seed` axis that remains is the
# *base* seed those N count up from, and is grid-file-only.
#
# Combos are queued algo-major -- every env/override for SAC is handed out
# before DIME's -- so a sweep reads top to bottom as it runs, modulo whatever
# reordering parallel workers finishing at different times introduces.

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(SCRIPT_DIR, "..", "config")

# barl_jax.config_tree imports nothing but yaml, so this launcher still never
# pulls in jax/mujoco -- and the snapshot it writes is produced by the same code
# that resolves it at run time, rather than by a copy of it that drifts.
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "src"))
from barl_jax.config_tree import ConfigTree  # noqa: E402

RESERVED_AXES = ("algo", "env", "seed")

# Workers write to the same terminal, so multi-line output is emitted under this.
PRINT_LOCK = threading.Lock()


def emit(text: str) -> None:
  with PRINT_LOCK:
    print(text, flush=True)


def detect_gpus() -> list[str]:
  try:
    out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout
  except (OSError, subprocess.CalledProcessError) as e:
    raise SystemExit(f"--gpus auto: could not query nvidia-smi ({e}) -- pass GPU indices explicitly")
  gpus = [line.strip() for line in out.splitlines() if line.strip()]
  if not gpus:
    raise SystemExit("--gpus auto: nvidia-smi reported no GPUs")
  return gpus


def build_axes(kwargs) -> tuple[str | None, dict, list[dict]]:
  """The sweep's axes from either a --grid file or the --algos/--envs
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
  # axis the grid file leaves to --algos/--envs.
  unknown = [k for ex in exclude for k in ex if k not in axes]
  if unknown:
    raise SystemExit(f"{kwargs.grid}: exclude names non-axis key(s) {unknown} -- it would never "
                     f"match anything. Axes are: {', '.join(axes)}")

  return base, axes, exclude


def build_combos(axes: dict, exclude: list[dict]) -> list[dict]:
  combos = [dict(zip(axes, values)) for values in itertools.product(*axes.values())]
  combos = [c for c in combos
            if not any(all(c.get(k) == v for k, v in ex.items()) for ex in exclude)]
  # Algo-major regardless of which order the axes were declared in, so runs
  # come out grouped the way `--algos SAC DIME AMDP` reads. Stable sort, so
  # within one algo the product's own order survives.
  order = {algo: i for i, algo in enumerate(axes["algo"])}
  return sorted(combos, key=lambda c: order[c["algo"]])


def name_combos(combos: list[dict], axes: dict, num_seeds: int) -> list[str]:
  """One filesystem-safe tag per combo: algo_env, plus the seed count when
  there's more than one, the base seed when that axis varies, and key=value for
  every *varying* non-reserved axis (a fixed --set value is the same in every
  run, so it only clutters the name -- it's recorded in the per-run YAML)."""
  varying = [k for k, v in axes.items() if k not in RESERVED_AXES and len(v) > 1]

  short = {k: k.rpartition(".")[2] for k in varying}
  for k, s in list(short.items()):
    if list(short.values()).count(s) > 1:
      short[k] = k.partition(".")[2].replace(".", "-")

  tags = []
  for combo in combos:
    parts = [str(combo["algo"]), str(combo["env"])]
    if len(axes["seed"]) > 1:
      parts.append(f"base{combo['seed']}")
    if num_seeds > 1:
      parts.append(f"{num_seeds}seeds")
    parts += [f"{short[k]}={combo[k]}" for k in varying]
    tags.append(re.sub(r"[^A-Za-z0-9.=_-]", "-", "_".join(parts)))
  return tags


def write_manifest(sweep_dir: str, path_args: dict, tags: list[str], combos: list[dict],
                   results: dict[int, int] | None = None) -> None:
  """`sweep.yaml`: what this sweep is, one entry per run, rewritten with each
  run's exit code once the grid is done. The per-run `config.yaml` files are
  each self-contained, so this is the index over them -- what varied, what it
  was called, and where it went."""
  manifest = dict(path_args)
  manifest["runs"] = [
    {"tag": tag, "dir": tag, "combo": combo,
     **({} if results is None or i not in results
        else {"exit_code": results[i], "status": "ok" if results[i] == 0 else "failed"})}
    for i, (tag, combo) in enumerate(zip(tags, combos))
  ]
  with open(os.path.join(sweep_dir, "sweep.yaml"), "w") as f:
    yaml.safe_dump(manifest, f, sort_keys=False)


def run_one(script: str, combo: dict, tag: str, defaults: dict, sweep_dir: str,
            num_seeds: int, gpu_id: str | None = None, stream: bool = False,
            progress: str = "") -> int:
  model, env = combo["algo"], combo["env"]

  # `seed` is the base the run's `num_seeds` vmapped seeds count up from
  # (barl_jax.trainer.Run: base_seed + i), not one seed per process.
  run_entry = {"model": model, "env": env, "seed": combo["seed"], "num_seeds": num_seeds}
  run_entry.update({k.partition(".")[2]: v for k, v in combo.items() if k.startswith("run.")})

  overrides = {k.partition(".")[2]: v for k, v in combo.items() if k.startswith("model.")}
  if overrides:
    run_entry["model_overrides"] = overrides

  # Everything this combo produces lands under one directory: the config it ran
  # with, its log, and (via `run_dir`, which trainer.train() checkpoints into)
  # its `best`/`latest`/`final` weights. A sweep is then self-contained -- no
  # hunting through a global checkpoints/ tree for which run made which file.
  run_dir = os.path.abspath(os.path.join(sweep_dir, tag))
  os.makedirs(run_dir, exist_ok=True)
  run_entry["run_dir"] = run_dir

  # The snapshot is fully resolved *under this combo's overrides*, so replaying
  # this config re-runs no layering and cannot drift with config/.
  snapshot = ConfigTree(CONFIG_DIR, overrides=overrides).snapshot_for(model)
  run_config = {"defaults": defaults, "runs": [run_entry],
                "snapshot": snapshot}

  run_config_path = os.path.join(run_dir, "config.yaml")
  with open(run_config_path, "w") as f:
    yaml.safe_dump(run_config, f)

  script_path = os.path.join(SCRIPT_DIR, f"{script}.py")
  log_path = os.path.join(run_dir, "run.log")
  cmd = [sys.executable, script_path, "--config", run_config_path]

  env_vars = os.environ.copy()
  label = tag
  if gpu_id is not None:
    env_vars["CUDA_VISIBLE_DEVICES"] = gpu_id
    label = f"gpu{gpu_id}|{tag}"

  emit(f"[{label}] starting {progress} -- dir: {run_dir}\n[{label}] $ {shlex.join(cmd)}")
  with open(log_path, "w") as logf:
    if stream:
      # --validate runs: tee each line (label-prefixed) to the terminal as well as the log file.
      proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env_vars,
      )
      for line in proc.stdout:
        emit(f"[{label}] {line.rstrip()}")
        logf.write(line)
      proc.wait()
    else:
      proc = subprocess.run(
        cmd, stdout=logf, stderr=subprocess.STDOUT, env=env_vars,
      )

  status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
  if proc.returncode == 0:
    emit(f"[{label}] {status}")
  else:
    with open(log_path, "r") as logf:
      emit(f"[{label}] {status}\n{logf.read()}")

  return proc.returncode


if __name__ == "__main__":

  args = argparse.ArgumentParser(
    description="Launch a grid of training runs sequentially, one OS process at a time, "
                "algo by algo.",
    epilog="Axes: algo/env, plus model.<dotted path> (a model override, e.g. "
           "model.components.critic.rho) and run.<key> (a run-config key, e.g. run.utd_ratio). "
           "Seeds are not an axis -- --seeds N vmaps N of them inside each run.")
  args.add_argument("--grid", default=None,
                     help="YAML sweep spec: {base: <run config>, axes: {name: [values]}, "
                          "exclude: [{axis: value, ...}]}. Combined with any --algos/--envs/"
                          "--set given, which win per axis.")
  args.add_argument("--algos", nargs="+", help="`algo` axis, e.g. SAC DIME AMDP")
  args.add_argument("--envs", nargs="+", help="`env` axis, e.g. HalfCheetah CheetahRun")
  args.add_argument("--seeds", type=int, default=1,
                     help="Seeds per run, vmapped inside the one process rather than run as "
                          "separate ones -- so `--algos SAC DIME --seeds 5` is 2 runs of 5 "
                          "seeds, not 10 runs. Bit-identical to running each seed alone.")
  args.add_argument("--set", action="append", metavar="AXIS=VALUE",
                     help="Pin one axis to a single value across the whole sweep, e.g. "
                          "--set model.components.critic.rho=0.0 --set run.batch_size=512. "
                          "Repeatable.")
  args.add_argument("--script", default="online_learning",
                     choices=["online_learning"],
                     help="Which training script to launch each run with")
  args.add_argument("--config", default=None,
                     help="Base experiment config providing `defaults:` for every run "
                          "(default: --grid's `base:`, else config/experiments/<script>.yaml). "
                          "Run keys themselves are declared in config/run/defaults.yaml and "
                          "per-env overrides in config/run/envs.yaml; this only carries what is "
                          "specific to these runs.")
  args.add_argument("--sweep_tag", default=None, help="Sweep tag")
  args.add_argument("--sweep_dir", default=None,
                     help="Where to write this sweep's per-run directories "
                          "(default: sweeps/<timestamp>). A batch job points this at node-local "
                          "scratch and copies the tree back when it finishes.")
  args.add_argument("--dry_run", action="store_true",
                     help="Print the runs the grid expands to and exit, without launching anything.")
  args.add_argument("--validate", action="store_true",
                     help="Smoke-test every {algo x env} combination instead of running a full "
                          "sweep: one rollout chunk + one train_step + optimizer step each, no "
                          "eval/checkpointing/wandb logging (see online_learning.validate_run). "
                          "Collapses every other axis to its first value -- exactly one run per "
                          "{algo, env} combination.")

  kwargs = args.parse_args()

  gpus = detect_gpus()

  grid_base, axes, exclude = build_axes(kwargs)

  if kwargs.validate:
    axes = {k: (v if k in ("algo", "env") else v[:1]) for k, v in axes.items()}

  if kwargs.seeds < 1:
    raise SystemExit(f"--seeds must be at least 1, got {kwargs.seeds}")

  combos = build_combos(axes, exclude)
  if not combos:
    raise SystemExit("grid is empty -- every combination was excluded")
  tags = name_combos(combos, axes, kwargs.seeds)

  base_config_path = kwargs.config or grid_base or os.path.join(CONFIG_DIR, "experiments", f"{kwargs.script}.yaml")
  with open(base_config_path) as f:
    base_config = yaml.safe_load(f) or {}
  defaults = base_config.get("defaults", {})

  if kwargs.validate:
    defaults = {**defaults, "validate": True}

  now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # noqa: DTZ005
  sweep_dir = os.path.abspath(kwargs.sweep_dir
                              or os.path.join(SCRIPT_DIR, "..", "sweeps", now))

  # Threaded through `defaults` so every run in the grid can be filtered together in wandb.
  if kwargs.sweep_tag is None:
    sweep_tag = "sweep-" + datetime.datetime.now().strftime("%m-%d-%H-%M")  # noqa: DTZ005
  else:
    sweep_tag = kwargs.sweep_tag

  defaults = {**defaults, "sweep_tag": sweep_tag}

  axis_desc = " x ".join(f"{len(v)} {k}" for k, v in axes.items() if len(v) > 1) or "1 run"
  kind = "validate run(s)" if kwargs.validate else "runs"
  seed_desc = f", {kwargs.seeds} seeds vmapped each" if kwargs.seeds > 1 else ""
  pace_desc = (f"{len(gpus)} at a time, one per gpu ({', '.join(gpus)})" if gpus
               else "one at a time")
  print(f"Running {len(combos)} {kind} ({axis_desc}) {pace_desc}{seed_desc}, "
        f"logs under {sweep_dir}")

  if kwargs.dry_run:
    for combo, tag in zip(combos, tags):
      print(f"  {tag}: {combo}")
    raise SystemExit(0)

  os.makedirs(sweep_dir, exist_ok=True)

  manifest_head = {
    "sweep_tag": sweep_tag,
    "created":   now,
    "command":   shlex.join(sys.argv),
    "script":    kwargs.script,
    "base_config": os.path.abspath(base_config_path),
    "seeds":     kwargs.seeds,
    "gpus":      gpus or None,
    "validate":  bool(kwargs.validate),
    "axes":      {k: list(v) for k, v in axes.items()},
    "exclude":   exclude or None,
  }
  write_manifest(sweep_dir, manifest_head, tags, combos)

  pending: queue.Queue = queue.Queue()
  for i, (combo, tag) in enumerate(zip(combos, tags)):
    pending.put((i, combo, tag))

  results: dict[int, int] = {}

  def worker(gpu_id: str | None) -> None:
    while True:
      try:
        i, combo, tag = pending.get_nowait()
      except queue.Empty:
        return
      results[i] = run_one(kwargs.script, combo, tag, defaults, sweep_dir,
                           num_seeds=kwargs.seeds, gpu_id=gpu_id, stream=kwargs.validate,
                           progress=f"{i + 1}/{len(combos)}")

  # One worker per GPU, each blocking on its own subprocess -- so a GPU holds
  # exactly one run at a time and picks up the next combo the moment it's free.
  threads = [threading.Thread(target=worker, args=(gpu_id,), daemon=True)
             for gpu_id in (gpus or [None])]
  for t in threads:
    t.start()
  for t in threads:
    t.join()

  write_manifest(sweep_dir, manifest_head, tags, combos, results)

  failed = sum(1 for r in results.values() if r != 0)
  print(f"Done: {len(results) - failed}/{len(results)} succeeded"
        + (f", {failed} failed -- see logs in {sweep_dir}" if failed else ""))
