"""The sweep language: a grid spec becomes a list of ready-to-run configs.

Everything here is pure. It expands axes, names each run, and writes each run's
self-contained `config.yaml` -- and executes nothing. That is what lets every
backend in executors.py be no more than "run these N commands, M at a time":
the grid is fully expanded before a single task starts, so no executor, job
script or queue manager ever has to know what an axis is.

Axes are `algo`/`env`/`seed`, plus any number of `model.<dotted path>` axes
(written into that run's `model_overrides:`) and `run.<key>` axes (written into
its run entry). Since the axes are just overrides, no config file is ever edited
to run a sweep.
"""

import itertools
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)


from barl_jax.config.toolkit import CONFIG_DIR, resolve_includes
from barl_jax.config.tree import ConfigTree

DEFAULT_BASE_CONFIG = str(CONFIG_DIR / "experiments" / "online_learning.yaml")

RESERVED_AXES = ("algo", "env", "seed")

# Workers write to the same terminal, so multi-line output is emitted under this.
PRINT_LOCK = threading.Lock()


def emit(text: str) -> None:
  with PRINT_LOCK:
    print(text, flush=True)


def detect_gpus() -> list[str]:
  """Every GPU nvidia-smi reports, as CUDA_VISIBLE_DEVICES values. Empty on a
  box with no nvidia-smi, which the local executor reads as "one run at a time"."""
  try:
    out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout
  except (FileNotFoundError, subprocess.CalledProcessError):
    return []
  return [line.strip() for line in out.splitlines() if line.strip()]


def default_num_seeds() -> int:
  """`num_seeds` from config/run/defaults.yaml -- the single place the seed count
  is declared. submit.py's `--seeds` overrides it; nothing else has an opinion."""
  return int(ConfigTree(CONFIG_DIR).run["num_seeds"])


# ---------------------------------------------------------------------------
# axes
# ---------------------------------------------------------------------------

def build_axes(*, grid: str = None, algos: list = None, envs: list = None,
               sets: list = None) -> tuple[str | None, dict, list[dict]]:
  """The sweep's axes from a `--grid` file, the `--algos`/`--envs` shorthand and
  any `--set` pins (which win per axis). Returns (base_config, axes, exclude).

  `$()` includes (barl_jax.configs, loaded above by path for the same reason
  config_tree.py itself is) are resolved against CONFIG_DIR, so e.g. an
  `exclude:` list shared by more than one grid spec can live in one place."""
  base, axes, exclude = None, {}, []

  if grid:
    with open(grid) as f:
      spec = yaml.safe_load(f) or {}
    spec = resolve_includes(spec, Path(grid).resolve().parent, Path(CONFIG_DIR))
    base = spec.get("base")
    axes = dict(spec.get("axes") or {})
    exclude = list(spec.get("exclude") or [])

  if algos:
    axes["algo"] = list(algos)
  if envs:
    axes["env"] = list(envs)

  for assignment in sets or []:
    key, sep, value = assignment.partition("=")
    if not sep:
      raise SystemExit(f"--set {assignment!r} is not of the form axis=value")
    axes[key.strip()] = [yaml.safe_load(value)]

  for name, values in axes.items():
    if not isinstance(values, list) or not values:
      raise SystemExit(f"axis {name!r} must be a non-empty list of values, got {values!r}")
    if name not in RESERVED_AXES and not name.startswith(("model.", "run.")):
      raise SystemExit(f"axis {name!r} must be one of {RESERVED_AXES}, or be prefixed "
                       f"'model.' (a dotted model override) or 'run.' (a run-config key)")
    if name in ("run.model", "run.env", "run.seed"):
      bare = name.partition(".")[2]
      raise SystemExit(f"axis {name!r} would fight the {bare!r} the run entry already gets -- "
                       f"use the {bare.replace('model', 'algo')!r} axis instead")

  for required in ("algo", "env"):
    if not axes.get(required):
      raise SystemExit(f"no {required!r} axis -- pass --{required}s or declare one in --grid")
  axes.setdefault("seed", [0])

  # Checked only once the CLI axes are in, since an exclude may well name an axis
  # the grid file leaves to --algos/--envs.
  unknown = [k for ex in exclude for k in ex if k not in axes]
  if unknown:
    raise SystemExit(f"{grid}: exclude names non-axis key(s) {unknown} -- it would never match "
                     f"anything. Axes are: {', '.join(axes)}")

  return base, axes, exclude


def build_combos(axes: dict, exclude: list[dict]) -> list[dict]:
  combos = [dict(zip(axes, values)) for values in itertools.product(*axes.values())]
  combos = [c for c in combos
            if not any(all(c.get(k) == v for k, v in ex.items()) for ex in exclude)]
  # Algo-major regardless of which order the axes were declared in, so runs come
  # out grouped the way `--algos SAC DIME AMDP` reads. Stable sort, so within one
  # algo the product's own order survives.
  order = {algo: i for i, algo in enumerate(axes["algo"])}
  return sorted(combos, key=lambda c: order[c["algo"]])


def name_combos(combos: list[dict], axes: dict, num_seeds: int) -> list[str]:
  """One filesystem-safe tag per combo: algo_env, plus the seed count when there
  is declared, the base seed when that axis varies, and key=value for every
  *varying* non-reserved axis (a fixed --set value is the same in every run, so
  it only clutters the name -- it is recorded in the per-run YAML)."""
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
    parts.append(f"{num_seeds}seeds")
    parts += [f"{short[k]}={combo[k]}" for k in varying]
    tags.append(re.sub(r"[^A-Za-z0-9.=_-]", "-", "_".join(parts)))
  return tags


# ---------------------------------------------------------------------------
# materialization
# ---------------------------------------------------------------------------

@dataclass
class Run:
  """One point of the grid, already written to disk and ready to execute. An
  executor needs nothing from this but `config` and somewhere to put the output."""
  index:  int
  tag:    str
  combo:  dict
  #: This run's directory, relative to the sweep dir -- the tree is relocatable
  #: (a Slurm task works on node-local scratch and copies it back), so nothing
  #: here is an absolute path.
  dir:    str
  config: str
  exit_code: int | None = field(default=None, compare=False)

  def command(self, sweep_dir: str, run_dir: str = None) -> list[str]:
    """The one command that runs this. `run_dir` redirects the output somewhere
    other than beside the config -- node-local scratch, for a batch job."""
    uv = shutil.which("uv")
    cmd = ([uv, "run", "python", os.path.join(SCRIPT_DIR, "online_learning.py")]
           if uv else [sys.executable, os.path.join(SCRIPT_DIR, "online_learning.py")])
    cmd += ["--config", os.path.join(sweep_dir, self.config)]
    if run_dir is not None:
      cmd += ["--run-dir", run_dir]
    return cmd


def materialize(combos: list[dict], tags: list[str], *, defaults: dict, sweep_dir: str,
                num_seeds: int | None) -> list[Run]:
  """Writes every run's self-contained `config.yaml` and returns the runs.

  This is the step that makes the rest of the system simple: after it, a run is
  a config path, and the grid it came from is nobody else's business. Each config
  carries a fully resolved `snapshot:` taken *under that run's own overrides*, so
  it replays exactly regardless of what config/ says later."""
  runs = []
  import barl_jax  # noqa: F401 -- registries needed for snapshot_for()
  for index, (combo, tag) in enumerate(zip(combos, tags)):
    # `seed` is the base the run's `num_seeds` vmapped seeds count up from
    # (barl_jax.trainer.Run: base_seed + i), not one seed per process.
    entry = {"model": combo["algo"], "env": combo["env"], "seed": combo["seed"]}
    if num_seeds is not None:
      entry["num_seeds"] = num_seeds
    entry.update({k.partition(".")[2]: v for k, v in combo.items() if k.startswith("run.")})

    overrides = {k.partition(".")[2]: v for k, v in combo.items() if k.startswith("model.")}
    if overrides:
      entry["model_overrides"] = overrides

    # Everything this run produces lands under one directory: the config it ran
    # with, its log, and (via `run_dir`, which trainer.train() checkpoints into)
    # its weights. A sweep is then self-contained -- no hunting through a global
    # checkpoints/ tree for which run made which file.
    run_dir = os.path.join(sweep_dir, tag)
    os.makedirs(run_dir, exist_ok=True)
    entry["run_dir"] = os.path.abspath(run_dir)

    snapshot = ConfigTree(CONFIG_DIR, overrides=overrides).snapshot_for(combo["algo"])
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
      yaml.safe_dump({"defaults": defaults, "runs": [entry], "snapshot": snapshot}, f)

    runs.append(Run(index=index, tag=tag, combo=combo,
                    dir=tag, config=os.path.join(tag, "config.yaml")))
  return runs


def write_sweep_yaml(sweep_dir: str, head: dict, runs: list[Run]) -> str:
  """`sweep.yaml`: what this sweep is, one entry per run. The per-run
  `config.yaml` files are each self-contained, so this is the index over them --
  what varied, what it was called, where it went, and how it ended."""
  manifest = {**head, "runs": [
    {"tag": r.tag, "dir": r.dir, "combo": r.combo,
     **({} if r.exit_code is None
        else {"exit_code": r.exit_code, "status": "ok" if r.exit_code == 0 else "failed"})}
    for r in runs]}
  path = os.path.join(sweep_dir, "sweep.yaml")
  with open(path, "w") as f:
    yaml.safe_dump(manifest, f, sort_keys=False)
  return path


def describe(combos: list[dict], axes: dict, num_seeds: int) -> str:
  axis_desc = " x ".join(f"{len(v)} {k}" for k, v in axes.items() if len(v) > 1) or "1 point"
  return (f"{len(combos)} run(s) ({axis_desc}), "
          f"{num_seeds} seed{'s' if num_seeds != 1 else ''} vmapped each")


def sweep_command() -> str:
  return shlex.join(sys.argv)
