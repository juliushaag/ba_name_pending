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
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)


from barl_jax.config.toolkit import CONFIG_DIR, resolve_includes
from barl_jax.config.tree import ConfigTree

DEFAULT_BASE_CONFIG = str(CONFIG_DIR / "experiments" / "online_learning.yaml")

RESERVED_AXES = ("algo", "env", "seed")


def default_num_seeds() -> int:
  """`num_seeds` from config/run/defaults.yaml -- the single place the seed count
  is declared. submit.py's `--seeds` overrides it; nothing else has an opinion."""
  return int(ConfigTree(CONFIG_DIR).run["num_seeds"])


# ---------------------------------------------------------------------------
# config check -- reusable across submit.py's grid sweeps and hpo.py's search
# sweeps, since both materialize the same kind of run and want the same
# preflight before submitting it.
# ---------------------------------------------------------------------------

#: The check itself, run as a program in a child process. Everything it needs
#: is on stdin; what it found comes back through the file named by argv[1],
#: which import chatter on stdout cannot corrupt.
CHECK_PROGRAM = """
import importlib, json, sys

algos, envs = json.load(sys.stdin)

import barl_jax  # noqa: F401  -- the import is what populates the registries
model_config = importlib.import_module("barl_jax.config.tree")
from barl_jax.env import Environment

known_envs = set(Environment.available())
problems = [f"{env!r} is not a registered environment"
            for env in envs if env not in known_envs]

for algo in algos:
    try:
        problems += [f"{algo}: {p}" for p in model_config.validate(algo, model_config.ConfigTree())]
    except Exception as exc:  # an unresolvable name raises rather than reporting
        problems.append(f"{algo}: {type(exc).__name__}: {exc}")

with open(sys.argv[1], "w") as out:
    json.dump(problems, out)
"""


def check_config(algos: list[str], envs: list[str]) -> list[str]:
  """Everything an array of 40 tasks would otherwise discover 40 times, at the
  point where it is still one error message: unregistered env or algo names,
  and anything `model_config.validate()` rejects (an override that lands on no
  slot, a kwarg no component accepts, a slot missing its `tier:`).

  Resolution only: no env is built, no weights are allocated, nothing touches
  a device."""
  # In a child process, and that is the whole point. The check wants
  # JAX_PLATFORMS=cpu -- a login node has no GPU to spend -- while every
  # executor hands *this* process's environment to the runs (LocalExecutor's
  # `env=`, sbatch's `--export=ALL`, tmux's `-e`). A JAX_PLATFORMS=cpu left in
  # os.environ here does not just slow the check down: it trains every run of
  # the sweep on the CPU, ~160x slower per gradient step and indistinguishable
  # from a hang. Setting it only in the child is what makes that unleakable,
  # for this variable and any other the check ever needs -- and it keeps jax
  # and mujoco out of the launcher process entirely.
  env = {**os.environ, "MUJOCO_GL": "egl", "JAX_PLATFORMS": "cpu"}

  with tempfile.TemporaryDirectory() as tmp:
    found = os.path.join(tmp, "problems.json")
    proc = subprocess.run([sys.executable, "-c", CHECK_PROGRAM, found],
                          input=json.dumps([list(algos), list(envs)]),
                          text=True, capture_output=True, env=env)

    if proc.returncode != 0 or not os.path.exists(found):
      tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
      return ["the config check could not run:", *(f"  {line}" for line in tail)]

    with open(found) as f:
      return json.load(f)


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


def prepare_sweep(combos: list[dict], tags: list[str], *, base_config_path: str,
                  sweep_dir: str, num_seeds: int | None, validate: bool,
                  sweep_name: str, executor_name: str, seeds: int,
                  head_extra: dict) -> tuple[list[Run], dict]:
  """The launch preparation submit.py's grid sweeps and hpo.py's search sweeps
  share once each has its own combos/tags: load the base config's `defaults:`,
  apply the two launch-time overrides every caller makes (`--validate`
  collapses every run to the smoke-test path; `sweep_tag` is stamped in so
  every run of this sweep is one filter in wandb), materialize every run's
  `config.yaml`, and publish `sweep.yaml`.

  `head_extra` carries whatever only the caller knows about the sweep it
  built -- a grid's `axes`/`exclude`/`debug`, or a search's `kind`/`sampler`/
  `n_trials`/`objective` -- passed through untouched rather than guessed at
  here, so a grid sweep's manifest and a search sweep's stay explicitly
  different shapes instead of one merged, lowest-common-denominator one."""
  with open(base_config_path) as f:
    defaults = dict((yaml.safe_load(f) or {}).get("defaults") or {})
  if validate:
    defaults["validate"] = True
  defaults["sweep_tag"] = sweep_name

  os.makedirs(sweep_dir, exist_ok=True)
  runs = materialize(combos, tags, defaults=defaults, sweep_dir=sweep_dir, num_seeds=num_seeds)

  head = {"sweep_tag": sweep_name, "created": sweep_name, "command": sweep_command(),
         "base_config": os.path.abspath(base_config_path), "executor": executor_name,
         "seeds": seeds, **head_extra}
  write_sweep_yaml(sweep_dir, head, runs)
  return runs, head


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
