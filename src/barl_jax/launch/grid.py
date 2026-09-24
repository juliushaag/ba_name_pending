"""Expand grid axes and exclusions into named, self-contained run configs.

algo/env/seed select each run; model.<path> and run.<key> override its settings.
All runs are materialized before an executor starts any tasks."""

import itertools
import os
import re
import shlex
import shutil
import sys
from dataclasses import dataclass, field

import yaml

from barl_jax.config import from_yaml
from barl_jax.config.run import RunConfig

RESERVED_AXES = ("algo", "env", "seed")


def _config_validate(config, *, context: str = "run") -> None:
  """Validate one candidate before it is written or submitted."""
  try:
    config.check()
  except Exception as exc:
    raise ValueError(f"{context}: invalid configuration: {exc}") from exc


def _nested_set(mapping: dict, dotted: str, value) -> None:
  """Set one dotted model axis in the canonical nested override shape."""
  node = mapping
  parts = dotted.split(".")
  for part in parts[:-1]:
    child = node.setdefault(part, {})
    if not isinstance(child, dict):
      raise ValueError(f"model override {dotted!r} collides with a scalar at {part!r}")
    node = child
  node[parts[-1]] = value


def _materialize_candidate(combo: dict, *, defaults: dict | None = None):
  """Build and validate one expanded point from Python values."""
  model, env = combo["algo"], combo["env"]
  payload = {"model": model, "env": env}
  payload.update({key.partition(".")[2]: value for key, value in combo.items()
                 if key.startswith("run.")})
  if "seed" in combo:
    payload["seed"] = combo["seed"]
  payload.update(defaults or {})
  model_overrides = {}
  for key, value in combo.items():
    if key.startswith("model."):
      _nested_set(model_overrides, key.removeprefix("model."), value)
  if model_overrides:
    payload["model_overrides"] = model_overrides
  config = RunConfig.from_dict(payload)
  _config_validate(config, context=f"{model} on {env}")
  return config


def default_num_seeds() -> int:
  """Default vmapped seed count owned by ``RunConfig``."""
  return int(RunConfig.defaults(model="SAC", env="HalfCheetah").num_seeds)


def check_config(algos: list[str], envs: list[str]) -> list[str]:
  """Resolve and validate every model/environment pair before submission."""
  problems = []
  for algo in algos:
    for env in envs:
      try:
        config = RunConfig.defaults(model=algo, env=env)
        config.check()
      except Exception as exc:
        problems.append(f"{algo} on {env}: {type(exc).__name__}: {exc}")
  return problems


# ---------------------------------------------------------------------------
# axes
# ---------------------------------------------------------------------------

def build_axes(*, grid: str = None, algos: list = None, envs: list = None,
               sets: list = None) -> tuple[dict, list[dict]]:
  """Read a tiny ``{axes, exclude}`` grid document and merge CLI pins."""
  axes, exclude = {}, []
  if grid:
    with open(grid) as f:
      spec = yaml.safe_load(f) or {}
    if not isinstance(spec, dict):
      raise SystemExit(f"{grid}: grid document must be a mapping")
    if "base" in spec:
      raise SystemExit(f"{grid}: `base` is not supported; declare only `axes` and `exclude`")
    unknown = sorted(set(spec) - {"axes", "exclude"})
    if unknown:
      raise SystemExit(f"{grid}: unknown grid key(s): {', '.join(unknown)}")
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
                       "'model.' (a model override) or 'run.' (a run-config key)")
    if name in ("run.model", "run.env", "run.seed"):
      bare = name.partition(".")[2]
      raise SystemExit(f"axis {name!r} would fight the {bare!r} selector; use the "
                       f"{bare.replace('model', 'algo')} axis instead")
  for required in ("algo", "env"):
    if not axes.get(required):
      raise SystemExit(f"no {required!r} axis -- pass --{required}s or declare one in --grid")
  axes.setdefault("seed", [0])
  unknown = [key for ex in exclude for key in ex if key not in axes]
  if unknown:
    raise SystemExit(f"{grid}: exclude names non-axis key(s) {unknown} -- axes: {', '.join(axes)}")
  return axes, exclude


def build_combos(axes: dict, exclude: list[dict]) -> list[dict]:
  combos = [dict(zip(axes, values)) for values in itertools.product(*axes.values())]
  combos = [c for c in combos
            if not any(all(c.get(k) == v for k, v in ex.items()) for ex in exclude)]
  # Env-major regardless of which order the axes were declared in, so runs come
  # out grouped the way `--envs Ant Humanoid` reads: every algo on one env
  # before moving to the next, so cross-algo comparisons are available early
  # in the sweep instead of only once every env has been through the first
  # algo. Stable sort, so within one env the product's own order survives.
  order = {env: i for i, env in enumerate(axes["env"])}
  return sorted(combos, key=lambda c: order[c["env"]])


def name_combos(combos: list[dict], axes: dict, num_seeds: int) -> list[str]:
  """One filesystem-safe tag per combo: algo_env, plus the seed count when there
  is declared, the base seed when that axis varies, and key=value for every
  *varying* non-reserved axis (a fixed --set value is the same in every run, so
  it only clutters the name -- it is recorded in the per-run YAML)."""
  varying = [k for k, v in axes.items() if k not in ("algo", "env", "seed") and len(v) > 1]

  short = {k: k.rpartition(".")[2] for k in varying}
  for k, s in list(short.items()):
    if list(short.values()).count(s) > 1:
      short[k] = k.partition(".")[2].replace(".", "-")

  tags = []
  for combo in combos:
    parts = [str(combo["algo"]), str(combo["env"])]
    if len(axes["seed"]) > 1:
      parts.append(f"base{combo['seed']}")
    parts.append(f"{combo.get('run.num_seeds', num_seeds)}seeds")
    parts += [f"{short[k]}={combo[k]}" for k in varying]
    tags.append(re.sub(r"[^A-Za-z0-9.=_-]", "-", "_".join(parts)))
  return tags


# ---------------------------------------------------------------------------
# materialization
# ---------------------------------------------------------------------------

def command_for(kind: str, config_path: str, *, output_path: str = None,
                run_dir: str = None) -> list[str]:
  """The one command that runs a manifest row, whatever launched it -- shared by
  `Run.command()` (LocalExecutor) and `tmux_pool.py`'s `Pool.start()`, so a
  submitted task's command line is declared exactly once, not once per
  backend. (The two .slurm scripts build their own command lines directly in
  bash -- they read a manifest row, not a `Run` -- but invoke the same two
  programs this does.)

  `kind='train'` (`scripts/train.py`): `run_dir`, given, redirects
  checkpoint output to node-local scratch. `kind='eval'`
  (`scripts/evaluate.py`): `output_path`, given, overrides where its combined
  `results.json` goes -- omitted, `evaluate.py` falls back to the config's own
  `output:` (an absolute path `materialize_eval()` already wrote), which is
  why no caller here needs to pass one; only run_eval.slurm's node-local
  scratch redirection does, and it passes `--output` itself."""
  uv = shutil.which("uv")
  py = [uv, "run", "python"] if uv else [sys.executable]
  if kind == "eval":
    cmd = [*py, "scripts/evaluate.py", "--config", config_path]
    if output_path is not None:
      cmd += ["--output", output_path]
    return cmd
  cmd = [*py, "scripts/train.py", "--config", config_path]
  if run_dir is not None:
    cmd += ["--run-dir", run_dir]
  return cmd


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
  #: 'train' (scripts/train.py, the default -- every existing caller) or
  #: 'eval' (scripts/evaluate.py, see materialize_eval()).
  kind: str = "train"
  exit_code: int | None = field(default=None, compare=False)

  def command(self, sweep_dir: str, run_dir: str = None) -> list[str]:
    """The one command that runs this. `run_dir` redirects a *training* run's
    checkpoint output somewhere other than beside the config -- node-local
    scratch, for a batch job; an eval batch's own config already carries an
    absolute `output:`, so it needs no such override here (see command_for)."""
    return command_for(self.kind, os.path.join(sweep_dir, self.config), run_dir=run_dir)


def materialize(combos: list[dict], tags: list[str], *, defaults: dict = None, sweep_dir: str,
                num_seeds: int | None) -> list[Run]:
  """Write one strict, self-contained training YAML per expanded point.

  The grid file is only an axis specification. Every point is materialized by
  ``RunConfig`` in Python, validated before it is written, and serialized as a
  single mapping accepted by ``from_yaml``.
  """
  runs = []
  for index, (combo, tag) in enumerate(zip(combos, tags)):
    metadata = dict(defaults or {})
    if num_seeds is not None:
      metadata["num_seeds"] = num_seeds
    config = _materialize_candidate(combo, defaults=metadata)

    run_dir = os.path.join(sweep_dir, tag)
    config = config.replace(run_dir=os.path.abspath(run_dir))
    _config_validate(config, context=f"{combo['algo']} on {combo['env']}")

    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "config.yaml")
    with open(path, "w") as f:
      yaml.safe_dump(config.to_dict(), f, sort_keys=False)
    # Validate the exact emitted mapping with the public loader. This catches
    # serializers that accidentally omit selector metadata or nested values.
    from_yaml(path)

    runs.append(Run(index=index, tag=tag, combo=combo, dir=tag,
                    config=os.path.join(tag, "config.yaml")))
  return runs


def batch_tags(items: list, batch_size: int | None) -> list[list]:
  """Chunks of `batch_size` (default: all of them, one chunk total) -- tags
  (`--from-sweep`) or eval entry dicts (`--eval-config`), `materialize_eval()`
  doesn't care which. Pure and side-effect-free on purpose -- separate from
  `materialize_eval()` so a caller can preview the grouping (submit.py's
  --dry-run) without writing anything to disk first."""
  batch_size = batch_size or len(items)
  return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def materialize_eval(entries: list[dict], *, batch_size: int | None,
                     eval_defaults: dict, sweep_dir: str) -> list[Run]:
  """Groups `entries` -- each a full `scripts/evaluate.py --config` `runs:`
  item (`{path, env?, ...}`, see experiments/eval.yaml for the schema) -- into
  batches of `batch_size` (default: all of them, one batch total -- eval is
  cheap enough that one task per checkpoint would waste a node's worth of
  scheduling overhead), writes one self-contained `eval.yaml` per batch (the
  exact shape `evaluate.py --config` already reads) and returns one
  `kind='eval'` Run per batch.

  This is submit.py's one path from "some checkpoints to evaluate" to
  `{"path": "<from_sweep>/<tag>"}` per axis combo, one tag per entry, `env`
  always omitted -- `evaluate.py`'s own discovery reads it from
  `<from_sweep>/<tag>/config.yaml`, the same file `materialize()` wrote there
  when that tag was trained. Either way, an entry with no matching checkpoint
  (never trained, or under a different `num_seeds`) surfaces as a
  per-checkpoint `error` in that batch's results.json -- evaluate_entry's own
  isolation, not something to special-case here."""
  batches = batch_tags(entries, batch_size)

  runs = []
  for index, batch in enumerate(batches):
    tag = f"batch{index:03d}"
    batch_dir = os.path.join(sweep_dir, tag)
    os.makedirs(batch_dir, exist_ok=True)

    # Absolute, like materialize()'s run_dir: self-contained regardless of
    # what invokes this file, not just Run.command()'s own --output (which
    # still wins if given -- evaluate.py's main() takes --output over a
    # config's own `output:`).
    config = {"output": os.path.join(os.path.abspath(batch_dir), "results.json"),
              "defaults": eval_defaults, "runs": batch}
    with open(os.path.join(batch_dir, "eval.yaml"), "w") as f:
      yaml.safe_dump(config, f, sort_keys=False)

    paths = [entry["path"] for entry in batch]
    runs.append(Run(index=index, tag=tag, combo={"paths": paths}, dir=tag,
                    config=os.path.join(tag, "eval.yaml"), kind="eval"))
  return runs


def prepare_sweep(combos: list[dict], tags: list[str], *,
                  sweep_dir: str, num_seeds: int | None, validate: bool,
                  sweep_name: str, executor_name: str, seeds: int,
                  head_extra: dict) -> tuple[list[Run], dict]:
  """Materialize run configs and write their initial sweep manifest."""
  
  defaults = {}

  if validate:
    defaults["validate"] = True
  defaults["sweep_tag"] = sweep_name

  os.makedirs(sweep_dir, exist_ok=True)
  runs = materialize(combos, tags, defaults=defaults, sweep_dir=sweep_dir, num_seeds=num_seeds)

  head = {"sweep_tag": sweep_name, "created": sweep_name, "command": shlex.join(sys.argv),
        "executor": executor_name,
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

