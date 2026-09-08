"""Report package versions, optional simulators, and submodule readiness.

    uv run python scripts/debug.py --probes --json

Backend and EGL probes run in subprocesses with timeouts to contain native
crashes. Missing JAX or a failed backend probe produces a nonzero exit status;
optional simulator and render failures are informational."""

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROBE_TIMEOUT = 30

# name -> the distribution's importable top-level module, for optional extras
# this repo integrates (see pyproject.toml's [project.optional-dependencies]
# and the root workspace). Import-checked, never imported for real here --
# CLAUDE.md keeps simulator imports inside environment factories.
OPTIONAL_SIMULATORS = {
  "mujoco": "mujoco",
  "dm-control": "dm_control",
  "mani-skill": "mani_skill",
  "humanoid-bench": "humanoid_bench",
}

DIST_NAMES = {"jax": "jax", "flax": "flax", "optax": "optax", "numpy": "numpy",
              "gymnasium": "gymnasium", "orbax.checkpoint": "orbax-checkpoint"}


def versions() -> dict:
  import importlib.metadata as md
  out = {"python": sys.version.split()[0]}
  for pkg, dist in DIST_NAMES.items():
    try:
      out[pkg] = md.version(dist)
    except md.PackageNotFoundError:
      out[pkg] = None
  return out


def simulator_availability() -> dict:
  return {name: importlib.util.find_spec(mod) is not None
          for name, mod in OPTIONAL_SIMULATORS.items()}


def submodule_status() -> dict:
  path = ROOT / "third_party" / "humanoid-bench"
  populated = path.exists() and any(path.iterdir())
  return {"third_party/humanoid-bench": "populated" if populated else
          "missing/empty (run `git submodule update --init`)"}


def _run_probe(code: str) -> dict:
  """Runs `code` in a fresh subprocess so a native crash (segfault, XLA abort)
  produces a reported failure, not a dead doctor process."""
  try:
    result = subprocess.run(
      [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
      timeout=PROBE_TIMEOUT)
  except subprocess.TimeoutExpired:
    return {"ok": False, "detail": f"timed out after {PROBE_TIMEOUT}s"}
  if result.returncode != 0:
    tail = (result.stderr or result.stdout).strip().splitlines()
    return {"ok": False, "detail": tail[-1] if tail else f"exit {result.returncode}"}
  return {"ok": True, "detail": result.stdout.strip()}


def backend_probe() -> dict:
  return _run_probe(
    "import jax; d = jax.devices(); "
    "print(f'{jax.default_backend()}: {d}')")


def render_probe() -> dict:
  return _run_probe(
    "import os; os.environ.setdefault('MUJOCO_GL', 'egl'); "
    "import mujoco; "
    "m = mujoco.MjModel.from_xml_string('<mujoco/>'); "
    "r = mujoco.Renderer(m); r.render(); "
    "print('egl render ok')")


def report(*, run_probes: bool) -> dict:
  data = {
    "versions": versions(),
    "simulators": simulator_availability(),
    "submodules": submodule_status(),
  }
  if run_probes:
    data["backend_probe"] = backend_probe()
    data["render_probe"] = render_probe()
  return data


def required_probes_failed(data: dict) -> bool:
  if data["versions"].get("jax") is None:
    return True
  if "backend_probe" in data and not data["backend_probe"]["ok"]:
    return True
  return False


def print_text(data: dict) -> None:
  print("versions")
  for pkg, v in data["versions"].items():
    print(f"  {pkg:<20} {v or 'NOT INSTALLED'}")

  print("\noptional simulators")
  for name, present in data["simulators"].items():
    print(f"  {name:<20} {'available' if present else 'not installed'}")

  print("\nsubmodules")
  for name, status in data["submodules"].items():
    print(f"  {name:<28} {status}")

  for key, title in (("backend_probe", "JAX backend probe"), ("render_probe", "EGL render probe")):
    if key not in data:
      continue
    result = data[key]
    status = "ok" if result["ok"] else "FAILED"
    print(f"\n{title}: {status}")
    if result["detail"]:
      print(f"  {result['detail']}")


if __name__ == "__main__":
  args = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  args.add_argument("--model", help="Include resolved model settings and validation problems")
  args.add_argument("--set", action="append", metavar="PATH=VALUE")
  args.add_argument("--probes", action="store_true",
                    help="Also run the JAX backend and EGL render probes (each a "
                         f"subprocess, {PROBE_TIMEOUT}s timeout).")
  args.add_argument("--json", action="store_true", help="Machine-readable output.")
  kwargs = args.parse_args()
  if kwargs.set and not kwargs.model:
    args.error("--set requires --model")

  data = report(run_probes=kwargs.probes)
  if kwargs.model:
    from barl_jax.config.toolkit import parse_set_overrides
    from barl_jax.config.tree import ConfigTree, validate
    tree = ConfigTree(overrides=parse_set_overrides(kwargs.set))
    data["model"] = tree.resolved(kwargs.model)
    data["config_problems"] = validate(kwargs.model, tree)

  if kwargs.json:
    print(json.dumps(data, indent=2, sort_keys=True))
  else:
    print_text(data)
    if "model" in data:
      print(json.dumps({k: data[k] for k in ("model", "config_problems")}, indent=2))

  sys.exit(1 if required_probes_failed(data) or data.get("config_problems") else 0)
