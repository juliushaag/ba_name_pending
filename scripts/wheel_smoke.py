"""Installed-wheel smoke test for the standalone `barl_jax` library.

Builds `src/barl_jax`, installs the wheel into a throwaway venv, and runs the
check *from that venv, in a directory outside the repo* -- so a source-path
leak (the repo root or `src/` landing on `sys.path`) cannot hide a package
missing from the wheel. Building the wheel is necessary but not sufficient:
this is the "does it actually work once installed" check the plan calls for.

    uv run python scripts/wheel_smoke.py

Requires `uv`. Leaves nothing behind: wheel output and venv are both in a
temporary directory that is removed on exit (pass --keep to inspect them).
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Runs inside the installed venv, with cwd outside the repo (see main()).
# Exercises package/config import, packaged defaults, registry resolution,
# and a tiny policy construction -- the things a broken setuptools package
# list or missing package-data silently drops from the wheel.
CHECK_SCRIPT = r"""
import os
import sys

assert not any(str(p).endswith("src") or str(p) == "" for p in sys.path), \
    f"source path leaked into sys.path: {sys.path!r}"

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import gymnasium as gym

import barl_jax  # noqa: F401 -- populates the policy/component registries
from barl_jax.config.tree import ConfigTree
from barl_jax.config.toolkit import CONFIG_DIR
from barl_jax.registration import available_components
from barl_jax.policy import Policy
from barl_jax.env import EnvStats

assert CONFIG_DIR.exists(), f"packaged defaults missing: {CONFIG_DIR}"
assert list(CONFIG_DIR.glob("**/*.yaml")), f"no packaged default YAML under {CONFIG_DIR}"

tree = ConfigTree()
models = tree.available_models()
assert models, "no models resolvable from packaged defaults"

components = available_components()
assert components, "component registry is empty"

class FakeEnv:
  observation_space = gym.spaces.Box(-np.inf, np.inf, (5,), np.float32)
  action_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)

stats = EnvStats(
  obs_shape=(5,), obs_dim=5, act_discrete=False, act_dim=3,
  act_low=np.full(3, -1.0, np.float32), act_high=np.full(3, 1.0, np.float32),
  max_episode_steps=16, return_min=-10.0, return_max=10.0)

name = sorted(models)[0]
policy = Policy.make(name, FakeEnv(), env_stats=stats, seed=0)
assert policy is not None

print(f"wheel smoke ok: {len(models)} models, {len(components)} components, "
      f"constructed {name!r}")
"""


def sh(cmd: list[str], **kwargs) -> None:
  print(f"$ {' '.join(cmd)}")
  subprocess.run(cmd, check=True, **kwargs)


if __name__ == "__main__":
  args = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  args.add_argument("--keep", action="store_true", help="Don't delete the temp workspace.")
  kwargs = args.parse_args()

  workspace = Path(tempfile.mkdtemp(prefix="barl-wheel-smoke-"))
  try:
    wheel_dir = workspace / "dist"
    sh(["uv", "build", "src/barl_jax", "--out-dir", str(wheel_dir)], cwd=ROOT)
    wheels = sorted(wheel_dir.glob("*.whl"))
    if not wheels:
      raise SystemExit(f"uv build produced no wheel in {wheel_dir}")
    wheel = wheels[-1]

    venv = workspace / "venv"
    sh(["uv", "venv", str(venv)])
    sh(["uv", "pip", "install", "--python", str(venv / "bin" / "python"), str(wheel)])

    check_dir = workspace / "outside_repo"
    check_dir.mkdir()
    check_file = check_dir / "check.py"
    check_file.write_text(CHECK_SCRIPT)
    sh([str(venv / "bin" / "python"), str(check_file)], cwd=check_dir)
  finally:
    if kwargs.keep:
      print(f"workspace kept at {workspace}")
    else:
      shutil.rmtree(workspace, ignore_errors=True)

  print("wheel_smoke passed")
