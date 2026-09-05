"""The documented fast local check: config lint plus the boundary tests that
catch config/registry/launcher drift, without paying for the full JAX
training suite. Wraps existing commands rather than re-implementing them --
`scripts/lint_config.py` and `pytest` stay independently runnable.

    JAX_PLATFORMS=cpu uv run python scripts/fastcheck.py

The full suite (`uv run pytest tests/ -q`) is still the one CI and pre-merge
verification run; this is for the edit/check loop while writing config,
registry, or launcher changes.
"""

import os
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Config precedence, the registry/component contract, and the three ways a
# run is launched -- cheap (no training loop) and where config/registry
# drift actually shows up. tests/test_modularity.py exercises checkpoint
# save/load subprocesses and belongs to the full suite, not this loop.
BOUNDARY_TESTS = [
  "tests/test_config_tree.py",
  "tests/test_configs.py",
  "tests/test_registry.py",
  "tests/test_launcher_devices.py",
  "tests/test_run_validate.py",
]


def run(cmd: list[str]) -> int:
  print(f"$ {' '.join(cmd)}")
  return subprocess.run(cmd, cwd=ROOT).returncode


if __name__ == "__main__":
  lint_status = run([sys.executable, "scripts/lint_config.py"])
  test_status = run([sys.executable, "-m", "pytest", "-q", *BOUNDARY_TESTS])

  ok = lint_status == 0 and test_status == 0
  print(f"\nfastcheck {'passed' if ok else 'FAILED'} "
        f"(lint exit {lint_status}, tests exit {test_status})")
  sys.exit(0 if ok else 1)
