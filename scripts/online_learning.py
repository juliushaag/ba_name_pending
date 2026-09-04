"""Runs one experiment config: the single task every executor launches.

**Not the entry point.** Everything -- a one-off run as much as a grid -- goes
through `scripts/submit.py`, which expands the sweep, writes each run's
self-contained config, and hands the list to an executor. This script is what
that executor then invokes per task, and it does exactly one thing: take a
config, run the {model, env} entries in it, in series.

It stays a working command for the one case where reaching past submit.py is the
point -- re-running a single finished run under a debugger, from the config it
already left behind:

    uv run python scripts/online_learning.py --config sweeps/<ts>/<tag>/config.yaml

`num_seeds` (config, or `--seeds`) runs N independently-initialized seeds as one
vmapped process, with `num_seeds: 1` as the degenerate case: one CUDA context,
one compiled step, one set of envs. A 1-seed run that way is bit-identical to the
same seed run alone. The loop itself lives in
barl_jax.trainer -- this file is the CLI around it.
"""

import argparse
import faulthandler
import os
import signal
import sys
import warnings

# Installed before jax and mujoco get a chance to fault: an EGL/mjr allocation
# failure inside the driver, or an XLA `Check failed`, kills the process with
# SIGSEGV/SIGABRT and no Python output at all -- which is exactly what sweep
# 2026-08-26_17-59-33's DIME/DogRun task left behind. `chain=True` keeps the
# normal handling (slurm's walltime USR1 still terminates) and only adds the dump.
faulthandler.enable(file=sys.stderr, all_threads=True)
for _sig in (signal.SIGUSR1, signal.SIGUSR2, signal.SIGTERM):
  faulthandler.register(_sig, file=sys.stderr, all_threads=True, chain=True)

# Must be set before the first `import mujoco` (transitively, via `barl_jax`
# below) -- mujoco picks its GL backend at import time, so setting this later
# leaves it locked onto GLFW, which hard-crashes env.render() on a headless box.
os.environ.setdefault("MUJOCO_GL", "egl")

warnings.filterwarnings("ignore", module="glfw")  # dm_control still probes GLFW at import regardless
# wandb's own sentry client trips half a dozen DeprecationWarnings against the
# wandb API at login, once per process -- nothing here can act on them.
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"wandb\..*")

import jax

from barl_jax.config import load_config
from barl_jax.trainer import setup, train, validate

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "..", "config", "experiments",
                              "online_learning.yaml")


def require_accelerator() -> None:
  """The last gate before a run costs anything, and the only one that holds
  whatever the cause -- a leaked JAX_PLATFORMS, a venv without the CUDA plugin,
  a node whose driver came back wrong. A run that lands on the CPU trains, logs
  and checkpoints exactly as it should, ~160x slower per gradient step, so
  nothing downstream can tell it from a slow algorithm: sweep 2026-08-23_17-54-53
  spent 10 GPU-hours reaching 2% of one run's steps. Better to fail in the first
  second than to be told 10 hours later."""
  if os.environ.get("BARL_ALLOW_CPU"):
    return

  devices = jax.devices()
  if any(device.platform in ("gpu", "tpu") for device in devices):
    return

  forced = {key: os.environ[key] for key in ("JAX_PLATFORMS", "JAX_PLATFORM_NAME")
            if key in os.environ}
  raise SystemExit(
    f"Refusing to start: jax has no GPU/TPU backend, only {devices}.\n"
    + (f"  {forced} is forcing it -- unset it and resubmit.\n" if forced else
       f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}; "
       "check the driver on this node and that jax[cuda13] is installed in this venv.\n")
    + "  BARL_ALLOW_CPU=1 runs on the CPU anyway.")


if __name__ == "__main__":
  args = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  args.add_argument("--config", default=DEFAULT_CONFIG,
                    help="Path to a YAML experiment config (see config/experiments/online_learning.yaml). "
                         "Lists one or more {model, env, ...} runs, executed in series.")
  args.add_argument("--seeds", type=int, default=None,
                    help="How many seeds to run vmapped in this process. "
                         "Overrides `num_seeds` in the config.")
  args.add_argument("--run-dir", default=None,
                    help="Where this run writes its checkpoints, overriding the config's "
                         "`run_dir`. A batch task points it at node-local scratch and copies "
                         "the tree back on exit, so the config itself stays the shared one.")
  kwargs = args.parse_args()

  require_accelerator()

  runs = load_config(kwargs.config)
  for i, cfg in enumerate(runs):
    if kwargs.run_dir is not None:
      cfg["run_dir"] = os.path.abspath(kwargs.run_dir)
    num_seeds = kwargs.seeds or cfg["num_seeds"]
    print(f"=== Run {i + 1}/{len(runs)}: {cfg['model']} on {cfg['env']}, "
          f"{num_seeds} seed{'s' if num_seeds != 1 else ''} vmapped ===")

    only_validate = bool(cfg["validate"])
    run = setup(cfg, num_seeds, validate_only=only_validate)
    (validate if only_validate else train)(run)
