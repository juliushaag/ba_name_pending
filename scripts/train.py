"""Run experiment config entries sequentially, with seeds vmapped per entry.

Normally launched through submit.py and its executors. To replay one saved run:

    uv run python scripts/submit.py --config sweeps/<timestamp>/<run>/config.yaml

Training lives in barl_jax.trainer. This CLI requires an accelerator unless
BARL_ALLOW_CPU is explicitly set."""

import argparse
import faulthandler
import os
import signal
import sys
import time
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

# On a shared multi-GPU node, mujoco.egl enumerates every EGL device on the
# node and silently uses the first one it can init a context on (mujoco/egl/
# __init__.py:create_initialized_egl_device_display) -- unlike CUDA, this
# enumeration ignores cgroup device isolation, so without a pin it can land on
# a GPU this job has no access to and render *something* that reads back as
# all-zero pixels: a black video with no error anywhere. SLURM_JOB_GPUS (set
# from --gres=gpu:1) is the node-wide physical index EGL's enumeration uses,
# not the possibly-remapped CUDA_VISIBLE_DEVICES -- same fix humanoid_bench's
# own tdmpc2/dreamerv3 runners apply, just under the env var mujoco.egl (not
# mujoco_py) actually reads.
if "MUJOCO_EGL_DEVICE_ID" not in os.environ:
  slurm_gpus = os.environ.get("SLURM_STEP_GPUS", os.environ.get("SLURM_JOB_GPUS"))
  if slurm_gpus:
    os.environ["MUJOCO_EGL_DEVICE_ID"] = slurm_gpus.split(",")[0]

warnings.filterwarnings("ignore", module="glfw")  # dm_control still probes GLFW at import regardless
# wandb's own sentry client trips half a dozen DeprecationWarnings against the
# wandb API at login, once per process -- nothing here can act on them.
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"wandb\..*")

import jax

from barl_jax.config import from_yaml
from barl_jax.trainer import setup, train, validate


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
  args.add_argument("--config", required=True,
                    help="Path to one YAML run mapping with required model and env selectors.")
  args.add_argument("--seeds", type=int, default=None,
                    help="How many seeds to run vmapped in this process. Overrides num_seeds.")
  args.add_argument("--run-dir", default=None,
                    help="Where this run writes checkpoints, overriding run_dir in the YAML.")
  kwargs = args.parse_args()

  if kwargs.seeds is not None and kwargs.seeds < 1:
    args.error("--seeds must be >= 1")

  require_accelerator()
  print(f"Accelerator: {jax.devices()}")
  config = from_yaml(kwargs.config)
  if kwargs.run_dir is not None:
    config = config.replace(run_dir=os.path.abspath(kwargs.run_dir))
  num_seeds = kwargs.seeds if kwargs.seeds is not None else config.num_seeds
  print(f"Loaded one run from {kwargs.config}")
  print(f"=== Run: {config.model} on {config.env}, "
        f"{num_seeds} seed{'s' if num_seeds != 1 else ''} vmapped ===")

  only_validate = bool(config.validate_config)
  t0 = time.perf_counter()
  run = setup(config, num_seeds, validate_only=only_validate)
  (validate if only_validate else train)(run)
  print(f"=== Run done in {(time.perf_counter() - t0) / 60:.1f} min ===")
