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
vmapped process: one CUDA context, one compiled step, one set of envs. A seed run
that way is bit-identical to the same seed run alone. The loop itself lives in
barl_jax.trainer -- this file is the CLI around it.
"""

import argparse
import os
import warnings

# Must be set before the first `import mujoco` (transitively, via `barl_jax`
# below) -- mujoco picks its GL backend at import time, so setting this later
# leaves it locked onto GLFW, which hard-crashes env.render() on a headless box.
os.environ.setdefault("MUJOCO_GL", "egl")

warnings.filterwarnings("ignore", module="glfw")  # dm_control still probes GLFW at import regardless
# wandb's own sentry client trips half a dozen DeprecationWarnings against the
# wandb API at login, once per process -- nothing here can act on them.
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"wandb\..*")

from barl_jax.config import load_config
from barl_jax.trainer import setup, train, validate

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "..", "config", "experiments",
                              "online_learning.yaml")


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

  runs = load_config(kwargs.config)
  for i, cfg in enumerate(runs):
    if kwargs.run_dir is not None:
      cfg["run_dir"] = os.path.abspath(kwargs.run_dir)
    num_seeds = kwargs.seeds or cfg["num_seeds"]
    seeds = f", {num_seeds} seeds vmapped" if num_seeds > 1 else ""
    print(f"=== Run {i + 1}/{len(runs)}: {cfg['model']} on {cfg['env']}{seeds} ===")

    only_validate = bool(cfg["validate"])
    run = setup(cfg, num_seeds, validate_only=only_validate)
    (validate if only_validate else train)(run)
