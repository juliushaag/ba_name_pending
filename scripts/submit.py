#!/usr/bin/env python3

"""Launch an experiment sweep -- the one entry point for running anything.

    uv run python scripts/submit.py --algos SAC DIME --envs HalfCheetah --seeds 5
    uv run python scripts/submit.py --algos SAC --envs HalfCheetah \
        --set model.components.critic.size=large
    uv run python scripts/submit.py --grid config/experiments/sweep_grid.yaml --dry-run
    uv run python scripts/submit.py --algos SAC --envs HalfCheetah --executor local --validate

A "sweep" here is any set of runs, including one. The grid is a cross product
over named axes -- `algo`/`env`/`seed`, plus any number of `model.<dotted path>`
axes and `run.<key>` axes -- declared on the command line or in a `--grid` spec
file, with an `exclude:` list for pruning. Because the axes are just overrides,
no config file is ever edited to run a sweep.

This script expands that grid, writes each run's self-contained config, and hands
the list to an executor: **slurm** where there is a scheduler, **tmux** where
there is not, **local** in the foreground when you ask. The grid is fully
expanded before anything is submitted, so all three backends are only "run these
N commands, M at a time" -- which is why an arbitrary model axis works the same
on the cluster as it does here.
"""

import argparse
import datetime
import os
from pathlib import Path

import yaml

import grid
from executors import EXECUTORS, SWEEP_NAME_FORMAT, SWEEPS_DIR, resolve


def check_config(algos: list[str], envs: list[str]) -> list[str]:
    """Everything an array of 40 tasks would otherwise discover 40 times, at the
    point where it is still one error message: unregistered env or algo names,
    and anything `model_config.validate()` rejects (an override that lands on no
    slot, a kwarg no component accepts, a slot missing its `size:`)."""
    # Imported here rather than at module scope so `--help` and a --dry-run of a
    # bad matrix don't pay for jax and mujoco. Resolution only: no env is built,
    # no weights are allocated, and nothing here touches a device.
    #
    # Nothing may be left in os.environ that wasn't already there: the executors
    # pass this process's environment on to the runs (sbatch's `--export=ALL`,
    # tmux's `-e PATH`). MUJOCO_GL is the one exception, and only because egl is
    # what the training scripts set for themselves anyway.
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")   # a login node has no GPU to spend

    import barl_jax  # noqa: F401  -- the import is what populates the registries
    from barl_jax import model_config
    from barl_jax.env import Environment

    known_envs = set(Environment.available())
    problems = [f"{env!r} is not a registered environment"
                for env in envs if env not in known_envs]

    for algo in algos:
        try:
            problems += [f"{algo}: {p}" for p in model_config.validate(algo)]
        except Exception as exc:  # an unresolvable name raises rather than reporting
            problems.append(f"{algo}: {type(exc).__name__}: {exc}")

    return problems


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Axes: algo/env/seed, plus model.<dotted path> (a model override, e.g. "
               "model.components.critic.size) and run.<key> (a run-config key, e.g. "
               "run.utd_ratio). Seeds are not an axis -- --seeds N vmaps N of them inside "
               "each run, so `--algos SAC DIME --seeds 5` is 2 runs of 5 seeds, not 10 runs.")

    what = parser.add_argument_group("what to run")
    what.add_argument("--algos", nargs="+",
                      help="`algo` axis, e.g. SAC DIME AMDP. Required unless --grid declares one.")
    what.add_argument("--envs", nargs="+",
                      help="`env` axis, e.g. HalfCheetah CheetahRun. Required unless --grid "
                           "declares one.")
    what.add_argument("--set", action="append", metavar="AXIS=VALUE", dest="sets",
                      help="Pin one axis to a single value across the sweep, e.g. "
                           "--set model.components.critic.size=large --set run.batch_size=512. "
                           "Repeatable.")
    what.add_argument("--grid", default=None,
                      help="YAML sweep spec: {base: <experiment config>, axes: {name: [values]}, "
                           "exclude: [{axis: value, ...}]}. Combined with any --algos/--envs/--set "
                           "given, which win per axis.")
    what.add_argument("--config", default=None,
                      help="Base experiment config providing `defaults:` for every run (default: "
                           "--grid's `base:`, else config/experiments/online_learning.yaml). Run "
                           "keys themselves are declared in config/run/defaults.yaml and per-env "
                           "overrides in config/run/envs.yaml.")
    what.add_argument("--seeds", type=int, default=None,
                      help="Seeds per run, vmapped inside the one process. Overrides `num_seeds` "
                           "in config/run/defaults.yaml, which is otherwise the only place the "
                           "seed count is declared.")

    where = parser.add_argument_group("where it runs")
    where.add_argument("--executor", choices=("auto", *EXECUTORS), default="auto",
                       help="`auto` (default) picks slurm when `sbatch` is on PATH, tmux when "
                            "only tmux is, and local otherwise -- so the same command works on "
                            "the cluster and on a workstation. `local` runs the sweep in the "
                            "foreground on this machine and is never chosen automatically.")
    where.add_argument("--max-parallel", type=int, default=None,
                       help="Runs in flight at once (Slurm's --array %%N). Default: 4 under slurm, "
                            "one per detected GPU under tmux/local. Setting it above the GPU count "
                            "is how you deliberately put more than one run on a GPU.")
    where.add_argument("--sweep-dir", default=None,
                       help="Where this sweep's tree goes (default: sweeps/<timestamp>).")

    how = parser.add_argument_group("how")
    how.add_argument("--validate", action="store_true",
                     help="Smoke-test every {algo x env} instead of training: one rollout chunk + "
                          "one train_step + one optimizer step each, no eval/checkpointing/wandb. "
                          "Collapses every other axis to its first value, and streams each run's "
                          "output.")
    how.add_argument("--debug", action="store_true",
                     help="Smoke-test submission: one run at a time and a 30-minute wall clock cap "
                          "(--max-parallel is ignored; under slurm it also means the dev queue). "
                          "The runs are real, just cut short -- whatever finished is still "
                          "collected when the cap kills them.")
    how.add_argument("--skip-check", action="store_true",
                     help="Don't resolve and validate every algorithm's model config first. Only "
                          "worth it if the check itself is in the way -- a bad config fails every "
                          "task identically.")
    how.add_argument("--dry-run", action="store_true",
                     help="Print the runs the grid expands to and exit, without writing or "
                          "submitting anything.")

    args = parser.parse_args()
    if args.seeds is not None and args.seeds < 1:
        parser.error("--seeds must be >= 1")
    if args.max_parallel is not None and args.max_parallel < 1:
        parser.error("--max-parallel must be >= 1")
    return args


def main():
    args = parse_args()

    base, axes, exclude = grid.build_axes(grid=args.grid, algos=args.algos,
                                          envs=args.envs, sets=args.sets)
    if args.validate:
        # One run per {algo, env}: every other axis collapses to its first value.
        axes = {k: (v if k in ("algo", "env") else v[:1]) for k, v in axes.items()}

    combos = grid.build_combos(axes, exclude)
    if not combos:
        raise SystemExit("grid is empty -- every combination was excluded")

    num_seeds = args.seeds if args.seeds is not None else grid.default_num_seeds()
    tags = grid.name_combos(combos, axes, num_seeds)

    executor = resolve(args.executor)
    max_parallel = 1 if args.debug else (args.max_parallel or executor.default_parallelism)

    sweep_name = datetime.datetime.now().strftime(SWEEP_NAME_FORMAT)  # noqa: DTZ005
    sweep_dir = Path(args.sweep_dir or (SWEEPS_DIR / sweep_name)).absolute()

    # ---- what we're about to do -------------------------------------------
    print()
    print(f"Sweep:    {grid.describe(combos, axes, num_seeds)}")
    print(f"Axes:     " + ", ".join(f"{k}={v}" for k, v in axes.items()))
    if exclude:
        print(f"Exclude:  {exclude}")
    print(f"Executor: {executor.name}" + (" (auto)" if args.executor == "auto" else ""))
    print(f"Parallel: {max_parallel}")
    if args.debug:
        print(f"Debug:    {executor.describe_debug()}")
    if args.validate:
        print(f"Validate: one rollout chunk + one train step per run, nothing logged")
    print(f"Dir:      {sweep_dir}")
    print()
    for index, tag in enumerate(tags):
        print(f"  [{index:03d}] {tag}")

    if args.dry_run:
        print()
        print("Dry run: nothing written, nothing submitted.")
        return

    # ---- check before it costs queue time ---------------------------------
    if args.skip_check:
        print("\nSkipping the config check (--skip-check).")
    else:
        print("\nChecking model configs ...", flush=True)
        problems = check_config(sorted({c["algo"] for c in combos}),
                                sorted({c["env"] for c in combos}))
        if problems:
            print("\nRefusing to submit -- unusable config:")
            for problem in problems:
                print(f"  {problem}")
            raise SystemExit(1)
        print("  OK")

    # ---- expand the grid onto disk ----------------------------------------
    base_config_path = args.config or base or grid.DEFAULT_BASE_CONFIG
    with open(base_config_path) as f:
        defaults = dict((yaml.safe_load(f) or {}).get("defaults") or {})

    if args.validate:
        defaults["validate"] = True
    # Threaded through `defaults` so every run of the sweep is one filter in wandb.
    defaults["sweep_tag"] = sweep_name

    sweep_dir.mkdir(parents=True, exist_ok=True)
    runs = grid.materialize(combos, tags, defaults=defaults, sweep_dir=str(sweep_dir),
                            num_seeds=args.seeds)

    head = {"sweep_tag": sweep_name, "created": sweep_name, "command": grid.sweep_command(),
            "base_config": os.path.abspath(base_config_path), "executor": executor.name,
            "seeds": num_seeds, "validate": bool(args.validate), "debug": bool(args.debug),
            "axes": {k: list(v) for k, v in axes.items()}, "exclude": exclude or None}
    grid.write_sweep_yaml(str(sweep_dir), head, runs)

    # ---- hand off ----------------------------------------------------------
    report = executor.submit(runs, sweep_name, sweep_dir,
                             max_parallel=max_parallel, debug=args.debug,
                             stream=args.validate)

    if executor.blocking:
        grid.write_sweep_yaml(str(sweep_dir), head, runs)

    print()
    width = max(len(label) for label in report)
    for label, value in report.items():
        print(f"  {label + ':':<{width + 1}} {value}")

    if executor.blocking and any(r.exit_code for r in runs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
