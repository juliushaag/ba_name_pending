#!/usr/bin/env python3

"""Expand a grid and launch it through Slurm, tmux, or local workers.

    uv run python scripts/submit.py --algos SAC DIME --envs HalfCheetah --seeds 5
    uv run python scripts/submit.py --grid src/barl_jax/defaults/experiments/sweep_grid.yaml --dry-run
    uv run python scripts/submit.py --algos SAC --envs HalfCheetah --executor local --validate

A sweep can contain one run. --set pins model.<path> or run.<key> axes; --grid
adds cross products and exclusions. Configs are frozen before submission."""

import argparse
import datetime
import subprocess
import sys
from pathlib import Path

from barl_jax.launch import grid
from barl_jax.launch.executors import EXECUTORS, SWEEP_NAME_FORMAT, SWEEPS_DIR, resolve


def submit_sweep(executor, runs, sweep_name, sweep_dir, head, *,
                 max_parallel, debug=False, stream=False):
    """Submit prepared runs, record blocking results, and propagate failures."""
    report = executor.submit(runs, sweep_name, sweep_dir,
                             max_parallel=max_parallel, debug=debug, stream=stream)
    if executor.blocking:
        grid.write_sweep_yaml(str(sweep_dir), head, runs)
    print()
    width = max(map(len, report), default=0)
    for label, value in report.items():
        print(f"  {label + ':':<{width + 1}} {value}")
    if executor.blocking and any(r.exit_code != 0 for r in runs):
        raise SystemExit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Axes: algo/env/seed, plus model.<dotted path> (a model override, e.g. "
               "model.components.critic.tier) and run.<key> (a run-config key, e.g. "
               "run.utd_ratio). Seeds are not an axis -- --seeds N vmaps N of them inside "
               "each run, so `--algos SAC DIME --seeds 5` is 2 runs of 5 seeds, not 10 runs.")

    what = parser.add_argument_group("what to run")
    what.add_argument("--config", help="Replay an experiment config in the foreground")
    what.add_argument("--algos", nargs="+",
                      help="`algo` axis, e.g. SAC DIME AMDP. Required unless --grid declares one.")
    what.add_argument("--envs", nargs="+",
                      help="`env` axis, e.g. HalfCheetah CheetahRun. Required unless --grid "
                           "declares one.")
    what.add_argument("--set", action="append", metavar="AXIS=VALUE", dest="sets",
                      help="Pin one axis to a single value across the sweep, e.g. "
                           "--set model.components.critic.tier=large --set run.batch_size=512. "
                           "Repeatable.")
    what.add_argument("--grid", default=None,
                      help="YAML sweep spec: {axes: {name: [values]}, "
                           "exclude: [{axis: value, ...}]}. Combined with any --algos/--envs/--set "
                           "given, which win per axis.")
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
                       help="Runs in flight at once (Slurm's --array %%N). Default: 8 under slurm, "
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
    if args.config:
        if (args.algos or args.envs or args.grid or args.sets or args.validate or args.debug
                or args.executor != "auto" or args.max_parallel or args.sweep_dir or args.dry_run):
            raise SystemExit("--config replays directly; only --seeds may override the saved config")
        command = [sys.executable, "-m", "barl_jax.launch.worker", "--config", args.config]
        if args.seeds is not None:
            command += ["--seeds", str(args.seeds)]
        raise SystemExit(subprocess.call(command))

    axes, exclude = grid.build_axes(grid=args.grid, algos=args.algos,
                                          envs=args.envs, sets=args.sets)
    if args.seeds is not None and "run.num_seeds" in axes:
        axes["run.num_seeds"] = [args.seeds]
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
    print("Axes:     " + ", ".join(f"{k}={v}" for k, v in axes.items()))
    if exclude:
        print(f"Exclude:  {exclude}")
    print(f"Executor: {executor.name}" + (" (auto)" if args.executor == "auto" else ""))
    print(f"Parallel: {max_parallel}")
    if args.debug:
        print(f"Debug:    {executor.describe_debug()}")
    if args.validate:
        print("Validate: one rollout chunk + one train step per run, nothing logged")
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
        problems = grid.check_config(sorted({c["algo"] for c in combos}),
                                sorted({c["env"] for c in combos}))
        if problems:
            print("\nRefusing to submit -- unusable config:")
            for problem in problems:
                print(f"  {problem}")
            raise SystemExit(1)
        print("  OK")

    # ---- expand the grid onto disk ----------------------------------------
    runs, head = grid.prepare_sweep(
        combos, tags, sweep_dir=str(sweep_dir),
        num_seeds=args.seeds, validate=args.validate, sweep_name=sweep_name,
        executor_name=executor.name, seeds=num_seeds,
        head_extra={"validate": bool(args.validate), "debug": bool(args.debug),
                   "axes": {k: list(v) for k, v in axes.items()}, "exclude": exclude or None})

    submit_sweep(executor, runs, sweep_name, sweep_dir, head,
                 max_parallel=max_parallel, debug=args.debug, stream=args.validate)


if __name__ == "__main__":
    main()
