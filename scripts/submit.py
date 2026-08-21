#!/usr/bin/env python3

import argparse
import datetime
import os
from itertools import product

import executors
from executors import SWEEP_NAME_FORMAT


def check_config(algos: list[str], envs: list[str]) -> list[str]:
    """Everything an array of 40 tasks would otherwise discover 40 times, at the
    point where it is still one error message: unregistered env or algo names,
    and anything `model_config.validate()` rejects (an override that lands on no
    slot, a kwarg no component accepts, a mistyped YAML value)."""
    # Imported here rather than at module scope so `--help` and a --dry-run of a
    # bad matrix don't pay for jax and mujoco. Resolution only: no env is built,
    # no weights are allocated, and nothing here touches a device.
    #
    # Nothing may be left in os.environ that wasn't already there: both executors
    # pass this process's environment on to the runs (sbatch's `--export=ALL`,
    # tmux's `-e PATH`). MUJOCO_GL is the one exception, and only because egl is
    # what the training scripts set for themselves anyway.
    os.environ.setdefault("MUJOCO_GL", "egl")

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


def main():
    parser = argparse.ArgumentParser(
        description="Launch an RL experiment matrix: as a Slurm job array where there is a "
                    "scheduler, or as one tmux window per run where there is not."
    )

    parser.add_argument(
        "--executor",
        choices=("auto", "slurm", "tmux"),
        default="auto",
        help="Where the runs execute. `auto` (default) picks slurm when `sbatch` is on PATH "
             "and tmux otherwise, so the same command works on the cluster and on a workstation.",
    )

    parser.add_argument(
        "--algos",
        nargs="+",
        required=True,
        help="Algorithms to run, e.g. NCLQL SDAC",
    )

    parser.add_argument(
        "--envs",
        nargs="+",
        required=True,
        help="Environments to run, e.g. WalkerRun HopperRun",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Number of vmapped seeds per run (default: 5).",
    )

    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Runs in flight at once (Slurm's --array %%N). Default: 4 under slurm, one per "
             "detected GPU under tmux. Setting it above the GPU count under tmux is how you "
             "deliberately put more than one run on a GPU.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Smoke-test submission: one task at a time and a 30-minute wall clock cap "
             "(--max-parallel is ignored; under slurm it also means the dev queue). The runs "
             "are real ones, just cut short -- whatever finished is still collected when the "
             "cap kills them.",
    )

    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Don't resolve and validate every algorithm's model config first. Only worth it "
             "if the check itself is in the way -- a bad config fails every array task.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the matrix but don't submit the jobs.",
    )

    args = parser.parse_args()

    if args.seeds < 1:
        parser.error("--seeds must be >= 1")

    if args.max_parallel is not None and args.max_parallel < 1:
        parser.error("--max-parallel must be >= 1")

    executor = executors.resolve(args.executor)

    if args.debug:
        max_parallel = 1
    elif args.max_parallel is not None:
        max_parallel = args.max_parallel
    else:
        max_parallel = executor.default_parallelism

    # ------------------------------------------------------------
    # Generate experiment matrix
    # ------------------------------------------------------------

    experiments = [
        {
            "algo": algo,
            "env": env,
        }
        for algo, env in product(args.algos, args.envs)
    ]

    if not experiments:
        raise RuntimeError("No experiments generated.")

    # ------------------------------------------------------------
    # Print what we're about to do
    # ------------------------------------------------------------

    print()
    print("Experiments:")
    print(f"  Algorithms: {', '.join(args.algos)}")
    print(f"  Environments: {', '.join(args.envs)}")
    print(f"  Number of experiments: {len(experiments)}")
    print(f"  Seeds per experiment: {args.seeds}")
    print(f"  Executor: {executor.name}"
          + (" (auto-detected)" if args.executor == "auto" else ""))
    print(f"  Max parallel jobs: {max_parallel}")
    if args.debug:
        print(f"  Debug: {executor.describe_debug()}")
    print()

    for i, experiment in enumerate(experiments):
        print(
            f"  [{i:03d}] "
            f"{experiment['algo']} / "
            f"{experiment['env']}"
        )

    # ------------------------------------------------------------
    # Check the config before it costs queue time
    # ------------------------------------------------------------

    if args.skip_check:
        print()
        print("Skipping the config check (--skip-check).")
    else:
        print()
        print("Checking model configs ...", flush=True)
        problems = check_config(args.algos, args.envs)
        if problems:
            print()
            print("Refusing to submit -- unusable config:")
            for problem in problems:
                print(f"  {problem}")
            raise SystemExit(1)
        print("  OK")

    # ------------------------------------------------------------
    # Hand off to the executor
    # ------------------------------------------------------------

    # Named here, not per task: every task of a submission has to agree on which
    # folder it writes into, and each would otherwise take its own clock reading.
    sweep_name = datetime.datetime.now().strftime(SWEEP_NAME_FORMAT)  # noqa: DTZ005

    if args.dry_run:
        print()
        print(f"Dry run: nothing submitted. Would have used {executor.name}, "
              f"sweeps/{sweep_name}/.")
        return

    report = executor.submit(experiments, sweep_name, seeds=args.seeds,
                             max_parallel=max_parallel, debug=args.debug)

    width = max(len(label) for label in report)
    for label, value in report.items():
        print(f"  {label + ':':<{width + 1}} {value}")


if __name__ == "__main__":
    main()
