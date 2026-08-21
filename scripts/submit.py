#!/usr/bin/env python3

import argparse
import contextlib
import csv
import datetime
import os
import subprocess
from pathlib import Path
from itertools import product

# The same directory -- and the same timestamp naming -- `run_sweep.py` uses for
# a local sweep, so a submission ends up as one sweeps/<timestamp>/ folder that
# reads exactly like a local one: this manifest, plus the per-combo run
# directories every array task copies back into it.
SWEEPS_DIR = Path("sweeps")
MANIFEST_NAME = "experiments.csv"
SWEEP_NAME_FORMAT = "%Y-%m-%d_%H-%M-%S"


@contextlib.contextmanager
def _env(**overrides):
    """Set env vars for the duration of a block and put the old values back."""
    previous = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def check_config(algos: list[str], envs: list[str]) -> list[str]:
    """Everything an array of 40 tasks would otherwise discover 40 times, at the
    point where it is still one error message: unregistered env or algo names,
    and anything `model_config.validate()` rejects (an override that lands on no
    slot, a kwarg no component accepts, a mistyped YAML value)."""
    # Imported here rather than at module scope so `--help` and a --dry-run of a
    # bad matrix don't pay for jax and mujoco. Resolution only: no env is built
    # and no weights are allocated.
    #
    # The two env vars are restored afterwards, and that is load-bearing rather
    # than tidy: sbatch below inherits this process's environment and hands it
    # to every array task via `--export=ALL`. Leaking JAX_PLATFORMS=cpu out of
    # here puts the whole submission on CPU -- which does not fail, it just
    # trains ~50x slower and looks like a hang.
    with _env(JAX_PLATFORMS="cpu",           # a login node has no GPU to spend
              MUJOCO_GL="egl"):              # else mujoco reaches for glfw and warns on DISPLAY
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


# --debug overrides for run_rl.slurm's own #SBATCH lines: sbatch's CLI wins over
# them, so the job script stays the single description of a *real* run.
DEBUG_PARTITION = "gpu_a100_short"
DEBUG_TIME      = "00:30:00"
DEBUG_JOB_NAME  = "rl_dev"


def main():
    parser = argparse.ArgumentParser(
        description="Submit an RL experiment matrix as a Slurm job array."
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
        default=4,
        help="Maximum number of Slurm array jobs running simultaneously.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help=f"Submit to the dev queue for a smoke test: {DEBUG_PARTITION}, a "
             f"{DEBUG_TIME} walltime cap and one task at a time (--max-parallel is "
             f"ignored). The runs are real ones, just cut short -- run_rl.slurm copies "
             f"whatever finished back to ~/results when the cap kills it.",
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

    if args.max_parallel < 1:
        parser.error("--max-parallel must be >= 1")

    max_parallel = 1 if args.debug else args.max_parallel

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
    print(f"  Max parallel jobs: {max_parallel}")
    if args.debug:
        print(f"  Debug: {DEBUG_PARTITION}, {DEBUG_TIME} cap, one task at a time")
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
    # Dry run
    # ------------------------------------------------------------

    if args.dry_run:
        print()
        print("Dry run: nothing submitted.")
        return

    # ------------------------------------------------------------
    # Submit Slurm array
    # ------------------------------------------------------------

    # Held so the manifest -- whose name and `run_id` column are the array job id
    # Slurm only hands out at submission -- is on disk before any task reads it.
    last_index = len(experiments) - 1

    # Named here, not per task: every task of the array has to agree on which
    # folder it copies its results into, and each would otherwise take its own
    # clock reading.
    sweep_name = datetime.datetime.now().strftime(SWEEP_NAME_FORMAT)  # noqa: DTZ005

    cmd = ["sbatch", "--parsable", "--hold",
           f"--array=0-{last_index}%{max_parallel}",
           f"--export=ALL,SEEDS={args.seeds},SWEEP_NAME={sweep_name}"]

    if args.debug:
        cmd += [f"--partition={DEBUG_PARTITION}", f"--time={DEBUG_TIME}",
                f"--job-name={DEBUG_JOB_NAME}"]

    cmd.append("scripts/run_rl.slurm")

    print()
    print("Submitting:")
    print(" ".join(cmd))
    print()

    submitted = subprocess.run(cmd, check=True, capture_output=True, text=True)
    run_id = submitted.stdout.strip().split(";")[0]

    try:
        manifest = write_manifest(sweep_name, run_id, experiments)
        subprocess.run(["scontrol", "release", run_id], check=True)
    except Exception:
        subprocess.run(["scancel", run_id], check=False)
        raise

    print(f"  Run ID:    {run_id}")
    print(f"  Sweep dir: {manifest.parent}")
    print(f"  Logs:      logs/{DEBUG_JOB_NAME if args.debug else 'rl'}/{run_id}_<task>/")
    print(f"  Sweep tag: {run_id}")


def write_manifest(sweep_name: str, run_id: str, experiments: list[dict]) -> Path:
    sweep_dir = SWEEPS_DIR / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)

    manifest = sweep_dir / MANIFEST_NAME

    # `lineterminator`, not csv's "\r\n" default: run_rl.slurm reads these rows
    # with `read`, which would carry the CR into the last field.
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run_id", "algo", "env"], lineterminator="\n")
        writer.writeheader()
        writer.writerows({"run_id": run_id, **e} for e in experiments)

    return manifest


if __name__ == "__main__":
    main()
