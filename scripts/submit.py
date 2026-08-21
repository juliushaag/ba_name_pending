#!/usr/bin/env python3

import argparse
import csv
import subprocess
from pathlib import Path
from itertools import product

RUN_DIR = Path("slurm_runs")


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
        "--dry-run",
        action="store_true",
        help="Print the matrix but don't submit the jobs.",
    )

    args = parser.parse_args()

    if args.seeds < 1:
        parser.error("--seeds must be >= 1")

    if args.max_parallel < 1:
        parser.error("--max-parallel must be >= 1")

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
    print(f"  Max parallel jobs: {args.max_parallel}")
    print()

    for i, experiment in enumerate(experiments):
        print(
            f"  [{i:03d}] "
            f"{experiment['algo']} / "
            f"{experiment['env']}"
        )

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

    cmd = [
        "sbatch",
        "--parsable",
        "--hold",
        f"--array=0-{last_index}%{args.max_parallel}",
        f"--export=ALL,SEEDS={args.seeds}",
        "scripts/run_rl.slurm",
    ]

    print()
    print("Submitting:")
    print(" ".join(cmd))
    print()

    submitted = subprocess.run(cmd, check=True, capture_output=True, text=True)
    run_id = submitted.stdout.strip().split(";")[0]

    try:
        manifest = write_manifest(run_id, experiments)
        subprocess.run(["scontrol", "release", run_id], check=True)
    except Exception:
        subprocess.run(["scancel", run_id], check=False)
        raise

    print(f"  Run ID:   {run_id}")
    print(f"  Manifest: {manifest}")
    print(f"  Logs:     logs/rl/{run_id}_<task>/")
    print(f"  Sweep tag: {run_id}")


def write_manifest(run_id: str, experiments: list[dict]) -> Path:
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    manifest = RUN_DIR / f"{run_id}.csv"

    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run_id", "algo", "env"])
        writer.writeheader()
        writer.writerows({"run_id": run_id, **e} for e in experiments)

    return manifest


if __name__ == "__main__":
    main()
