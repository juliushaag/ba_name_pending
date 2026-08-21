#!/usr/bin/env python3

import argparse
import csv
import subprocess
from pathlib import Path
from itertools import product


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
        help="Generate the manifest but don't submit the jobs.",
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
    # Write manifest
    # ------------------------------------------------------------

    run_dir = Path("slurm_runs")
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = run_dir / "experiments.csv"

    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["algo", "env"],
        )
        writer.writeheader()
        writer.writerows(experiments)

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
    print(f"  Manifest: {manifest}")
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

    last_index = len(experiments) - 1

    cmd = [
        "sbatch",
        f"--array=0-{last_index}%{args.max_parallel}",
        f"--export=ALL,SEEDS={args.seeds},MANIFEST={manifest}",
        "scripts/run_rl.slurm",
    ]

    print()
    print("Submitting:")
    print(" ".join(cmd))
    print()

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()