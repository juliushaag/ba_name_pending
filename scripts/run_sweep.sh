#!/bin/bash

#SBATCH --job-name=ml_job                  # name shown in squeue
#SBATCH --partition=gpu_a100_il            # bwUniCluster 3.0 GPU queues:
                                           #   dev_gpu_h100  -> quick tests, short walltime, 4x H100 nodes
                                           #   gpu_h100      -> production runs, 4x H100 nodes (AMD EPYC 9454)
                                           #   gpu_h100_short-> shorter max walltime, higher priority
                                           #   gpu_a100_il / gpu_h100_il -> Ice Lake + A100/H100 nodes
                                           #   gpu_a100_short-> Cascade Lake + 4x A100
                                           #   gpu_mi300     -> AMD Instinct MI300A APU nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2                       # number of GPUs (nodes have 4x GPU each, request only what you need)

#SBATCH --output=logs/%x/%j/term.out
#SBATCH --error=logs/%x/%j/term.err

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ulqho@student.kit.edu

#SBATCH --signal=B:USR1@300

#SBATCH --time=4:00:00                     # HH:MM:SS, check current queue limits with `sinfo` / scontrol

# One job, one grid edited in place below. For a matrix that wants its own
# array task (and its own GPU) per {algo, env}, use scripts/submit.py +
# scripts/run_rl.slurm instead.

source scripts/slurm_common.sh

slurm_header

# The grid runs sequentially, algo by algo, one GPU each (run_sweep.py detects
# them); --seeds N is N seeds vmapped inside each run, not N processes.
uv run python scripts/run_sweep.py \
    --algos NCLQL SDAC \
    --envs WalkerRun \
    --seeds 5 \
    --sweep_tag "$RUN_ID" \
    --sweep_dir "$SWEEP_DIR"

slurm_footer
