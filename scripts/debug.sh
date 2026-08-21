#!/bin/bash

#SBATCH --job-name=ml_job_dev              # name shown in squeue
#SBATCH --partition=dev_gpu_a100_il        # dev_* queues: short walltime, quick turnaround
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32                 # CPU cores for env stepping etc. (nodes have 96 cores total)
#SBATCH --gres=gpu:1                       # number of GPUs (nodes have 4x GPU each, request only what you need)
#SBATCH --mem=64gb                         # host RAM (H100 nodes have up to 768 GiB total)

#SBATCH --output=logs/%x/%j/term.out
#SBATCH --error=logs/%x/%j/term.err

#SBATCH --signal=B:USR1@300

#SBATCH --time=30                          # HH:MM:SS, check current queue limits with `sinfo` / scontrol

# Same shape as run_sweep.sh, on the dev queue with a 30-minute cap: for
# checking that a config actually runs on the cluster before spending a real
# allocation on it. Add --validate below for the cheapest possible check (one
# rollout chunk + one train step per {algo, env}, no eval or checkpointing).

source scripts/slurm_common.sh

slurm_header

uv run --link-mode=copy python scripts/run_sweep.py \
    --algos SDAC \
    --envs HumanoidRun \
    --seeds 5 \
    --sweep_tag "$RUN_ID" \
    --sweep_dir "$SWEEP_DIR"

slurm_footer
