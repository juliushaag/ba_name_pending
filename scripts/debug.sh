#!/bin/bash
#SBATCH --job-name=ml_job_dev                 # name shown in squeue
#SBATCH --partition=dev_gpu_a100_il              # bwUniCluster 3.0 GPU queues:
                                           #   dev_gpu_h100  -> quick tests, short walltime, 4x H100 nodes
                                           #   gpu_h100      -> production runs, 4x H100 nodes (AMD EPYC 9454)
                                           #   gpu_h100_short-> shorter max walltime, higher priority
                                           #   gpu_a100_il / gpu_h100_il -> Ice Lake + A100/H100 nodes
                                           #   gpu_a100_short-> Cascade Lake + 4x A100
                                           #   gpu_mi300     -> AMD Instinct MI300A APU nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32                 # CPU cores for data loading etc. (nodes have 96 cores total)
#SBATCH --gres=gpu:1                      # number of GPUs (nodes have 4x GPU each, request only what you need)
#SBATCH --mem=64gb                        # host RAM (H100 nodes have up to 768 GiB total)
#SBATCH --output=logs/%x/%j/term.out           # stdout -> logs/ml_job_<jobid>.out
#SBATCH --error=logs/%x/%j/term.err            # stderr
#SBATCH --signal=B:USR1@300
#SBATCH --time=30                          # HH:MM:SS, check current queue limits with `sinfo` / scontrol

set -euo pipefail
trap 'cp -r "$TMPDIR/results" ~/results/$SLURM_JOB_ID/ 2>/dev/null' EXIT

export PYTHONUNBUFFERED=1

echo "== Job started on $(hostname) at $(date) =="
echo "Job ID: $SLURM_JOB_ID | Partition: $SLURM_JOB_PARTITION | GPUs: $CUDA_VISIBLE_DEVICES"

# --- Run your script -------------------------------------------------
# Runs sequentially, algo by algo; --seeds N is N seeds vmapped inside each run.
uv run --link-mode=copy python scripts/run_sweep.py \
    --algos SDAC \
    --envs HumanoidRun \
    --seeds 5

# --- Copy results back to persistent storage --------------------------
mkdir -p ~/results/$SLURM_JOB_ID
cp -r "$TMPDIR/results" ~/results/$SLURM_JOB_ID/

echo "== Job finished at $(date) =="