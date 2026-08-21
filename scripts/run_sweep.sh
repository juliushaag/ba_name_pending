#!/bin/bash
#SBATCH --job-name=ml_job                 # name shown in squeue
#SBATCH --partition=gpu_a100_il              # bwUniCluster 3.0 GPU queues:
                                           #   dev_gpu_h100  -> quick tests, short walltime, 4x H100 nodes
                                           #   gpu_h100      -> production runs, 4x H100 nodes (AMD EPYC 9454)
                                           #   gpu_h100_short-> shorter max walltime, higher priority
                                           #   gpu_a100_il / gpu_h100_il -> Ice Lake + A100/H100 nodes
                                           #   gpu_a100_short-> Cascade Lake + 4x A100
                                           #   gpu_mi300     -> AMD Instinct MI300A APU nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2                      # number of GPUs (nodes have 4x GPU each, request only what you need)

#SBATCH --output=logs/%x/%j/term.out           # stdout -> logs/ml_job_<jobid>.out
#SBATCH --error=logs/%x/%j/term.err            # stderr

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ulqho@student.kit.edu
#SBATCH --signal=B:USR1@300

#SBATCH --time=4:00:00                   # HH:MM:SS, check current queue limits with `sinfo` / scontrol

set -euo pipefail
trap 'cp -r "$TMPDIR/results" ~/results/$SLURM_JOB_ID/ 2>/dev/null' EXIT

export PYTHONUNBUFFERED=1

echo "== Job started on $(hostname) at $(date) =="
echo "Job ID: $SLURM_JOB_ID | Partition: $SLURM_JOB_PARTITION | GPUs: $CUDA_VISIBLE_DEVICES"

# --- Sanity checks -------------------------------------------------
uv run python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# --- Run your script -------------------------------------------------
# Runs sequentially, algo by algo; --seeds N is N seeds vmapped inside each run.
uv run python scripts/run_sweep.py \
    --algos NCLQL SDAC \
    --envs WalkerRun \
    --seeds 5

# --- Copy results back to persistent storage --------------------------
mkdir -p ~/results/$SLURM_JOB_ID
cp -r "$TMPDIR/results" ~/results/$SLURM_JOB_ID/

echo "== Job finished at $(date) =="