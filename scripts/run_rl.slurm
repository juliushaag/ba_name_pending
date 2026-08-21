#!/bin/bash

#SBATCH --job-name=rl
#SBATCH --partition=gpu_a100_il

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

#SBATCH --output=logs/%x/%A_%a/term.out
#SBATCH --error=logs/%x/%A_%a/term.err

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ulqho@student.kit.edu

#SBATCH --signal=B:USR1@300

#SBATCH --time=5:00:00

set -euo pipefail

export PYTHONUNBUFFERED=1

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

SEEDS="${SEEDS:-5}"
RUN_ID="${RUN_ID:-$SLURM_ARRAY_JOB_ID}"
MANIFEST="${MANIFEST:-slurm_runs/$RUN_ID.csv}"

echo "=================================================="
echo "Job started: $(date)"
echo "Hostname:    $(hostname)"
echo "Run ID:      $RUN_ID"
echo "Job ID:      $SLURM_JOB_ID"
echo "Array ID:    $SLURM_ARRAY_TASK_ID"
echo "Partition:   $SLURM_JOB_PARTITION"
echo "GPU:         $CUDA_VISIBLE_DEVICES"
echo "Seeds:       $SEEDS"
echo "Manifest:    $MANIFEST"
echo "=================================================="

# ------------------------------------------------------------
# Get this array element's experiment
# ------------------------------------------------------------

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$MANIFEST")

IFS=',' read -r RUN_ID ALGO ENV <<< "$LINE"

echo ""
echo "Experiment:"
echo "  Algo:   $ALGO"
echo "  Env:    $ENV"
echo "  Seeds:  $SEEDS"
echo "  Tag:    $RUN_ID"
echo ""

# ------------------------------------------------------------
# Sanity check
# ------------------------------------------------------------

uv run python -c "
import torch

print('CUDA available:', torch.cuda.is_available())

if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('CUDA:', torch.version.cuda)
"

# ------------------------------------------------------------
# Run experiment
# ------------------------------------------------------------

mkdir -p "$TMPDIR/results"

uv run python scripts/run_sweep.py \
    --algos "$ALGO" \
    --envs "$ENV" \
    --seeds "$SEEDS" \
    --sweep_tag "$RUN_ID"

# ------------------------------------------------------------
# Copy results to persistent storage
# ------------------------------------------------------------

RESULT_DIR="$HOME/results/$RUN_ID/$SLURM_ARRAY_TASK_ID"

mkdir -p "$RESULT_DIR"

if [ -d "$TMPDIR/results" ]; then
    cp -r "$TMPDIR/results/." "$RESULT_DIR/"
fi

echo ""
echo "Results:"
echo "  $RESULT_DIR"

echo ""
echo "=================================================="
echo "Job finished: $(date)"
echo "=================================================="