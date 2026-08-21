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

# One array task per row of the manifest scripts/submit.py wrote. Submit with
# that script, not with sbatch directly -- it is what names the manifest after
# the array job id and fills in the `run_id` column.

source scripts/slurm_common.sh

SEEDS="${SEEDS:-5}"
MANIFEST="${MANIFEST:-slurm_runs/$RUN_ID.csv}"

slurm_header

# ------------------------------------------------------------
# This array element's experiment
# ------------------------------------------------------------

# +2: one for the header row, one because array indices start at 0.
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$MANIFEST")

IFS=',' read -r ROW_RUN_ID ALGO ENV <<< "$LINE"

if [ "$ROW_RUN_ID" != "$RUN_ID" ]; then
    echo "WARNING: $MANIFEST row $SLURM_ARRAY_TASK_ID says run id $ROW_RUN_ID, job says $RUN_ID"
fi

echo "Experiment:"
echo "  Manifest: $MANIFEST"
echo "  Algo:     $ALGO"
echo "  Env:      $ENV"
echo "  Seeds:    $SEEDS"
echo ""

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------

uv run python scripts/run_sweep.py \
    --algos "$ALGO" \
    --envs "$ENV" \
    --seeds "$SEEDS" \
    --sweep_tag "$RUN_ID" \
    --sweep_dir "$SWEEP_DIR"

slurm_footer
