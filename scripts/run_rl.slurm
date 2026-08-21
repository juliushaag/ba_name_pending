#!/bin/bash

#SBATCH --job-name=rl
#SBATCH --partition=gpu_a100_il

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16                 # one core per few subprocess envs -- rollouts are
                                           # AsyncVectorEnv, num_seeds x num_envs processes deep
#SBATCH --gres=gpu:1
#SBATCH --mem=96gb                         # host RAM: the replay buffer is per seed, on the host

#SBATCH --output=logs/%x/%A_%a/term.out
#SBATCH --error=logs/%x/%A_%a/term.err

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ulqho@student.kit.edu

#SBATCH --signal=B:USR1@300

#SBATCH --time=5:00:00

# One array task per row of the manifest scripts/submit.py wrote. Submit with
# that script, not with sbatch directly -- it is what names the manifest after
# the array job id and fills in the `run_id` column.

source scripts/slurm_common.sh || {
    echo "ERROR: scripts/slurm_common.sh not found -- sbatch this from the repo root." >&2
    exit 1
}

SEEDS="${SEEDS:-5}"
MANIFEST="${MANIFEST:-$RESULT_DIR/experiments.csv}"

slurm_header

# ------------------------------------------------------------
# This array element's experiment
# ------------------------------------------------------------

# +2: one for the header row, one because array indices start at 0.
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$MANIFEST")

IFS=',' read -r ROW_RUN_ID ALGO ENV <<< "${LINE%$'\r'}"

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
