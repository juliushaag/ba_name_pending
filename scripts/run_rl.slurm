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
# that script, never sbatch directly: it creates sweeps/<SWEEP_NAME>/, writes the
# manifest there, and exports SEEDS and SWEEP_NAME. Every task of a submission
# writes into that same folder, so a finished submission is one directory of
# per-combo runs -- exactly what `run_sweep.py` leaves behind locally.

set -euo pipefail

export PYTHONUNBUFFERED=1

SEEDS="${SEEDS:-5}"
# The array job id: shared by every task of one submission, so it is also the
# manifest's `run_id` column and the `--sweep_tag` wandb filters on.
RUN_ID="$SLURM_ARRAY_JOB_ID"
RESULT_DIR="${SWEEPS_ROOT:-$PWD/sweeps}/$SWEEP_NAME"
SWEEP_DIR="$TMPDIR/sweep-$SLURM_JOB_ID"
MANIFEST="${MANIFEST:-$RESULT_DIR/experiments.csv}"

# run_sweep.py works on node-local scratch; this copies it out on *every* exit
# path, so a walltime kill (see --signal above) or a crash still leaves whatever
# finished. The task's sweep.yaml describes its one run, so it is filed inside
# that run's directory rather than colliding at the shared root with every other
# task's -- experiments.csv is the root index.
save_results() {
  [ -d "$SWEEP_DIR" ] || return 0
  mkdir -p "$RESULT_DIR"

  for run in "$SWEEP_DIR"/*/; do
    [ -d "$run" ] || continue
    dest="$RESULT_DIR/$(basename "$run")"
    cp -a "$run" "$RESULT_DIR/" || true
    [ -f "$SWEEP_DIR/sweep.yaml" ] && cp -a "$SWEEP_DIR/sweep.yaml" "$dest/"

    # `mv`, not `cp`: slurmstepd keeps writing through its open descriptor, so a
    # rename relocates the log *and* keeps collecting what comes after -- the
    # footer, and whatever slurmstepd appends about a cancellation. Asking
    # scontrol rather than rebuilding the #SBATCH pattern above, since sbatch's
    # command line can override --output/--error.
    for log in $(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
                 | tr ' ' '\n' | grep -E '^Std(Out|Err)=' | cut -d= -f2- | sort -u); do
      [ -f "$log" ] && mv "$log" "$dest/"
    done
  done
}
trap save_results EXIT

echo "=================================================="
echo "Job started: $(date)"
echo "Hostname:    $(hostname)"
echo "Run ID:      $RUN_ID"
echo "Sweep:       $SWEEP_NAME"
echo "Job ID:      $SLURM_JOB_ID"
echo "Array task:  $SLURM_ARRAY_TASK_ID"
echo "Partition:   $SLURM_JOB_PARTITION"
echo "GPU:         $CUDA_VISIBLE_DEVICES"
echo "Sweep dir:   $SWEEP_DIR"
echo "Results:     $RESULT_DIR"
echo "=================================================="
uv run python -c "import jax; print('JAX devices:', jax.devices())"

# +2: one for the header row, one because array indices start at 0.
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$MANIFEST")
IFS=',' read -r ROW_RUN_ID ALGO ENV <<< "${LINE%$'\r'}"

if [ "$ROW_RUN_ID" != "$RUN_ID" ]; then
    echo "WARNING: $MANIFEST row $SLURM_ARRAY_TASK_ID says run id $ROW_RUN_ID, job says $RUN_ID"
fi

echo ""
echo "Experiment: $ALGO / $ENV, $SEEDS seeds  (manifest: $MANIFEST)"
echo ""

uv run python scripts/run_sweep.py \
    --algos "$ALGO" \
    --envs "$ENV" \
    --seeds "$SEEDS" \
    --sweep_tag "$RUN_ID" \
    --sweep_dir "$SWEEP_DIR"

echo ""
echo "=================================================="
echo "Results:      $RESULT_DIR"
echo "Job finished: $(date)"
echo "=================================================="
