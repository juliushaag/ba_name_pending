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

# One array task per row of the manifest scripts/submit.py wrote -- i.e. one per
# point of the grid, whatever the grid varied. Submit with that script, never
# sbatch directly: it creates sweeps/<SWEEP_NAME>/, expands the sweep into one
# self-contained config.yaml per run, writes the manifest, and exports
# SWEEP_NAME. Every task of a submission writes into that same folder, so a
# finished submission is one directory of per-run results.
#
# This script knows nothing about algorithms, environments or axes: a manifest
# row points at a config, and running it is one command.

set -euo pipefail

export PYTHONUNBUFFERED=1

# The array job id: shared by every task of one submission, so it is also the
# manifest's `run_id` column and the sweep tag wandb filters on.
RUN_ID="$SLURM_ARRAY_JOB_ID"
RESULT_DIR="${SWEEPS_ROOT:-$PWD/sweeps}/$SWEEP_NAME"
SCRATCH="$TMPDIR/sweep-$SLURM_JOB_ID"
MANIFEST="${MANIFEST:-$RESULT_DIR/experiments.csv}"

# The run works on node-local scratch; this copies it out on *every* exit path,
# so a walltime kill (see --signal above) or a crash still leaves whatever
# finished. The run's own directory under $RESULT_DIR already exists -- it is
# where submit.py put its config.yaml -- so this merges into it rather than
# replacing it.
save_results() {
  [ -d "$SCRATCH" ] || return 0
  mkdir -p "$RESULT_DIR"

  for run in "$SCRATCH"/*/; do
    [ -d "$run" ] || continue
    dest="$RESULT_DIR/$(basename "$run")"
    mkdir -p "$dest"
    cp -a "$run." "$dest/" || true

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
echo "Scratch:     $SCRATCH"
echo "Results:     $RESULT_DIR"
echo "=================================================="
uv run python -c "import jax; print('JAX devices:', jax.devices())"

# +2: one for the header row, one because array indices start at 0.
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$MANIFEST")
IFS=',' read -r ROW_INDEX ROW_RUN_ID TAG CONFIG <<< "${LINE%$'\r'}"

if [ "$ROW_RUN_ID" != "$RUN_ID" ]; then
    echo "WARNING: $MANIFEST row $SLURM_ARRAY_TASK_ID says run id $ROW_RUN_ID, job says $RUN_ID"
fi

echo ""
echo "Run:    $TAG  (row $ROW_INDEX of $MANIFEST)"
echo "Config: $RESULT_DIR/$CONFIG"
echo ""

# --run-dir redirects this run's output to node-local scratch; the config itself
# stays the shared one submit.py wrote, so the run is reproducible from it.
uv run python scripts/online_learning.py \
    --config "$RESULT_DIR/$CONFIG" \
    --run-dir "$SCRATCH/$TAG"

echo ""
echo "=================================================="
echo "Results:      $RESULT_DIR"
echo "Job finished: $(date)"
echo "=================================================="
