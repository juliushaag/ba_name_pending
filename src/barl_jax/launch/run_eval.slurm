#!/bin/bash

#SBATCH --job-name=eval
#SBATCH --partition=gpu_a100_short,gpu_h100_short

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16                 # one core per few subprocess envs -- eval's vector
                                           # env is num_envs processes deep, same as a rollout
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb                         # no replay buffer here -- one policy plus episodes

#SBATCH --output=logs/%x/%A_%a/term.out
#SBATCH --error=logs/%x/%A_%a/term.err

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ulqho@student.kit.edu

#SBATCH --signal=B:USR1@300

#SBATCH --time=00:30:00

# run_rl.slurm's sibling for evaluation: one array task per *batch* of
# checkpoints (scripts/submit.py --from-sweep groups several {algo, env}
# checkpoints per task on purpose -- eval is cheap enough that one task per
# checkpoint would waste a node's worth of scheduling overhead on work that
# finishes in seconds). Submit through that script, never sbatch directly --
# see run_rl.slurm's own header for the sweep-directory contract, shared here.
#
# Unlike run_rl.slurm, this never renders (scripts/evaluate.py's run_episodes
# call is always render=False), so there is no offscreen GL context to reserve
# device memory for and no MUJOCO_EGL_DEVICE_ID pinning to do at this level --
# evaluate.py's own top-of-file guard covers the case where some environment
# still needs mujoco initialized without ever rendering.

set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

unset JAX_PLATFORMS JAX_PLATFORM_NAME

RUN_ID="$SLURM_ARRAY_JOB_ID"
RESULT_DIR="${RESULT_DIR:-${SWEEPS_ROOT:-$PWD/sweeps}/$SWEEP_NAME}"
SCRATCH="$TMPDIR/sweep-$SLURM_JOB_ID"
MANIFEST="${MANIFEST:-$RESULT_DIR/experiments.csv}"

# Identical to run_rl.slurm's: the batch's own directory under $RESULT_DIR
# already exists (submit.py put its eval.yaml there), so this merges the
# scratch copy of results.json (plus a postmortem) into it rather than
# replacing it. See that script for why each exit path needs this.
save_results() {
  exit_code=$?
  [ -d "$SCRATCH" ] || return 0
  mkdir -p "$RESULT_DIR"

  for run in "$SCRATCH"/*/; do
    [ -d "$run" ] || continue
    dest="$RESULT_DIR/$(basename "$run")"
    mkdir -p "$dest"
    cp -a "$run." "$dest/" || true

    {
      echo "=== postmortem: exit $exit_code at $(date) ==="
      sacct -j "$SLURM_JOB_ID" --units=G \
        -o JobID%20,State,ExitCode,DerivedExitCode,MaxRSS,MaxVMSize,Elapsed,Timelimit,Reason 2>&1
      echo "--- cgroup memory ---"
      for f in /sys/fs/cgroup/memory.peak /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.events; do
        [ -r "$f" ] && { echo "$f:"; cat "$f"; }
      done
      echo "--- gpu ---"
      nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv 2>&1
    } >> "$dest/postmortem.txt" 2>&1

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

# +2: one for the header row, one because array indices start at 0. ROW_KIND
# is "eval" for every row of this manifest -- named because it is on every
# row (executors.py's MANIFEST_FIELDS), not because this script reads it.
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$MANIFEST")
IFS=',' read -r ROW_INDEX ROW_RUN_ID TAG CONFIG ROW_KIND <<< "${LINE%$'\r'}"

if [ "$ROW_RUN_ID" != "$RUN_ID" ]; then
    echo "WARNING: $MANIFEST row $SLURM_ARRAY_TASK_ID says run id $ROW_RUN_ID, job says $RUN_ID"
fi

echo ""
echo "Batch:  $TAG  (row $ROW_INDEX of $MANIFEST)"
echo "Config: $RESULT_DIR/$CONFIG"
echo ""

# --output redirects this batch's results.json to node-local scratch, same
# reasoning as run_rl.slurm's --run-dir. --signal=B:USR1@300 forwards to
# python so faulthandler dumps every thread's stack before the wall kills it.
on_usr1() {
  echo "=== USR1 at $(date): walltime in <300s, forwarding to $EVAL_PID ==="
  kill -USR1 "$EVAL_PID" 2>/dev/null || true
}
trap on_usr1 USR1

uv run python scripts/evaluate.py \
    --config "$RESULT_DIR/$CONFIG" \
    --output "$SCRATCH/$TAG/results.json" &
EVAL_PID=$!

set +e
wait "$EVAL_PID"; STATUS=$?
while [ "$STATUS" -gt 128 ] && kill -0 "$EVAL_PID" 2>/dev/null; do
  wait "$EVAL_PID"; STATUS=$?
done
set -e

echo ""
echo "=================================================="
echo "Results:      $RESULT_DIR"
echo "Job finished: $(date)"
echo "Exit status:  $STATUS"
echo "=================================================="

exit "$STATUS"
