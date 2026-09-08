#!/bin/bash

#SBATCH --job-name=rl
#SBATCH --partition=gpu_a100_il,gpu_h100_il,gpu_h100

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

#SBATCH --time=10:00:00

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

# Belt-and-braces duplicate of online_learning.py's faulthandler.enable(): a
# fatal signal in a C library (EGL, mujoco, XLA) otherwise kills the process with
# nothing on stderr at all.
export PYTHONFAULTHANDLER=1

# jax otherwise takes 75% of the device up front. The offscreen GL context
# mujoco creates lazily on the first render() -- 1920x1080 with 4x MSAA for
# dog.xml, the only DMC model that asks for one -- has to fit in what is left.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.6}"

# `--export=ALL` hands this task whatever the submitting shell happened to hold,
# so where jax runs is decided here rather than there: a JAX_PLATFORMS=cpu left
# over from a config check or exported by hand trains the run on the CPU, ~160x
# slower per gradient step and looking exactly like a hang. The job asked for a
# GPU; it uses it. online_learning.py refuses to start without one.
unset JAX_PLATFORMS JAX_PLATFORM_NAME

# The array job id: shared by every task of one submission, so it is also the
# manifest's `run_id` column and the sweep tag wandb filters on.
RUN_ID="$SLURM_ARRAY_JOB_ID"
RESULT_DIR="${RESULT_DIR:-${SWEEPS_ROOT:-$PWD/sweeps}/$SWEEP_NAME}"
SCRATCH="$TMPDIR/sweep-$SLURM_JOB_ID"
MANIFEST="${MANIFEST:-$RESULT_DIR/experiments.csv}"

# The run works on node-local scratch; this copies it out on *every* exit path,
# so a walltime kill (see --signal above) or a crash still leaves whatever
# finished. The run's own directory under $RESULT_DIR already exists -- it is
# where submit.py put its config.yaml -- so this merges into it rather than
# replacing it.
save_results() {
  exit_code=$?
  [ -d "$SCRATCH" ] || return 0
  mkdir -p "$RESULT_DIR"

  for run in "$SCRATCH"/*/; do
    [ -d "$run" ] || continue
    dest="$RESULT_DIR/$(basename "$run")"
    mkdir -p "$dest"
    cp -a "$run." "$dest/" || true

    # What slurm knows and the run does not: whether this was a walltime kill, a
    # cgroup OOM (the limit is --mem above, *not* the node's RAM -- which is why
    # a node-level memory graph can look flat right through one), or a raw
    # signal. Nothing asked it before sweep 2026-08-26_17-59-33 lost a task to an
    # unexplained kill.
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
# --signal=B:USR1@300 above sends USR1 to *this* shell 300s before the wall, and
# bash's default action for it is to terminate -- so without this trap the run
# died silently and the EXIT trap saved whatever happened to be on disk. Forward
# it to python, whose faulthandler dumps every thread's stack before it goes.
on_usr1() {
  echo "=== USR1 at $(date): walltime in <300s, forwarding to $TRAIN_PID ==="
  kill -USR1 "$TRAIN_PID" 2>/dev/null || true
}
trap on_usr1 USR1

uv run python -m barl_jax.launch.worker \
    --config "$RESULT_DIR/$CONFIG" \
    --run-dir "$SCRATCH/$TAG" &
TRAIN_PID=$!

# `wait` is interruptible, which is the whole point: a foreground command would
# not let the shell run on_usr1 until it finished. The cost is that a handled
# signal makes `wait` return 128+n straight away while the child is still going,
# so this waits again until the child is actually gone -- otherwise USR1 would
# end the job 300s early instead of just dumping a stack.
set +e
wait "$TRAIN_PID"; STATUS=$?
while [ "$STATUS" -gt 128 ] && kill -0 "$TRAIN_PID" 2>/dev/null; do
  wait "$TRAIN_PID"; STATUS=$?
done
set -e

echo ""
echo "=================================================="
echo "Results:      $RESULT_DIR"
echo "Job finished: $(date)"
echo "Exit status:  $STATUS"
echo "=================================================="

exit "$STATUS"
