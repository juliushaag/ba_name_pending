# Shared setup for every Slurm job in this repo. Sourced, never executed --
# sbatch only ships the submitted script, and Slurm starts the job in the
# submit directory, which is the repo root: the same assumption every
# `uv run python scripts/...` line in those jobs already makes.
#
# What it establishes, so the job scripts only have to say what they run:
#
#   RUN_ID      one id for the whole submission (see below)
#   SWEEP_DIR   node-local scratch that run_sweep.py writes its per-run
#               directories into -- config, log and checkpoints per run
#   SWEEP_NAME  the sweep's folder name, one timestamp per submission
#   RESULT_DIR  sweeps/<SWEEP_NAME>, where that tree is copied on every exit path
#
#   slurm_header / slurm_footer   the identical banner every job prints
#
# Every task of one submission copies into the *same* sweeps/<SWEEP_NAME>, next
# to the experiments.csv submit.py wrote there -- so a finished cluster
# submission is one folder of per-combo run directories, indistinguishable from
# what `run_sweep.py` alone leaves behind locally.

set -euo pipefail

export PYTHONUNBUFFERED=1

# The array's job id where there is an array, so every task of one submit.py
# call shares it -- it is the manifest's `run_id` column, the `--sweep_tag`
# every run is filtered by in wandb, and the results directory's name. A job
# submitted without an array falls back to its own id.
RUN_ID="${RUN_ID:-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local-$$}}}"

# Named the way run_sweep.py names a local sweep -- a timestamp, not a job id --
# so a cluster sweep and a local one sit side by side in sweeps/ and read the
# same. submit.py exports one name for a whole array (every task copies into it);
# a single job takes its own clock reading.
SWEEP_NAME="${SWEEP_NAME:-$(date +%Y-%m-%d_%H-%M-%S)}"

RESULT_DIR="${SWEEPS_ROOT:-$PWD/sweeps}/$SWEEP_NAME"

SWEEP_DIR="${TMPDIR:-/tmp}/sweep-${SLURM_JOB_ID:-$$}"

# `mv`, not `cp`: slurmstepd keeps writing through its open descriptor, so a
# rename within the same filesystem relocates the log *and* keeps collecting
# what comes after this -- the footer, and whatever slurmstepd appends about a
# cancellation. A copy would snapshot the file just short of the part that
# explains why a job died. Falls back to cp if the two are on different mounts.
save_slurm_log() {
  local dest="$1" path
  # scontrol, not the #SBATCH pattern above: sbatch's command line can override
  # --output/--error, and scontrol reports where the file actually went.
  for path in $(scontrol show job "${SLURM_JOB_ID:-}" 2>/dev/null \
                | tr ' ' '\n' | grep -E '^Std(Out|Err)=' | cut -d= -f2- | sort -u); do
    [ -f "$path" ] || continue
    mv "$path" "$dest/" 2>/dev/null || cp "$path" "$dest/" 2>/dev/null || true
  done
}

# Copied on *every* exit path, not just a clean one: a walltime kill or a
# crashed run still leaves whatever the sweep had finished by then.
save_results() {
  [ -d "$SWEEP_DIR" ] || return 0
  mkdir -p "$RESULT_DIR"

  if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    cp -a "$SWEEP_DIR/." "$RESULT_DIR/" || true
    # One job, several runs: no single run owns the job's log.
    save_slurm_log "$RESULT_DIR"
    return 0
  fi

  # An array task ran a one-combo grid, so its sweep.yaml describes that single
  # run -- it goes inside the run's own directory rather than colliding at the
  # shared root with every other task's, and so does the job's own stdout/stderr.
  # experiments.csv is the root index.
  for run in "$SWEEP_DIR"/*/; do
    [ -d "$run" ] || continue
    cp -a "$run" "$RESULT_DIR/" || true
    if [ -f "$SWEEP_DIR/sweep.yaml" ]; then
      cp -a "$SWEEP_DIR/sweep.yaml" "$RESULT_DIR/$(basename "$run")/" || true
    fi
    save_slurm_log "$RESULT_DIR/$(basename "$run")"
  done
}
trap save_results EXIT

slurm_header() {
  echo "=================================================="
  echo "Job started: $(date)"
  echo "Hostname:    $(hostname)"
  echo "Run ID:      $RUN_ID"
  echo "Sweep:       $SWEEP_NAME"
  echo "Job ID:      ${SLURM_JOB_ID:-n/a}"
  echo "Array task:  ${SLURM_ARRAY_TASK_ID:-n/a}"
  echo "Partition:   ${SLURM_JOB_PARTITION:-n/a}"
  echo "GPU:         ${CUDA_VISIBLE_DEVICES:-none}"
  echo "Sweep dir:   $SWEEP_DIR"
  echo "Results:     $RESULT_DIR"
  echo "=================================================="
  # The training stack is JAX, so whether *JAX* sees the GPU is the thing worth
  # failing on here. Falling back to CPU does not error anywhere downstream --
  # it just trains ~50x slower for the whole allocation and reads as a hang.
  uv run python -c "
import sys, jax
print('JAX devices:', jax.devices())
if jax.default_backend() == 'cpu':
    sys.exit('ERROR: JAX sees no GPU. Refusing to train on CPU -- check '
             'JAX_PLATFORMS and CUDA_VISIBLE_DEVICES in the job environment.')
"
  echo ""
}

slurm_footer() {
  echo ""
  echo "=================================================="
  echo "Results:      $RESULT_DIR"
  echo "Job finished: $(date)"
  echo "=================================================="
}
