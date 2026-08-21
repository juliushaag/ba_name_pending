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
#   RESULT_DIR  where that tree is copied to on every exit path
#
#   slurm_header / slurm_footer   the identical banner every job prints

set -euo pipefail

export PYTHONUNBUFFERED=1

# The array's job id where there is an array, so every task of one submit.py
# call shares it -- it is the manifest's `run_id` column, the `--sweep_tag`
# every run is filtered by in wandb, and the results directory's name. A job
# submitted without an array falls back to its own id.
RUN_ID="${RUN_ID:-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local-$$}}}"

RESULT_DIR="$HOME/results/$RUN_ID${SLURM_ARRAY_TASK_ID:+/$SLURM_ARRAY_TASK_ID}"

SWEEP_DIR="${TMPDIR:-/tmp}/sweep-${SLURM_JOB_ID:-$$}"

# Copied on *every* exit path, not just a clean one: a walltime kill or a
# crashed run still leaves whatever the sweep had finished by then.
save_results() {
  if [ -d "$SWEEP_DIR" ]; then
    mkdir -p "$RESULT_DIR"
    cp -a "$SWEEP_DIR/." "$RESULT_DIR/" || true
  fi
}
trap save_results EXIT

slurm_header() {
  echo "=================================================="
  echo "Job started: $(date)"
  echo "Hostname:    $(hostname)"
  echo "Run ID:      $RUN_ID"
  echo "Job ID:      ${SLURM_JOB_ID:-n/a}"
  echo "Array task:  ${SLURM_ARRAY_TASK_ID:-n/a}"
  echo "Partition:   ${SLURM_JOB_PARTITION:-n/a}"
  echo "GPU:         ${CUDA_VISIBLE_DEVICES:-none}"
  echo "Sweep dir:   $SWEEP_DIR"
  echo "Results:     $RESULT_DIR"
  echo "=================================================="
  # The training stack is JAX, so whether *JAX* sees the GPU is the thing worth
  # failing on here -- a job that silently falls back to CPU wastes its walltime.
  uv run python -c "import jax; print('JAX devices:', jax.devices())"
  echo ""
}

slurm_footer() {
  echo ""
  echo "=================================================="
  echo "Results:      $RESULT_DIR"
  echo "Job finished: $(date)"
  echo "=================================================="
}
