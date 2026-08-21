#!/usr/bin/env python3

"""Where a submitted matrix actually executes.

Two backends behind one interface, so scripts/submit.py is only ever a CLI: it
builds the matrix, validates it, and hands it to whichever `Executor` is
available. Each owns the whole of its own mechanism -- how a task is described,
how `--debug` is expressed, how the run id is come by, and when the manifest can
be written -- because those differ in ways that do not usefully generalise:

* SlurmExecutor's run id only exists once `sbatch` has accepted the job, so it
  submits held, writes the manifest, then releases.
* TmuxExecutor has no scheduler to ask, so the sweep's own name is the id and
  the manifest is written before anything starts.

What they share is this module's sweep layout: one `sweeps/<timestamp>/` folder
per submission, holding `experiments.csv` and, once the runs land, one directory
per `{algo, env}`. A finished sweep is the same shape either way, and the same
shape `run_sweep.py` leaves behind when run directly.
"""

import csv
import os
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path

# The same directory -- and the same timestamp naming -- `run_sweep.py` uses for
# a local sweep, so a submission reads exactly like one.
SWEEPS_DIR = Path("sweeps")
MANIFEST_NAME = "experiments.csv"
SWEEP_NAME_FORMAT = "%Y-%m-%d_%H-%M-%S"

DEBUG_TIME_LABEL = "00:30:00"
DEBUG_SECONDS = 30 * 60


def write_manifest(sweep_name: str, run_id: str, experiments: list[dict]) -> Path:
    """The submitted matrix, one row per task. run_rl.slurm reads its own row out
    of this by array index; tmux_pool.py reads the whole thing as its queue."""
    sweep_dir = SWEEPS_DIR / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)
    manifest = sweep_dir / MANIFEST_NAME

    # `lineterminator`, not csv's "\r\n" default: run_rl.slurm reads these rows
    # with `read`, which would carry the CR into the last field.
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run_id", "algo", "env"], lineterminator="\n")
        writer.writeheader()
        writer.writerows({"run_id": run_id, **e} for e in experiments)

    return manifest


class Executor(ABC):
    """One submission mechanism. `submit` returns the lines submit.py prints, so
    a backend reports whatever is worth knowing about itself -- a Slurm job id,
    a tmux session to attach to -- without submit.py knowing which is which."""

    name: str
    #: Runs in flight when the caller does not say (Slurm's --array %N).
    default_parallelism: int

    @staticmethod
    @abstractmethod
    def available() -> bool:
        """Is this mechanism usable on this machine?"""

    @abstractmethod
    def submit(self, experiments: list[dict], sweep_name: str, *,
               seeds: int, max_parallel: int, debug: bool) -> dict[str, str]:
        ...

    def describe_debug(self) -> str:
        return f"{DEBUG_TIME_LABEL} cap, one task at a time"


class SlurmExecutor(Executor):
    name = "slurm"
    default_parallelism = 4

    # --debug overrides for run_rl.slurm's own #SBATCH lines: sbatch's CLI wins
    # over them, so the job script stays the single description of a *real* run.
    DEBUG_PARTITION = "gpu_a100_short"
    DEBUG_JOB_NAME = "rl_dev"

    @staticmethod
    def available() -> bool:
        return shutil.which("sbatch") is not None

    def describe_debug(self) -> str:
        return f"{self.DEBUG_PARTITION}, {super().describe_debug()}"

    def submit(self, experiments, sweep_name, *, seeds, max_parallel, debug):
        # Held so the manifest -- whose `run_id` column is the array job id Slurm
        # only hands out at submission -- is on disk before any task reads it.
        cmd = ["sbatch", "--parsable", "--hold",
               f"--array=0-{len(experiments) - 1}%{max_parallel}",
               f"--export=ALL,SEEDS={seeds},SWEEP_NAME={sweep_name}"]

        if debug:
            cmd += [f"--partition={self.DEBUG_PARTITION}", f"--time={DEBUG_TIME_LABEL}",
                    f"--job-name={self.DEBUG_JOB_NAME}"]

        cmd.append("scripts/run_rl.slurm")

        print()
        print("Submitting:")
        print(" ".join(cmd))
        print()

        submitted = subprocess.run(cmd, check=True, capture_output=True, text=True)
        run_id = submitted.stdout.strip().split(";")[0]

        try:
            manifest = write_manifest(sweep_name, run_id, experiments)
            subprocess.run(["scontrol", "release", run_id], check=True)
        except Exception:
            subprocess.run(["scancel", run_id], check=False)
            raise

        job_name = self.DEBUG_JOB_NAME if debug else "rl"
        return {"Run ID": run_id,
                "Sweep dir": str(manifest.parent),
                "Logs": f"logs/{job_name}/{run_id}_<task>/",
                "Sweep tag": run_id,
                "Cancel": f"scancel {run_id}"}


class TmuxExecutor(Executor):
    """One detached session per sweep: scripts/tmux_pool.py in window 0 owning
    the queue, and one window per run that it opens as a slot frees up. Detached
    and returning immediately is the point -- `sbatch` does not block either."""

    name = "tmux"

    @staticmethod
    def available() -> bool:
        return shutil.which("tmux") is not None

    @property
    def default_parallelism(self) -> int:
        # One run per GPU, mirroring a Slurm task's own GPU allocation.
        from run_sweep import detect_gpus
        return len(detect_gpus())

    def submit(self, experiments, sweep_name, *, seeds, max_parallel, debug):
        # There is no scheduler to hand out an id and nothing has to wait for
        # one, so the sweep's own name is it: folder, manifest column and wandb
        # tag all agree.
        run_id = sweep_name
        manifest = write_manifest(sweep_name, run_id, experiments)

        session = f"barl-{sweep_name}"
        manager = [sys.executable, "scripts/tmux_pool.py",
                   "--session", session,
                   "--sweep-dir", str(manifest.parent),
                   "--run-id", run_id,
                   "--seeds", str(seeds),
                   "--max-parallel", str(max_parallel)]
        if debug:
            manager += ["--time-limit", str(DEBUG_SECONDS)]

        # A window inherits the tmux *server's* environment, not this shell's,
        # and the server may have been started long before this venv existed.
        # PATH decides which `uv` the runs get, so it is threaded through
        # explicitly -- the tmux answer to sbatch's --export=ALL. tmux_pool.py
        # passes it on again to each run window.
        subprocess.run(["tmux", "new-session", "-d", "-s", session, "-n", "manager",
                        "-c", str(Path.cwd()), "-e", f"PATH={os.environ['PATH']}",
                        *manager], check=True)

        attach = (f"tmux switch-client -t {session}" if os.environ.get("TMUX")
                  else f"tmux attach -t {session}")

        return {"Session": session,
                "Sweep dir": str(manifest.parent),
                "Attach": attach,
                "Sweep tag": run_id,
                "Cancel": f"tmux kill-session -t {session}"}


EXECUTORS = {cls.name: cls for cls in (SlurmExecutor, TmuxExecutor)}


def resolve(choice: str) -> Executor:
    """`auto` is what makes one command work in both places: Slurm where there is
    a scheduler, tmux where there is only this box."""
    if choice == "auto":
        for cls in EXECUTORS.values():
            if cls.available():
                return cls()
        raise SystemExit("--executor auto: found neither `sbatch` nor `tmux` on PATH")

    cls = EXECUTORS[choice]
    if not cls.available():
        raise SystemExit(f"--executor {choice}: not usable here (nothing to run it with on PATH)")
    return cls()
