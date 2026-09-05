#!/usr/bin/env python3

"""Where a submitted sweep actually executes.

Three backends behind one interface, so scripts/submit.py is only ever a CLI: it
expands the grid, writes every run's config, and hands the list to whichever
`Executor` is available.

The grid is already fully expanded by the time anything here runs (see grid.py),
so an executor's whole job is "run these N commands, at most M at a time". None
of them knows what an algorithm, an environment or an axis is -- which is why an
arbitrary `model.*` sweep axis reaches the cluster without the job script, the
manifest format or the queue manager learning anything about it.

Each still owns the whole of its own mechanism, because the parts that differ do
not usefully generalise:

* SlurmExecutor's run id only exists once `sbatch` has accepted the job, so it
  submits held, writes the manifest, then releases.
* TmuxExecutor has no scheduler to ask, so the sweep's own name is the id and
  the manifest is written before anything starts.
* LocalExecutor is the whole queue itself -- one worker thread per GPU, blocking
  on its own subprocess -- and finishes before it returns rather than detaching.

What they share is this module's sweep layout: one `sweeps/<timestamp>/` folder
per submission holding `experiments.csv`, `sweep.yaml` and one directory per run.
A finished sweep is the same shape whichever backend produced it.
"""

import csv
import os
import queue
import shlex
import shutil
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path

from grid import Run, detect_gpus, emit

SWEEPS_DIR = Path("sweeps")
MANIFEST_NAME = "experiments.csv"
SWEEP_NAME_FORMAT = "%Y-%m-%d_%H-%M-%S"

DEBUG_TIME_LABEL = "00:30:00"
DEBUG_SECONDS = 30 * 60

#: Variables that decide *where* jax runs, and which therefore never travel from
#: a launcher to a run. A JAX_PLATFORMS=cpu picked up from the submitting shell
#: is silent: the run trains, logs and checkpoints normally, ~160x slower per
#: gradient step. run_rl.slurm and tmux_pool.py unset the same list at their end,
#: since `--export=ALL` and a tmux server's environment are not this process's to
#: filter.
DEVICE_VARS = ("JAX_PLATFORMS", "JAX_PLATFORM_NAME")


def child_env(**extra: str) -> dict:
    """This process's environment as a run should see it: device choice left to
    the machine, plus whatever the caller pins itself."""
    env = {k: v for k, v in os.environ.items() if k not in DEVICE_VARS}
    env.update(extra)
    return env


#: The manifest's columns. `config` is relative to the sweep dir, so the tree
#: stays relocatable -- a Slurm task copies it to node-local scratch and back.
MANIFEST_FIELDS = ["index", "run_id", "tag", "config"]


def write_manifest(sweep_dir: Path, run_id: str, runs: list[Run]) -> Path:
    """The submitted sweep, one row per run. run_rl.slurm reads its own row out
    of this by array index; tmux_pool.py reads the whole thing as its queue.

    Four columns, and only one of them is load-bearing: whatever the grid varied,
    a row still points at one self-contained config."""
    sweep_dir.mkdir(parents=True, exist_ok=True)
    manifest = sweep_dir / MANIFEST_NAME

    # `lineterminator`, not csv's "\r\n" default: run_rl.slurm reads these rows
    # with `read`, which would carry the CR into the last field.
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({"index": r.index, "run_id": run_id, "tag": r.tag, "config": r.config}
                         for r in runs)

    return manifest


def read_manifest(sweep_dir: Path) -> list[dict]:
    with (Path(sweep_dir) / MANIFEST_NAME).open(newline="") as f:
        return list(csv.DictReader(f))


class Executor(ABC):
    """One submission mechanism. `submit` returns the lines submit.py prints, so
    a backend reports whatever is worth knowing about itself -- a Slurm job id, a
    tmux session to attach to -- without submit.py knowing which is which."""

    name: str
    #: Runs in flight when the caller does not say (Slurm's --array %N).
    default_parallelism: int
    #: Whether submit() returns only once every run has finished.
    blocking: bool = False

    @staticmethod
    @abstractmethod
    def available() -> bool:
        """Is this mechanism usable on this machine?"""

    @abstractmethod
    def submit(self, runs: list[Run], sweep_name: str, sweep_dir: Path, *,
               max_parallel: int, debug: bool, stream: bool = False) -> dict[str, str]:
        ...

    def describe_debug(self) -> str:
        return f"{DEBUG_TIME_LABEL} cap, one run at a time"


class LocalExecutor(Executor):
    """This machine, now: one worker thread per GPU, each pinning its subprocess
    with CUDA_VISIBLE_DEVICES and blocking on it before taking the next run -- so
    a GPU holds exactly one run at a time. Unlike the other two it does not
    detach; it is the foreground way to run a small grid or a --validate pass."""

    name = "local"
    blocking = True

    @staticmethod
    def available() -> bool:
        return True

    @property
    def default_parallelism(self) -> int:
        return max(1, len(detect_gpus()))

    def submit(self, runs, sweep_name, sweep_dir, *, max_parallel, debug, stream=False):
        write_manifest(sweep_dir, sweep_name, runs)

        gpus = detect_gpus()[:max_parallel] or [None] * min(max_parallel, 1)
        # More slots than GPUs is a deliberate ask (submit.py --max-parallel), so
        # the extra workers share devices round-robin rather than being dropped.
        slots = [gpus[i % len(gpus)] for i in range(max_parallel)] if gpus else [None]

        pending: queue.Queue = queue.Queue()
        for run in runs:
            pending.put(run)

        def worker(gpu_id):
            while True:
                try:
                    run = pending.get_nowait()
                except queue.Empty:
                    return
                run.exit_code = self._run_one(run, sweep_dir, gpu_id, stream,
                                              f"{run.index + 1}/{len(runs)}",
                                              DEBUG_SECONDS if debug else None)

        threads = [threading.Thread(target=worker, args=(gpu,), daemon=True) for gpu in slots]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        failed = [r for r in runs if r.exit_code != 0]
        return {"Sweep dir": str(sweep_dir),
                "Sweep tag": sweep_name,
                "Result": f"{len(runs) - len(failed)}/{len(runs)} succeeded"
                          + (f", {len(failed)} failed" if failed else "")}

    def _run_one(self, run: Run, sweep_dir: Path, gpu_id, stream, progress, timeout):
        cmd = run.command(str(sweep_dir))
        if timeout:
            cmd = ["timeout", str(timeout), *cmd]

        env_vars = child_env()
        label = run.tag
        if gpu_id is not None:
            env_vars["CUDA_VISIBLE_DEVICES"] = gpu_id
            label = f"gpu{gpu_id}|{run.tag}"

        log_path = sweep_dir / run.dir / "run.log"
        emit(f"[{label}] starting {progress} -- dir: {sweep_dir / run.dir}")

        with open(log_path, "w") as logf:
            if stream:
                # --validate: tee each line (label-prefixed) to the terminal too.
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, bufsize=1, env=env_vars)
                for line in proc.stdout:
                    emit(f"[{label}] {line.rstrip()}")
                    logf.write(line)
                proc.wait()
            else:
                proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env_vars)

        if proc.returncode == 0:
            emit(f"[{label}] OK")
        else:
            emit(f"[{label}] FAILED (exit {proc.returncode})\n{log_path.read_text()}")
        return proc.returncode


class SlurmExecutor(Executor):
    name = "slurm"
    default_parallelism = 8

    # --debug overrides for run_rl.slurm's own #SBATCH lines: sbatch's CLI wins
    # over them, so the job script stays the single description of a *real* run.
    DEBUG_PARTITION = "gpu_a100_short"
    DEBUG_JOB_NAME = "rl_dev"

    @staticmethod
    def available() -> bool:
        return shutil.which("sbatch") is not None

    def describe_debug(self) -> str:
        return f"{self.DEBUG_PARTITION}, {super().describe_debug()}"

    def submit(self, runs, sweep_name, sweep_dir, *, max_parallel, debug, stream=False):
        # One array task per *run*, i.e. per grid point -- not per {algo, env}.
        # A grid that varies a model axis therefore submits that many more tasks,
        # which is the point: each is one GPU's work and they queue independently.
        cmd = ["sbatch", "--parsable", "--hold",
               f"--array=0-{len(runs) - 1}%{max_parallel}",
               f"--export=ALL,SWEEP_NAME={sweep_name}"]

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

        # Held so the manifest -- whose `run_id` column is the array job id Slurm
        # only hands out at submission -- is on disk before any task reads it.
        try:
            write_manifest(sweep_dir, run_id, runs)
            subprocess.run(["scontrol", "release", run_id], check=True)
        except Exception:
            subprocess.run(["scancel", run_id], check=False)
            raise

        job_name = self.DEBUG_JOB_NAME if debug else "rl"
        return {"Run ID": run_id,
                "Sweep dir": str(sweep_dir),
                "Logs": f"logs/{job_name}/{run_id}_<task>/",
                "Sweep tag": run_id,
                "Status": f"sacct -j {run_id}",
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
        return max(1, len(detect_gpus()))

    def submit(self, runs, sweep_name, sweep_dir, *, max_parallel, debug, stream=False):
        # There is no scheduler to hand out an id and nothing has to wait for
        # one, so the sweep's own name is it: folder, manifest column and wandb
        # tag all agree.
        run_id = sweep_name
        write_manifest(sweep_dir, run_id, runs)

        session = f"barl-{sweep_name}"
        manager = [sys.executable, "scripts/tmux_pool.py",
                   "--session", session,
                   "--sweep-dir", str(sweep_dir),
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

        # `tmux new-session` succeeds as long as tmux started, even if the manager
        # it was given dies on the spot -- which would otherwise be reported here
        # as a submitted sweep that never runs. The session outliving its first
        # moment is the actual evidence that it took.
        time.sleep(1.0)
        alive = subprocess.run(["tmux", "has-session", "-t", session],
                               capture_output=True).returncode == 0
        if not alive:
            raise SystemExit(
                f"tmux session {session!r} exited immediately -- the manager failed to start. "
                f"Reproduce it with:\n  {shlex.join(manager)}")

        attach = (f"tmux switch-client -t {session}" if os.environ.get("TMUX")
                  else f"tmux attach -t {session}")

        return {"Session": session,
                "Sweep dir": str(sweep_dir),
                "Attach": attach,
                "Sweep tag": run_id,
                "Status": f"{sweep_dir}/tasks.yaml",
                "Cancel": f"tmux kill-session -t {session}"}


EXECUTORS = {cls.name: cls for cls in (SlurmExecutor, TmuxExecutor, LocalExecutor)}


def resolve(choice: str) -> Executor:
    """`auto` is what makes one command work in both places: Slurm where there is
    a scheduler, tmux where there is only this box. `local` is never auto-picked
    -- running a sweep in the foreground is something you ask for."""
    if choice == "auto":
        for name in ("slurm", "tmux"):
            if EXECUTORS[name].available():
                return EXECUTORS[name]()
        return LocalExecutor()

    cls = EXECUTORS[choice]
    if not cls.available():
        raise SystemExit(f"--executor {choice}: not usable here (nothing to run it with on PATH)")
    return cls()
