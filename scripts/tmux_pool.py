#!/usr/bin/env python3

"""The local stand-in for Slurm, run by scripts/submit.py in window 0 of its own
tmux session.

One tmux window per manifest row, running exactly the command a Slurm array task
runs -- so this is Slurm *and* run_rl.slurm together: it owns the queue that
`--array=0-N%M` expresses, and the scratch-dir/copy-back that run_rl.slurm's
save_results does. A window is a task: it has its own terminal to watch, its own
scrollback, and can be killed on its own.

Each task gets its own scratch directory because two concurrent run_sweep.py
processes pointed at one --sweep_dir would both write sweep.yaml at its root and
clobber each other; Slurm sidesteps this with node-local $TMPDIR. The finished
run directory is moved up into the sweep folder afterwards -- same filesystem,
so it is a rename.
"""

import argparse
import csv
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml
from rich.console import Console
from rich.live import Live
from rich.table import Table

from executors import MANIFEST_NAME
from run_sweep import detect_gpus, name_combos

REPO_ROOT = Path(__file__).resolve().parent.parent

POLL_SECONDS = 0.5

STATE_STYLE = {"queued": "dim", "running": "cyan", "done": "green",
               "failed": "bold red", "cancelled": "yellow"}


@dataclass
class Task:
  index: int
  algo: str
  env: str
  tag: str
  scratch: Path
  state: str = "queued"
  gpu: str | None = None
  window: str | None = None
  exit_code: int | None = None
  started: float | None = None
  finished: float | None = None

  @property
  def elapsed(self) -> float:
    if self.started is None:
      return 0.0
    return (self.finished or time.time()) - self.started

  def as_record(self) -> dict:
    return {"index": self.index, "run": self.tag, "algo": self.algo, "env": self.env,
            "state": self.state, "gpu": self.gpu, "window": self.window,
            "exit_code": self.exit_code, "elapsed_s": round(self.elapsed, 1)}


class Pool:
  """Queue, GPU pool and tmux windows for one sweep."""

  def __init__(self, session: str, sweep_dir: Path, run_id: str, seeds: int,
               gpus: list[str], time_limit: int | None):
    self.session = session
    self.sweep_dir = sweep_dir
    self.run_id = run_id
    self.seeds = seeds
    self.free_gpus = list(gpus)
    self.time_limit = time_limit
    self.tasks = self._load_tasks()
    self.pending = list(self.tasks)
    self.running: list[Task] = []

  # ---- manifest ------------------------------------------------------------

  def _load_tasks(self) -> list[Task]:
    """The same experiments.csv run_rl.slurm reads a single row out of, and the
    tags run_sweep.py will give the runs it launches for these combos."""
    manifest = self.sweep_dir / MANIFEST_NAME
    with manifest.open(newline="") as f:
      rows = list(csv.DictReader(f))

    axes = {"algo": sorted({r["algo"] for r in rows}),
            "env":  sorted({r["env"] for r in rows}),
            "seed": [0]}
    combos = [{"algo": r["algo"], "env": r["env"], "seed": 0} for r in rows]
    tags = name_combos(combos, axes, self.seeds)

    scratch_root = self.sweep_dir / ".tasks"
    return [Task(index=i, algo=r["algo"], env=r["env"], tag=tag,
                 scratch=scratch_root / str(i))
            for i, (r, tag) in enumerate(zip(rows, tags))]

  # ---- tmux ----------------------------------------------------------------

  def _tmux(self, *args: str, capture: bool = False) -> str:
    proc = subprocess.run(["tmux", *args], text=True, check=capture,
                          capture_output=capture)
    return proc.stdout.strip() if capture else ""

  def _live_windows(self) -> set[str]:
    out = self._tmux("list-windows", "-t", self.session, "-F", "#{window_id}",
                     capture=True)
    return set(out.splitlines())

  def start(self, task: Task, gpu: str) -> None:
    task.scratch.mkdir(parents=True, exist_ok=True)

    inner = ["uv", "run", "python", "scripts/run_sweep.py",
             "--algos", task.algo, "--envs", task.env,
             "--seeds", str(self.seeds),
             "--sweep_tag", self.run_id,
             "--sweep_dir", str(task.scratch)]
    if self.time_limit:
      inner = ["timeout", str(self.time_limit), *inner]

    # PIPESTATUS, not $?: `tee` would otherwise be the exit code that gets
    # recorded, and it succeeds even when the run under it dies.
    script = (f"{shlex.join(inner)} 2>&1 | tee {shlex.quote(str(task.scratch / 'term.out'))}\n"
              f"echo ${{PIPESTATUS[0]}} > {shlex.quote(str(task.scratch / 'exit_code'))}\n")

    # PATH for the same reason submit.py sets it on the session: a new window
    # otherwise takes the tmux server's, which need not have this venv's `uv`.
    task.window = self._tmux(
      "new-window", "-d", "-t", f"{self.session}:", "-n", task.tag,
      "-c", str(REPO_ROOT), "-e", f"CUDA_VISIBLE_DEVICES={gpu}",
      "-e", f"PATH={os.environ['PATH']}",
      "-P", "-F", "#{window_id}", "--", "bash", "-c", script, capture=True)

    # Kept alive after the command exits so a failure stays on screen; a task
    # that succeeds has its window killed explicitly in `collect`.
    self._tmux("set-option", "-t", task.window, "-w", "remain-on-exit", "on")

    task.state, task.gpu, task.started = "running", gpu, time.time()
    self.running.append(task)

  # ---- lifecycle -----------------------------------------------------------

  def reap(self) -> bool:
    """Finishes whatever is done. True if anything changed."""
    changed = False
    live = None

    for task in list(self.running):
      code_file = task.scratch / "exit_code"
      if code_file.exists():
        try:
          task.exit_code = int(code_file.read_text().strip())
        except ValueError:
          task.exit_code = -1
      else:
        # No exit code and no window: killed from outside before it could write
        # one. Anything else is still running.
        live = self._live_windows() if live is None else live
        if task.window in live:
          continue
        task.exit_code = -1

      task.finished = time.time()
      task.state = "done" if task.exit_code == 0 else "failed"
      self.collect(task)
      self.running.remove(task)
      self.free_gpus.append(task.gpu)
      changed = True

    return changed

  def collect(self, task: Task) -> None:
    """run_rl.slurm's save_results, locally: the run directory moves up into the
    sweep folder, and this task's sweep.yaml and terminal output go inside it."""
    run_dirs = [d for d in task.scratch.iterdir() if d.is_dir()] if task.scratch.exists() else []
    dest = self.sweep_dir / (run_dirs[0].name if run_dirs else task.tag)

    if run_dirs:
      shutil.rmtree(dest, ignore_errors=True)
      run_dirs[0].replace(dest)
    else:
      # run_sweep.py died before it made one -- term.out is all there is, and it
      # is the only thing that says why.
      dest.mkdir(parents=True, exist_ok=True)

    for name in ("sweep.yaml", "term.out"):
      source = task.scratch / name
      if source.exists():
        source.replace(dest / name)

    shutil.rmtree(task.scratch, ignore_errors=True)

    if task.state == "done" and task.window:
      self._tmux("kill-window", "-t", task.window)

  def clear_scratch(self) -> None:
    """Each task removed its own scratch dir in collect(); this is their now-empty
    parent. Left alone if anything is still in it, since that is evidence."""
    root = self.sweep_dir / ".tasks"
    if root.is_dir() and not any(root.iterdir()):
      root.rmdir()

  def cancel(self) -> None:
    for task in self.running:
      if task.window:
        self._tmux("kill-window", "-t", task.window)
      task.state, task.finished = "cancelled", time.time()
      self.collect(task)
    for task in self.pending:
      task.state = "cancelled"
    self.running.clear()
    self.pending.clear()

  # ---- reporting -----------------------------------------------------------

  def write_status(self) -> None:
    """`sacct` for a local sweep. Deliberately not sweep.yaml (each task's own
    run_sweep.py writes one of those into its run directory) and not
    experiments.csv, which stays the matrix as submitted."""
    payload = {"run_id": self.run_id, "session": self.session,
               "sweep_dir": str(self.sweep_dir), "seeds": self.seeds,
               "tasks": [t.as_record() for t in self.tasks]}
    (self.sweep_dir / "tasks.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))

  def table(self) -> Table:
    counts = {}
    for task in self.tasks:
      counts[task.state] = counts.get(task.state, 0) + 1
    summary = "  ".join(f"{n} {state}" for state, n in counts.items())

    table = Table(title=f"{self.session}   {len(self.tasks)} runs   {summary}",
                  title_justify="left")
    for column in ("#", "run", "state", "gpu", "elapsed", "window"):
      table.add_column(column)

    for task in self.tasks:
      minutes, seconds = divmod(int(task.elapsed), 60)
      table.add_row(str(task.index), task.tag,
                    f"[{STATE_STYLE[task.state]}]{task.state}[/]",
                    task.gpu or "-",
                    f"{minutes}:{seconds:02d}" if task.started else "-",
                    task.window or "-")
    return table


def main():
  args = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  args.add_argument("--session", required=True, help="tmux session to open run windows in")
  args.add_argument("--sweep-dir", required=True, help="sweeps/<name>, holding experiments.csv")
  args.add_argument("--run-id", required=True, help="Shared id: the runs' wandb sweep tag")
  args.add_argument("--seeds", type=int, required=True, help="Seeds vmapped inside each run")
  args.add_argument("--max-parallel", type=int, required=True, help="Runs in flight at once")
  args.add_argument("--time-limit", type=int, default=None,
                    help="Per-task wall clock in seconds (the local --debug cap)")
  kwargs = args.parse_args()

  gpus = detect_gpus()
  # More slots than GPUs means deliberate oversubscription, so hand the same GPU
  # out more than once rather than leaving slots idle.
  slots = [gpus[i % len(gpus)] for i in range(kwargs.max_parallel)]

  pool = Pool(kwargs.session, Path(kwargs.sweep_dir).resolve(), kwargs.run_id,
              kwargs.seeds, slots, kwargs.time_limit)

  console = Console()

  def on_signal(*_):
    pool.cancel()
    pool.write_status()
    pool.clear_scratch()
    console.print("[yellow]Cancelled.[/]")
    sys.exit(1)

  # SIGHUP included deliberately: `tmux kill-session` -- the local `scancel` --
  # hangs the pane up rather than terminating it, so without this the manager
  # dies without collecting what was in flight or marking anything cancelled.
  for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(sig, on_signal)

  pool.write_status()
  with Live(pool.table(), console=console, refresh_per_second=2) as live:
    while pool.pending or pool.running:
      while pool.pending and pool.free_gpus:
        pool.start(pool.pending.pop(0), pool.free_gpus.pop(0))
        pool.write_status()

      if pool.reap():
        pool.write_status()

      live.update(pool.table())
      time.sleep(POLL_SECONDS)

    live.update(pool.table())

  pool.clear_scratch()

  failed = [t for t in pool.tasks if t.state != "done"]
  console.print(f"Done: {len(pool.tasks) - len(failed)}/{len(pool.tasks)} succeeded"
                + (f", {len(failed)} failed -- their windows are still open" if failed else ""))
  console.print(f"Results: {pool.sweep_dir}")
  # The manager's own window is the last thing left; without this it closes the
  # instant the sweep ends and takes the summary with it.
  console.input("\nPress enter to close this window. ")


if __name__ == "__main__":
  main()
