#!/usr/bin/env python3

"""The local stand-in for Slurm, run by scripts/submit.py in window 0 of its own
tmux session.

One tmux window per manifest row, running exactly the command a Slurm array task
runs -- so this is Slurm *and* run_rl.slurm together: it owns the queue that
`--array=0-N%M` expresses, and hands each task a GPU. A window is a task: it has
its own terminal to watch, its own scrollback, and can be killed on its own.

It knows nothing about algorithms, environments or sweep axes. A manifest row is
an index, a tag and a path to a config that submit.py already wrote; running it
is one command. That is the whole interface, which is why a sweep over any axis
needs no change here.

Unlike a Slurm task there is no node-local scratch to work in and copy back: this
is the same filesystem, and each run writes only into its own directory under the
sweep folder, so it writes there directly.
"""

import argparse
import os
import shlex
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

from executors import read_manifest
from grid import detect_gpus

REPO_ROOT = Path(__file__).resolve().parent.parent

POLL_SECONDS = 0.5

STATE_STYLE = {"queued": "dim", "running": "cyan", "done": "green",
               "failed": "bold red", "cancelled": "yellow"}


@dataclass
class Task:
  index: int
  tag: str
  #: This run's directory under the sweep folder -- already holding the
  #: config.yaml submit.py wrote, and where the run puts everything else.
  dir: Path
  config: Path
  run_id: str = ""
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
    return {"index": self.index, "run": self.tag, "state": self.state, "gpu": self.gpu,
            "window": self.window, "exit_code": self.exit_code,
            "elapsed_s": round(self.elapsed, 1)}


class Pool:
  """Queue, GPU pool and tmux windows for one sweep."""

  def __init__(self, session: str, sweep_dir: Path, gpus: list[str], time_limit: int | None):
    self.session = session
    self.sweep_dir = sweep_dir
    self.free_gpus = list(gpus)
    self.time_limit = time_limit
    self.tasks = self._load_tasks()
    self.pending = list(self.tasks)
    self.running: list[Task] = []
    self.run_id = self.tasks[0].run_id if self.tasks else ""

  # ---- manifest ------------------------------------------------------------

  def _load_tasks(self) -> list[Task]:
    """The same experiments.csv run_rl.slurm reads a single row out of. A row is
    already everything a task is -- no combo to re-derive, no tag to recompute."""
    tasks = []
    for row in read_manifest(self.sweep_dir):
      config = self.sweep_dir / row["config"]
      tasks.append(Task(index=int(row["index"]), tag=row["tag"], run_id=row["run_id"],
                        dir=config.parent, config=config))
    return tasks

  # ---- tmux ----------------------------------------------------------------

  def _tmux(self, *args: str, capture: bool = False) -> str:
    proc = subprocess.run(["tmux", *args], text=True, check=capture, capture_output=capture)
    return proc.stdout.strip() if capture else ""

  def _live_windows(self) -> set[str]:
    out = self._tmux("list-windows", "-t", self.session, "-F", "#{window_id}", capture=True)
    return set(out.splitlines())

  def start(self, task: Task, gpu: str) -> None:
    inner = ["uv", "run", "python", "scripts/online_learning.py", "--config", str(task.config)]
    if self.time_limit:
      inner = ["timeout", str(self.time_limit), *inner]

    # PIPESTATUS, not $?: `tee` would otherwise be the exit code that gets
    # recorded, and it succeeds even when the run under it dies.
    script = (f"{shlex.join(inner)} 2>&1 | tee {shlex.quote(str(task.dir / 'term.out'))}\n"
              f"echo ${{PIPESTATUS[0]}} > {shlex.quote(str(task.dir / 'exit_code'))}\n")

    # PATH for the same reason submit.py sets it on the session: a new window
    # otherwise takes the tmux server's, which need not have this venv's `uv`.
    task.window = self._tmux(
      "new-window", "-d", "-t", f"{self.session}:", "-n", task.tag,
      "-c", str(REPO_ROOT), "-e", f"CUDA_VISIBLE_DEVICES={gpu}",
      "-e", f"PATH={os.environ['PATH']}",
      "-P", "-F", "#{window_id}", "--", "bash", "-c", script, capture=True)

    # Kept alive after the command exits so a failure stays on screen; a task
    # that succeeds has its window killed explicitly in `reap`.
    self._tmux("set-option", "-t", task.window, "-w", "remain-on-exit", "on")

    task.state, task.gpu, task.started = "running", gpu, time.time()
    self.running.append(task)

  # ---- lifecycle -----------------------------------------------------------

  def reap(self) -> bool:
    """Finishes whatever is done. True if anything changed."""
    changed = False
    live = None

    for task in list(self.running):
      code_file = task.dir / "exit_code"
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
      if task.state == "done" and task.window:
        self._tmux("kill-window", "-t", task.window)
      self.running.remove(task)
      self.free_gpus.append(task.gpu)
      changed = True

    return changed

  def cancel(self) -> None:
    for task in self.running:
      if task.window:
        self._tmux("kill-window", "-t", task.window)
      task.state, task.finished = "cancelled", time.time()
    for task in self.pending:
      task.state = "cancelled"
    self.running.clear()
    self.pending.clear()

  # ---- reporting -----------------------------------------------------------

  def write_status(self) -> None:
    """`sacct` for a local sweep. Deliberately not sweep.yaml (submit.py's index
    over what the grid was) and not experiments.csv (the matrix as submitted) --
    this is the live state of the queue."""
    payload = {"run_id": self.run_id, "session": self.session,
               "sweep_dir": str(self.sweep_dir),
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
  args.add_argument("--max-parallel", type=int, required=True, help="Runs in flight at once")
  args.add_argument("--time-limit", type=int, default=None,
                    help="Per-task wall clock in seconds (the local --debug cap)")
  kwargs = args.parse_args()

  gpus = detect_gpus() or ["0"]
  # More slots than GPUs means deliberate oversubscription, so hand the same GPU
  # out more than once rather than leaving slots idle.
  slots = [gpus[i % len(gpus)] for i in range(kwargs.max_parallel)]

  pool = Pool(kwargs.session, Path(kwargs.sweep_dir).resolve(), slots, kwargs.time_limit)

  console = Console()

  def on_signal(*_):
    pool.cancel()
    pool.write_status()
    console.print("[yellow]Cancelled.[/]")
    sys.exit(1)

  # SIGHUP included deliberately: `tmux kill-session` -- the local `scancel` --
  # hangs the pane up rather than terminating it, so without this the manager
  # dies without marking anything cancelled.
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

  failed = [t for t in pool.tasks if t.state != "done"]
  console.print(f"Done: {len(pool.tasks) - len(failed)}/{len(pool.tasks)} succeeded"
                + (f", {len(failed)} failed -- their windows are still open" if failed else ""))
  console.print(f"Results: {pool.sweep_dir}")
  # The manager's own window is the last thing left; without this it closes the
  # instant the sweep ends and takes the summary with it.
  console.input("\nPress enter to close this window. ")


if __name__ == "__main__":
  main()
