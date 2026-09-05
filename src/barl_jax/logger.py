import time
from collections import defaultdict
from contextlib import ExitStack

import numpy as np
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

import wandb


def _to_scalar(v):
  if hasattr(v, "item"):
    try:
      return v.item()
    except (RuntimeError, ValueError):
      return None
  if isinstance(v, (int, float, np.floating, np.integer)):
    return float(v)
  return None


def _fmt_value(v) -> str:
  s = _to_scalar(v)
  if s is None:
    return str(v)
  if abs(s) >= 1000 or (0 < abs(s) < 1e-3):
    return f"{s:.4g}"
  return f"{s:,.4f}"


def _fmt_duration(seconds: float) -> str:
  seconds = int(seconds)
  h, rem = divmod(seconds, 3600)
  m, s = divmod(rem, 60)
  return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


class Logger():
  """Wraps wandb with a live rich terminal display, both driven off the
  same log()/increment() calls."""

  REFRESH_INTERVAL = 0.5   # seconds between terminal redraws, regardless of log() call frequency

  def __init__(self, log_name : str, step_keys : set, *, finish_timeout : float,
               run_name : str = None, tags : list[str] = None,
               config : dict = None, group=None, seeds : list[int] = (0,)):
    """One wandb run **per seed**, all sharing `group` so wandb's group view
    draws them as one experiment with a mean/spread band -- rather than one run
    carrying `seed0/`-prefixed copies of every key, which no wandb aggregation
    understands. `reinit="create_new"` is what lets them coexist in a process;
    note it leaves `wandb.run` as None, so nothing here may use the global
    `wandb.log`/`wandb.run` -- every call goes through an explicit run.

    Only the first run captures the console. Every run here shares one process,
    hence one stdout, so the others would each store an identical copy of it --
    and wandb installs a *separate* stream wrapper (and flush thread) per run,
    so N of them stack on the same stream and the first run's finish() has to
    unwind from under the rest. That is where a finished run was observed to
    wedge for 17 minutes with wandb-core idle and the network fine."""

    self._seeds = list(seeds)
    self._resources = ExitStack()
    self._runs = []
    try:
      for i, seed in enumerate(self._seeds):
        run = wandb.init(project=log_name, group=group or run_name,
                         name=f"{run_name}/s{seed}",
                         config={**(config or {}), "seed": seed}, tags=tags,
                         reinit="create_new",
                         settings=wandb.Settings(finish_timeout=finish_timeout,
                                                console="auto" if i == 0 else "off"))
        self._runs.append(run)
        self._resources.callback(run.finish)
    except BaseException:
      self._resources.close()
      raise

    try:
      self._steps = {k: 0 for k in step_keys}

      for k in self._steps:
        self._each("define_metric", k)

      self._run_name    = run_name or self._runs[0].name
      self._console     = Console()
      self._latest      = dict()
      self._start_time  = time.monotonic()
      self._last_render = 0.0
      self._rate_ref    = dict(self._steps)
      self._rate_time   = self._start_time

      self._live = Live(self._render(), console=self._console, refresh_per_second=4, transient=False)
      self._resources.callback(self._live.stop)
      self._live.start()
    except BaseException:
      self._resources.close()
      raise

  def _each(self, method : str, *args, **kwargs):
    """Fan a wandb call out over every seed's run."""
    for run in self._runs:
      getattr(run, method)(*args, **kwargs)

  def register(self, key : str, step_key : str):
    self._each("define_metric", key, step_metric=step_key)

  def get_step(self, step_key : str):
    return self._steps[step_key]

  def advance(self, steps):
    """Bumps the step counters without touching wandb. They are x-axes
    (define_metric(step_metric=...)), so wandb only needs them alongside the
    data they index -- and a wandb.log() call costs ~150 us whatever it carries,
    which at one per grad step is minutes of CPU over a run."""
    for k, v in steps.items():
      if k not in self._steps:
        raise KeyError(f"Unknown step key '{k}'")

      self._steps[k] += v

    self._maybe_render()

  def log(self, stats : dict, quiet : dict = None):
    """Run-level quantities -- the ones with no seed axis (rollout/, buffer/,
    model/). Written to *every* seed's run, so each one's history is complete
    on its own. `quiet` is the debug tier: shipped to wandb but kept out of the
    terminal panel, whose ~200 extra keys would swamp the display without
    telling you anything at a glance."""
    if not stats and not quiet:
      return

    # Step counters ride in the SAME wandb.log() call as the data -- logging them
    # separately risks an off-by-one/misaligned plot (define_metric aligns a
    # key's x-axis to whatever step value was logged alongside it). Same reason
    # this does not pass wandb's own `step=`: the x-axes here are ordinary
    # metrics named by define_metric(step_metric=...), not wandb's global step.
    self._each("log", {**stats, **(quiet or {}), **self._steps})
    self._latest.update(stats)
    self._maybe_render()

  def log_per_seed(self, stats : list[dict], quiet : list[dict] = None, panel : dict = None):
    """Seed i's own values, under plain key names, into seed i's own run --
    which is the whole point of a run per seed: `critic/q` is comparable across
    them in one wandb panel, and the group view derives the mean and band.

    `panel` is what the terminal shows instead (the seed mean); it is not sent
    to wandb, since it would collide with the per-seed keys and wandb can
    recompute it from the group anyway."""
    if not any(stats) and not (quiet and any(quiet)):
      return

    for i, run in enumerate(self._runs):
      payload = {**stats[i], **(quiet[i] if quiet else {}), **self._steps}
      run.log(payload)

    self._latest.update(panel or {})
    self._maybe_render()

  def message(self, text: str):
    """Prints a one-off line (e.g. "Iteration 3: ...") above the live
    stats panel, through the same Console the panel renders on so the two
    never fight over the terminal."""
    self._console.print(text)

  def summary(self, key: str, value, seed_index: int = None):
    """Sets a wandb run-summary field (e.g. best_reward-so-far) -- shown on
    the run's overview page instead of buried in the metric history. With no
    `seed_index` it lands on every seed's run; with one, only on that seed's."""
    value = _to_scalar(value) if _to_scalar(value) is not None else value
    runs = self._runs if seed_index is None else [self._runs[seed_index]]
    for run in runs:
      run.summary[key] = value

  def _maybe_render(self):
    now = time.monotonic()
    if now - self._last_render < self.REFRESH_INTERVAL:
      return
    self._last_render = now
    self._live.update(self._render())

  def _render(self) -> Panel:
    now = time.monotonic()
    elapsed = now - self._rate_time

    header = Table.grid(padding=(0, 3))
    header.add_column(style="dim")
    header.add_column(justify="right")
    header.add_row("elapsed", _fmt_duration(now - self._start_time))
    for k, v in self._steps.items():
      rate = (v - self._rate_ref.get(k, v)) / elapsed if elapsed > 1e-6 else 0.0
      header.add_row(k, f"{v:,}" + (f"  [dim]({rate:,.0f}/s)[/]" if rate else ""))
    self._rate_ref = dict(self._steps)
    self._rate_time = now

    groups: dict[str, dict] = defaultdict(dict)
    for k, v in sorted(self._latest.items()):
      prefix = k.split("/")[0] if "/" in k else "misc"
      groups[prefix][k] = v

    # Headline sections first, then one per component/algorithm alphabetically.
    lead = [p for p in ("eval", "rollout", "batch", "buffer", "model") if p in groups]
    tables = [header]
    for prefix in lead + sorted(set(groups) - set(lead)):
      t = Table(title=f"[bold]{prefix}[/]", title_justify="left", show_header=False, box=None, pad_edge=False)
      t.add_column(style="cyan")
      t.add_column(justify="right")
      for k, v in groups[prefix].items():
        t.add_row(k, _fmt_value(v))
      tables.append(t)

    return Panel(Group(*tables), title=f"[bold]{self._run_name}[/]", border_style="cyan")

  def log_video(self, key : str, video : list[np.ndarray], seed_index : int = 0):
    """Videos go to one seed's run (the caller renders only one), not all --
    the same clip duplicated across every run is just storage."""
    video = np.stack(video)

    if video.ndim != 4:
      raise ValueError(
          f"Expected video shape (T,H,W,C) or (T,C,H,W), got {video.shape}"
      )

    if video.shape[-1] == 3:
      video = video.transpose(0, 3, 1, 2)

    self._runs[seed_index].log({key: wandb.Video(video, fps=30, format="mp4")})

  def finish(self):
    try:
      self._live.update(self._render())
    finally:
      self._resources.close()
