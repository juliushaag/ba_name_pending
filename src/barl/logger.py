import time
from collections import defaultdict

import numpy as np
import wandb

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table


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

  def __init__(self, log_name : str, step_keys : set, run_name : str = None, tags : list[str] = None, config : dict = None, group=None):

    wandb.init(project=log_name, group=group, name=run_name, config=config, tags=tags)

    self._step_key_dict = dict()
    self._steps = { k : 0 for k in step_keys}

    for k in self._steps:
      wandb.define_metric(k)

    self._run_name    = run_name or wandb.run.name
    self._console     = Console()
    self._latest      = dict()
    self._start_time  = time.monotonic()
    self._last_render = 0.0
    self._rate_ref    = dict(self._steps)
    self._rate_time   = self._start_time

    self._live = Live(self._render(), console=self._console, refresh_per_second=4, transient=False)
    self._live.start()

  def register(self, key : str, step_key : str):
    wandb.define_metric(key, step_metric=step_key)

  def get_step(self, step_key : str):
    return self._steps[step_key]

  def _ensure_step_key(self, key: str, step_key: str):

    if step_key not in self._steps:
        raise KeyError(
            f"Unknown step key '{key}'. "
            f"Known keys: {list(self._steps.keys())}"
        )

    if self._step_key_dict.get(key) == step_key:
      return

    wandb.define_metric(key, step_metric=step_key)
    self._step_key_dict[key] = step_key

  def increment(self, steps):
    for k, v in steps.items():
      if k not in self._steps:
        raise KeyError(f"Unknown step key '{k}'")

      self._steps[k] += v

    wandb.log({**self._steps})
    self._maybe_render()

  def log(self, stats : dict):
    # Step counters ride in the SAME wandb.log() call as the data -- logging them
    # via a separate increment() risks an off-by-one/misaligned plot (define_metric
    # aligns a key's x-axis to whatever step value was logged alongside it).
    wandb.log({**stats, **self._steps})

    self._latest.update(stats)
    self._maybe_render()

  def log_quiet(self, stats : dict):
    """wandb only, no terminal panel -- for the debug tier, whose ~200 keys
    would swamp the live display without telling you anything at a glance."""
    if not stats:
      return
    wandb.log({**stats, **self._steps})

  def message(self, text: str):
    """Prints a one-off line (e.g. "Iteration 3: ...") above the live
    stats panel, through the same Console the panel renders on so the two
    never fight over the terminal."""
    self._console.print(text)

  def summary(self, key: str, value):
    """Sets a wandb run-summary field (e.g. best_reward-so-far) -- shown on
    the run's overview page instead of buried in the metric history."""
    wandb.run.summary[key] = _to_scalar(value) if _to_scalar(value) is not None else value

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

  def log_video(self, key : str, video : list[np.ndarray]):

    video = np.stack(video)

    if video.ndim != 4:
      raise ValueError(
          f"Expected video shape (T,H,W,C) or (T,C,H,W), got {video.shape}"
      )

    if video.shape[-1] == 3:
      video = video.transpose(0, 3, 1, 2)

    wandb.log({
        key : wandb.Video(
            video,
            fps=30,
            format="mp4"
        )
    })

  def finish(self):
    self._live.update(self._render())
    self._live.stop()
    wandb.finish()
