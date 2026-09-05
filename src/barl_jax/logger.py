"""Three interchangeable reporting sinks behind one contract (`register`,
`get_step`, `advance`, `log`, `log_per_seed`, `message`, `summary`,
`log_video`, `finish`) -- `make_logger()` picks one by `cfg["logger"]`.
`trainer.py` calls only that contract, never wandb or Rich directly, so it
runs unmodified with any of the three.

* `WandbLogger` -- wandb runs (one per seed, see its own docstring) plus the
  live terminal panel.
* `LocalLogger` -- the same terminal panel and step bookkeeping, with metrics
  appended to a local JSON-lines file instead of shipped to wandb. Works
  without wandb installed or configured.
* `NullLogger` -- step bookkeeping only, so the training loop's own
  `while log.get_step(...) < ...` condition still advances; every other call
  is a no-op. `message` still prints -- it is free and it is the only signal
  a `logger: none` run gives on stdout.

wandb is only imported inside `WandbLogger.__init__`, so `cfg["logger"]` set
to `local` or `none` never requires it to be installed."""

import json
import time
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from barl_jax.result import presentable


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


class BaseLogger:
  """Step bookkeeping and the shared `finish()` resource contract. Subclasses
  supply how `log`/`log_per_seed`/`message`/`summary`/`log_video` actually
  report -- everything here is backend-independent."""

  def __init__(self, step_keys: set):
    self._resources = ExitStack()
    self._steps = {k: 0 for k in step_keys}

  def register(self, key: str, step_key: str) -> None:
    """Declares `key`'s x-axis as `step_key`. Only wandb needs telling
    (`define_metric`); the other sinks read `self._steps` directly."""

  def get_step(self, step_key: str):
    return self._steps[step_key]

  def advance(self, steps: dict) -> None:
    for k, v in steps.items():
      if k not in self._steps:
        raise KeyError(f"Unknown step key '{k}'")
      self._steps[k] += v

  def log(self, stats: dict, quiet: dict = None) -> None:
    raise NotImplementedError

  def log_per_seed(self, stats: list, quiet: list = None, panel: dict = None) -> None:
    raise NotImplementedError

  def message(self, text: str) -> None:
    raise NotImplementedError

  def summary(self, key: str, value, seed_index: int = None) -> None:
    raise NotImplementedError

  def log_video(self, key: str, video: list, seed_index: int = 0) -> None:
    raise NotImplementedError

  def finish(self) -> None:
    self._resources.close()


class _TerminalMixin:
  """The live Rich panel `WandbLogger` and `LocalLogger` both show: elapsed
  time, step rates, and every logged key grouped by its `prefix/` namespace.
  Requires `self._steps`, `self._run_name`, and `self._latest` (updated by
  the subclass's own `log`/`log_per_seed`)."""

  REFRESH_INTERVAL = 0.5   # seconds between terminal redraws, regardless of log() call frequency

  def _start_terminal(self, run_name: str) -> None:
    self._run_name    = run_name
    self._console     = Console()
    self._latest      = dict()
    self._start_time  = time.monotonic()
    self._last_render = 0.0
    self._rate_ref    = dict(self._steps)
    self._rate_time   = self._start_time

    self._live = Live(self._render(), console=self._console, refresh_per_second=4, transient=False)
    self._resources.callback(self._live.stop)
    self._live.start()

  def message(self, text: str) -> None:
    self._console.print(text)

  def _maybe_render(self) -> None:
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

  def finish(self) -> None:
    try:
      self._live.update(self._render())
    finally:
      super().finish()


class WandbLogger(_TerminalMixin, BaseLogger):
  """Wraps wandb with the live terminal display, both driven off the same
  log()/increment() calls."""

  def __init__(self, run_name: str, *, step_keys: set, finish_timeout: float,
               tags: list[str] = None, config: dict = None, group=None,
               seeds: list[int] = (0,), **_ignored):
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
    import wandb
    self._wandb = wandb

    super().__init__(step_keys)
    self._seeds = list(seeds)
    self._runs = []
    try:
      for i, seed in enumerate(self._seeds):
        run = wandb.init(project="BA-RL", group=group or run_name,
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
      for k in self._steps:
        self._each("define_metric", k)
      self._start_terminal(run_name or self._runs[0].name)
    except BaseException:
      self._resources.close()
      raise

  def _each(self, method: str, *args, **kwargs):
    """Fan a wandb call out over every seed's run."""
    for run in self._runs:
      getattr(run, method)(*args, **kwargs)

  def register(self, key: str, step_key: str) -> None:
    self._each("define_metric", key, step_metric=step_key)

  def advance(self, steps: dict) -> None:
    """Bumps the step counters without touching wandb. They are x-axes
    (define_metric(step_metric=...)), so wandb only needs them alongside the
    data they index -- and a wandb.log() call costs ~150 us whatever it carries,
    which at one per grad step is minutes of CPU over a run."""
    super().advance(steps)
    self._maybe_render()

  def log(self, stats: dict, quiet: dict = None) -> None:
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

  def log_per_seed(self, stats: list, quiet: list = None, panel: dict = None) -> None:
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

  def summary(self, key: str, value, seed_index: int = None) -> None:
    """Sets a wandb run-summary field (e.g. best_reward-so-far) -- shown on
    the run's overview page instead of buried in the metric history. With no
    `seed_index` it lands on every seed's run; with one, only on that seed's."""
    value = _to_scalar(value) if _to_scalar(value) is not None else value
    runs = self._runs if seed_index is None else [self._runs[seed_index]]
    for run in runs:
      run.summary[key] = value

  def log_video(self, key: str, video: list, seed_index: int = 0) -> None:
    """Videos go to one seed's run (the caller renders only one), not all --
    the same clip duplicated across every run is just storage."""
    video = np.stack(video)
    if video.ndim != 4:
      raise ValueError(f"Expected video shape (T,H,W,C) or (T,C,H,W), got {video.shape}")
    if video.shape[-1] == 3:
      video = video.transpose(0, 3, 1, 2)
    self._runs[seed_index].log({key: self._wandb.Video(video, fps=30, format="mp4")})


class LocalLogger(_TerminalMixin, BaseLogger):
  """The same terminal panel as `WandbLogger`, with metrics appended to a
  local `metrics.jsonl` (one JSON object per `log`/`log_per_seed` call) and
  the run summary kept in `summary.json` -- both strict presentation JSON
  (`barl_jax.result`), and both readable with no wandb account or network."""

  def __init__(self, run_name: str, *, step_keys: set, run_dir: str = None,
               seeds: list[int] = (0,), **_ignored):
    super().__init__(step_keys)
    self._seeds = list(seeds)
    self._summary: dict = {}
    path = Path(run_dir) if run_dir else Path("checkpoints") / run_name
    path.mkdir(parents=True, exist_ok=True)
    self._metrics_path = path / "metrics.jsonl"
    self._summary_path = path / "summary.json"
    # Truncate: a run_dir is reused across a resume, and a stale tail from a
    # previous attempt would otherwise read as this run's own history.
    self._metrics_file = open(self._metrics_path, "w")
    self._resources.callback(self._metrics_file.close)
    self._start_terminal(run_name)

  def _write_row(self, payload: dict) -> None:
    # `_to_scalar` also pulls a Python float out of a numpy/jax scalar --
    # `stats`/`quiet` carry those directly from device-side metrics, and
    # `json.dumps` has no default encoder for them.
    row = {}
    for k, v in payload.items():
      scalar = _to_scalar(v)
      row[k] = presentable(scalar) if scalar is not None else v
    self._metrics_file.write(json.dumps(row, sort_keys=True) + "\n")
    self._metrics_file.flush()

  def log(self, stats: dict, quiet: dict = None) -> None:
    if not stats and not quiet:
      return
    self._write_row({**stats, **(quiet or {}), **self._steps})
    self._latest.update(stats)
    self._maybe_render()

  def log_per_seed(self, stats: list, quiet: list = None, panel: dict = None) -> None:
    if not any(stats) and not (quiet and any(quiet)):
      return
    for i, seed in enumerate(self._seeds):
      self._write_row({"seed": seed, **stats[i], **(quiet[i] if quiet else {}), **self._steps})
    self._latest.update(panel or {})
    self._maybe_render()

  def summary(self, key: str, value, seed_index: int = None) -> None:
    from barl_jax.result import write_json_atomic
    value = _to_scalar(value) if _to_scalar(value) is not None else value
    target = self._summary if seed_index is None else self._summary.setdefault(
      "per_seed", {}).setdefault(str(self._seeds[seed_index]), {})
    target[key] = presentable(value) if isinstance(value, (int, float)) else value
    write_json_atomic(self._summary_path, self._summary)

  def log_video(self, key: str, video: list, seed_index: int = 0) -> None:
    """No local video sink -- keeping one means re-implementing wandb's mp4
    encode path for a file nothing then displays. `logger: wandb` is what
    videos need."""


class NullLogger(BaseLogger):
  """Step bookkeeping only. `message` still prints -- see module docstring."""

  def __init__(self, run_name: str = None, *, step_keys: set, **_ignored):
    super().__init__(step_keys)

  def message(self, text: str) -> None:
    print(text)

  def log(self, stats: dict, quiet: dict = None) -> None:
    pass

  def log_per_seed(self, stats: list, quiet: list = None, panel: dict = None) -> None:
    pass

  def summary(self, key: str, value, seed_index: int = None) -> None:
    pass

  def log_video(self, key: str, video: list, seed_index: int = 0) -> None:
    pass


LOGGERS = {"wandb": WandbLogger, "local": LocalLogger, "none": NullLogger}


def make_logger(kind: str, run_name: str, *, step_keys: set, **kwargs) -> BaseLogger:
  """`kind` is `cfg["logger"]` (`config/run.py`'s `LOGGER_KINDS`), validated
  at preflight (`config.run.validate_run`) -- an unknown kind here is a bug in
  that validation, not a run mistake, hence AssertionError rather than a
  message aimed at a config author."""
  cls = LOGGERS.get(kind)
  assert cls is not None, f"unknown logger kind {kind!r} (validate_run should have rejected it)"
  return cls(run_name, step_keys=step_keys, **kwargs)
