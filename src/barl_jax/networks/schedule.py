"""Scalar schedules as ordinary components: one curve shape per registered
class, all sharing the same (start, end, steps, warmup) parameterization and a
single abstract `value(t)` over normalized progress t in [0, 1].

Being components means a schedule is a slot like any other -- `class:` picks the
curve, so switching a noise ladder from geometric to cosine spacing is a
one-word edit in config/models.yaml and a sweepable `model.components.<slot>.class`
axis:

    sigmas: {class: ExponentialSchedule, start: 0.1, end: 1.0e-3, steps: 10}

Two ways to read one, both off the same object:

    sched(step)   # step-indexed and clamped at `end` past `steps`; Schedule is
                  # callable, so it composes with any step-indexed caller
    sched.ladder()  # the whole curve as a (steps,) array -- the noise-schedule
                    # use, where `steps` is a ladder height rather than a horizon
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax.numpy as jnp
from flax import nnx

from barl_jax.components import Component
from barl_jax.registration import register_component


@dataclass(frozen=True)
class ScheduleConfig:
  class_name: str = "ExponentialSchedule"
  start: float = 0.1
  end: float | None = 0.001
  steps: int = 10
  warmup: int = 0
  power: float = 2.0
  s: float = 0.008


class Schedule(Component, nnx.Module, ABC):
  """Warmup is a linear ramp 0 -> `start` over `warmup` steps; the curve proper
  runs `start` -> `end` over the remaining `steps - warmup`, then stays flat at
  `end` (so a `steps` estimate that undershoots the real run is harmless)."""

  kind     = "schedule"
  config_type = ScheduleConfig

  def __init__(self, env_stats=None, config=None, **kw):
    super().__init__(env_stats, config, **kw)
    start, end, steps, warmup = config.start, config.end, config.steps, config.warmup
    if steps < 1 or warmup < 0 or warmup >= steps:
      raise ValueError(f"schedule requires steps >= 1 and 0 <= warmup < steps, got steps={steps}, warmup={warmup}")
    self.start, self.end, self.steps, self.warmup = start, (start if end is None else end), steps, warmup

  @abstractmethod
  def value(self, t: float) -> float:
    ...

  def __call__(self, step: int) -> float:
    if step < self.warmup:
      # step + 1, so the very first step isn't a wasted zero one.
      return self.start * (step + 1) / self.warmup
    return self.value(self.progress(step))

  def progress(self, step: int) -> float:
    return min(1.0, max(0.0, (step - self.warmup) / max(1, self.steps - self.warmup)))

  def lerp(self, w: float) -> float:
    return self.start + w * (self.end - self.start)

  def ladder(self, n: int | None  = None, dtype=jnp.float32):
    """`n` points spanning the curve endpoint-inclusive, ignoring `warmup` --
    the ladder view, for schedules used as a fixed set of levels."""
    n  = self.steps if n is None else n
    ts = [i / (n - 1) for i in range(n)] if n > 1 else [0.0]
    return jnp.asarray([self.value(t) for t in ts], dtype=dtype)

  def __repr__(self) -> str:
    return f"{type(self).__name__}(start={self.start}, end={self.end}, steps={self.steps}, warmup={self.warmup})"


@register_component
class ConstantSchedule(Schedule):

  def value(self, t: float) -> float:
    return self.start


@register_component
class LinearSchedule(Schedule):

  def value(self, t: float) -> float:
    return self.lerp(t)


@register_component
class CosineSchedule(Schedule):
  """Half-cosine (SGDR annealing): flat at both ends, steepest halfway."""

  def value(self, t: float) -> float:
    return self.lerp(0.5 * (1.0 - math.cos(math.pi * t)))


@register_component
class SineSchedule(Schedule):
  """Quarter-sine: leaves `start` slowly, arrives at `end` steeply."""

  def value(self, t: float) -> float:
    return self.lerp(1.0 - math.sin((1.0 - t) * math.pi / 2))


@register_component
class PolynomialSchedule(Schedule):

  config_type = ScheduleConfig

  def __init__(self, env_stats=None, config=None, **kw):
    super().__init__(env_stats, config, **kw)
    self.power = config.power

  def value(self, t: float) -> float:
    return self.lerp(t ** self.power)


@register_component
class ExponentialSchedule(Schedule):
  """Geometric interpolation -- constant *ratio* per step rather than constant
  difference, which is what a sigma ladder or a decay-to-1%-of-base wants.
  Both endpoints must be strictly positive."""

  def __init__(self, env_stats=None, config=None, **kw):
    super().__init__(env_stats, config, **kw)
    if self.start <= 0.0 or self.end <= 0.0:
      raise ValueError("ExponentialSchedule requires positive start and end")

  def value(self, t: float) -> float:
    return self.start * (self.end / self.start) ** t


@register_component
class CosineAlphaBarSchedule(Schedule):
  """Nichol & Dhariwal's cosine alpha-bar ladder, rescaled to run
  `start` -> `end`; `s` is their small offset keeping beta_0 away from 0."""

  config_type = ScheduleConfig

  def __init__(self, env_stats=None, config=None, **kw):
    super().__init__(env_stats, config, **kw)
    self.s = config.s

  def value(self, t: float) -> float:
    f  = lambda u: math.cos((u + self.s) / (1 + self.s) * math.pi / 2) ** 2
    return self.lerp(1.0 - f(t) / f(0.0))
