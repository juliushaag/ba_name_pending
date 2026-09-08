"""Config defaults and algorithmic HParam declarations.

MLP settings are declared directly by their components. Capacity tiers live
in networks/mlp.py; time embeddings may still use Config(style=TIME_STYLES).
"""
from __future__ import annotations

from typing import Any

_MISSING = object()


def infer_type(default) -> type:
  if isinstance(default, bool):
    return bool
  if isinstance(default, int):
    return int
  if isinstance(default, float):
    return float
  if isinstance(default, str):
    return str
  return type(default)


class Config:
  """A non-search default: a value, its type inferred.

  `style` is the **style dict** this value selects from: a module-level
  `name -> surface` table (for example `networks.time_embedding.py`'s `TIME_STYLES`). At resolution the chosen entry's whole
  surface is merged in below the component's own declarations, so the value is
  a configuration choice ("which MLP", "which time embedding") rather than an
  inheritance decision."""

  __slots__ = ("default", "type", "style")

  def __init__(self, default: Any, *, type=None, style: dict = None):
    self.default = default
    self.type = type or infer_type(default)
    self.style = style

  def coerce(self, value: Any) -> Any:
    t = self.type
    if t is None:
      return value
    if isinstance(value, t) or value is None:
      return value
    # The one coercion that matters in practice: YAML writes 1 not 1.0, and a
    # float-typed value should accept both.
    if t is float and isinstance(value, (int,)) and not isinstance(value, bool):
      return float(value)
    raise TypeError(f"{value!r} is not a {t.__name__}")
  def __repr__(self):
    return f"Config({self.default!r})"


class HParam(Config):
  """A searchable algorithmic knob: a default, the `range`/`choices` a sweep
  varies, and a description. See Config for the non-searchable structural case."""

  __slots__ = ("range", "choices", "descr")

  def __init__(self, default: Any, *, type=None, range=None, choices: tuple = None,
               descr: str = ""):
    super().__init__(default, type=type)
    self.range = range
    self.choices = choices
    self.descr = descr

  def __repr__(self):
    return f"HParam({self.default!r})"
