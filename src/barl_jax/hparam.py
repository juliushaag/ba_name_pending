"""Config declarations: every class-attribute default a component resolves
against its YAML layer, structural or algorithmic alike -- there is no
separate searchable kind; a sweep varies any of them the same way.

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


class ConfigValues:
  """The read-back for a resolved `ConfigDict`: plain attribute access
  (`self.mlp.dtype`), not a dict lookup. Hashable by identity -- no `__eq__`/
  `__hash__` override -- so it can sit directly on an nnx.Module the same way
  a plain str/float/bool-valued `Config` already does, whatever its own
  fields' hashability (a `norm_kwargs={}` field included)."""

  def __init__(self, **fields):
    self.__dict__.update(fields)

  def todict(self) -> dict:
    """This group's fields as a plain dict, for `**self.mlp.todict()` at a
    call site whose keyword names match the group's field names one-for-one."""
    return dict(self.__dict__)

  def __repr__(self):
    return f"ConfigValues({self.__dict__!r})"


class ConfigDict(Config):
  """A named group of `Config` declarations, read back as one namespace
  instead of one class attribute per field:

      mlp = ConfigDict(dtype='float32', activation='gelu', norm='none')
      # ...
      self.mlp.dtype, self.mlp.activation, self.mlp.norm

  Each field still resolves at its own flat name against the YAML layer --
  `--set components.actor.dtype=bfloat16` overrides it exactly as it would a
  standalone `Config(...)` declaration. Grouping only changes how the *class*
  declares and the *component* reads the fields back, not the override
  surface or what a run's resolved config logs. A raw value is auto-wrapped
  the same way `Config`'s own constructor sugars one; pass an explicit
  `Config(...)` for a field that needs its own `type` (`style` is not
  supported on a grouped field -- give it its own top-level Config instead)."""

  __slots__ = ("fields",)

  def __init__(self, **fields):
    self.fields = {name: value if isinstance(value, Config) else Config(value)
                   for name, value in fields.items()}
    super().__init__({name: f.default for name, f in self.fields.items()})

  def __repr__(self):
    return f"ConfigDict({', '.join(f'{k}={v.default!r}' for k, v in self.fields.items())})"
