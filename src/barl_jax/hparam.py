"""Class-attribute declarations of a component's tunables and structure.

Two explicit kinds of class attribute, and keeping them apart is the point:

* **`HParam`** -- a *searchable algorithmic knob*. A default, a `range` (or
  `choices`) a hyperparameter sweep varies over, and a `descr`. These are the
  only class attributes a search axis may target (`lr`, `gamma`, `num_bon`,
  `utd_ratio`, ...).

* **`Config`** -- a *structural default*. Just a value; the type is inferred.
  How a net is put together is not a hyperparameter you sweep per-trial, so it
  is declared as `Config`, not `HParam`, and a search axis onto it is rejected.

**Which net / which time embedding is a configuration value, not an inheritance
choice.** The named styles live as plain dicts (`networks.mlp.py`'s
`MLP_STYLES`, `networks.flow.py`'s `TIME_STYLES`): `name -> surface`. A
component *selects* one with a `Config` tagged with that dict:

    class DDPMActor(Actor, ActorOptim):
        net_style = Config("gelu",   style=MLP_STYLES)    # MLP_STYLES["gelu"]
        time_emb  = Config("sin_20", style=TIME_STYLES)   # TIME_STYLES["sin_20"]

so switching `--set model.components.actor.net_style=silu` (or a grid axis)
changes the whole net, and the resolved config records the *choice* -- a style's
surface keys are attributed as `style: MLP_STYLES.gelu`, never hidden as class
defaults. A component can still pin one key on top of a style (a per-slot
`activation:` beats the style's), and a style's surface sits below the component
own declarations and the slot's YAML.

Composition beyond styles is still plain class inheritance for *algorithm
geometry* -- an optimizer role (`networks/optim.py`'s `CriticOptim`), a critic's
Bellman geometry (`networks/critic.py`'s `CriticLossConfig`, `Polyak`) -- since
those are decisions about which algorithm code a component implements, not about
a selectable default.

`Component`/`Policy` collect `_hparams` (HParam) and `_configs` (Config) down
the MRO and resolve them at construction -- see HParamsOwner in policy.py."""
from __future__ import annotations

import operator
import types
from typing import Any, Optional

from barl_jax.configs import deep_merge

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
  """A structural default: a value, its type inferred. Not searchable.

  `style` is the **style dict** this value selects from: a module-level
  `name -> surface` table (`networks.mlp.py`'s `MLP_STYLES`,
  `networks.flow.py`'s `TIME_STYLES`). At resolution the chosen entry's whole
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


# ---------------------------------------------------------------------------
# size ladders -- the capacity tables a `size:` resolves against
# ---------------------------------------------------------------------------

BUNDLES: dict[str, dict] = {}


def register_bundle(path: str, kwargs: dict) -> None:
  if path in BUNDLES:
    raise ValueError(f"bundle {path!r} already registered")
  BUNDLES[path] = dict(kwargs)


def ladder(kind: str, name: str) -> dict:
  """`sizes.<kind>.<name>` as a dict of kwargs."""
  path = f"sizes.{kind}.{name}"
  if path not in BUNDLES:
    raise KeyError(f"no size {kind!r} x {name!r} -- registered: {sorted(BUNDLES)}")
  return BUNDLES[path]


def register_ladder(kind: str, name: str, kwargs: dict) -> None:
  register_bundle(f"sizes.{kind}.{name}", kwargs)


def ladder_names(kind: str) -> list[str]:
  prefix = f"sizes.{kind}."
  return sorted(k for k in BUNDLES if k.startswith(prefix))


for _kind, _sizes in (("critic", {"small": 256, "medium": 512, "large": 1024, "xlarge": 2048}),
                      ("actor", {"small": 256, "medium": 512, "large": 1024, "xlarge": 2048})):
  for _name, _w in _sizes.items():
    register_ladder(_kind, _name, {"latent_dim": _w, "layers": 1 if _kind == "critic" else 2,
                                   **({"n_models": 2} if _kind == "critic" else {})})