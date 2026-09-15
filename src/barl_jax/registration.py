"""Explicit, idempotent discovery of built-in, plugin, and external
implementations. Three ways to name an implementation, in the order every
`class_for`/`get_component` tries them:

1. A built-in name (`load_builtins` below) -- always tried first, so an
   installed plugin can never silently shadow one.
2. A plugin name -- an entry point installed under `barl_jax.<group>`
   (`barl_jax.policies`, `barl_jax.environments`, `barl_jax.components`),
   discovered once per process and loaded lazily, only when actually
   selected. A name registered by more than one installed distribution, or
   that collides with a built-in, is a deterministic error at lookup time --
   never a silent "whichever was scanned first".
3. `module:Class` -- direct, explicit, and unaffected by what plugins are
   installed; the mechanism `module:Class` selection already was, still the
   right answer for a one-off or unpublished implementation."""
from importlib import import_module
from importlib import metadata as _metadata
from threading import RLock

_LOCK = RLock()
_LOADED = set()
_PLUGINS: dict[str, dict] = {}


def load_builtins(group):
  modules = {
    "components": ("critics", "actors", "networks.squash", "networks.regulator", "networks.schedule"),
    # Per-algorithm Policy classes (AMDP, DIME, NCLQL, SAC, SDAC) and their
    # ModelConfig schemas live beside their actor -- importing "actors" is
    # what makes both `Policy._classes` and `config.schemas.MODEL_SCHEMAS`
    # complete.
    "policies": ("policies", "actors"),
    "environments": tuple(f"environments.{name}" for name in
                          ("mujoco_gym", "dm_control", "mani_skill", "humanoid_bench")),
    "executors": ("executors",),
  }
  with _LOCK:
    if group in _LOADED:
      return
    for module in modules[group]:
      import_module(f"barl_jax.{module}")
    _LOADED.add(group)


def _discover_plugins(group: str) -> dict:
  """Every entry point installed under `barl_jax.<group>`, scanned once per
  process (not per lookup, not per component/policy/environment class
  construction) and cached. A name two different distributions both
  register is a collision here, not a coin flip at whichever happened to be
  imported/scanned first."""
  with _LOCK:
    if group in _PLUGINS:
      return _PLUGINS[group]
    entries: dict[str, _metadata.EntryPoint] = {}
    for ep in _metadata.entry_points(group=f"barl_jax.{group}"):
      prior = entries.get(ep.name)
      if prior is not None and (prior.dist is None or ep.dist is None or prior.dist.name != ep.dist.name):
        a = prior.dist.name if prior.dist else "?"
        b = ep.dist.name if ep.dist else "?"
        raise ValueError(f"plugin name {ep.name!r} is registered by both {a!r} and {b!r} "
                         f"under barl_jax.{group} -- rename one of them")
      entries[ep.name] = ep
    _PLUGINS[group] = entries
    return entries


def resolve_named(group: str, name: str, builtins: dict, base: type):
  """`name` against `builtins` (already-loaded built-ins) first, then
  installed `barl_jax.<group>` plugin entry points -- `None` if `name` is
  neither, so the caller can raise its own "not found" with its own
  built-in-name listing. A name present in both a built-in and a plugin is
  an error, not a silent built-in win: `module:Class` already exists for
  "I mean exactly this one", so an ambiguous shared short name has no
  legitimate reason to resolve quietly."""
  entries = _discover_plugins(group)
  if name in builtins and name in entries:
    dist = entries[name].dist.name if entries[name].dist else "?"
    raise ValueError(f"{name!r} is both a built-in {base.__name__} and a plugin from "
                     f"{dist!r} (barl_jax.{group}) -- rename the plugin's entry point")
  if name in builtins:
    return builtins[name]
  if name not in entries:
    return None
  ep = entries[name]
  try:
    cls = ep.load()
  except ImportError as e:
    dist = ep.dist.name if ep.dist else "?"
    raise ImportError(f"plugin {name!r} ({dist}, entry point {ep.value!r}) failed to "
                      f"import -- its distribution's own dependency is likely missing: {e}") from e
  if not isinstance(cls, type) or not issubclass(cls, base):
    raise TypeError(f"plugin {name!r} ({ep.value!r}) does not implement {base.__name__}")
  return cls


def available_names(group: str, builtins: dict) -> list[str]:
  """Built-in names plus installed plugin names, for listing -- entry points
  carry their name without `.load()`ing the target, so this never imports a
  plugin just to say it exists."""
  return sorted(set(builtins) | set(_discover_plugins(group)))


def import_type(path):
  module, separator, name = path.partition(":")
  if not separator:
    raise KeyError(f"Unknown implementation {path!r}; external classes use 'module:Class'")
  value = import_module(module)
  for part in name.split("."):
    value = getattr(value, part)
  if not isinstance(value, type):
    raise TypeError(f"{path!r} does not name a class")
  return value


# ---------------------------------------------------------------------------
# Component registry -- concrete Actor/Critic/Squash/Schedule/Regulator
# classes, so config/models.yaml can name one (`class: HLGaussCritic`)
# instead of a Python import path. Same shape as Policy._classes/
# Environment._classes (policy.py, env.py), kept here rather than under
# `networks/` since it is not itself a network and is looked up the same
# way policies and environments are, through this module's own
# load_builtins/resolve_named/available_names.
# ---------------------------------------------------------------------------

_COMPONENT_REGISTRY: dict[str, type] = {}


def register_component(cls: type) -> type:
  name = cls.__name__
  if name in _COMPONENT_REGISTRY:
    raise ValueError(f"Component {name} already registered")
  _COMPONENT_REGISTRY[name] = cls
  return cls


def get_component(name: str) -> type:
  load_builtins("components")
  if ":" in name:
    return import_type(name)
  from barl_jax.components import Component
  cls = resolve_named("components", name, _COMPONENT_REGISTRY, Component)
  if cls is None:
    raise KeyError(f"{name} not in registered components: "
                   f"{available_names('components', _COMPONENT_REGISTRY)}")
  return cls


def available_components() -> list[str]:
  load_builtins("components")
  return available_names("components", _COMPONENT_REGISTRY)


# ---------------------------------------------------------------------------
# Executor registry -- concrete ActionExecutor classes (barl_jax.executors),
# resolved by name the same way components are, through this module's own
# load_builtins/resolve_named/available_names. Kept separate from
# _COMPONENT_REGISTRY rather than folded into it: resolve_named's
# issubclass(cls, base) gate only actually runs on the plugin-entry-point
# path, so reusing "components" would silently let a third-party plugin
# executor skip the ActionExecutor check every built-in executor is exempt
# from anyway -- a dedicated group enforces it for plugins here too.
# ---------------------------------------------------------------------------

_EXECUTOR_REGISTRY: dict[str, type] = {}


def register_executor(cls: type) -> type:
  name = cls.__name__
  if name in _EXECUTOR_REGISTRY:
    raise ValueError(f"Executor {name} already registered")
  _EXECUTOR_REGISTRY[name] = cls
  return cls


def get_executor(name: str) -> type:
  load_builtins("executors")
  if ":" in name:
    return import_type(name)
  from barl_jax.executors.base import ActionExecutor
  cls = resolve_named("executors", name, _EXECUTOR_REGISTRY, ActionExecutor)
  if cls is None:
    raise KeyError(f"{name} not in registered executors: "
                   f"{available_names('executors', _EXECUTOR_REGISTRY)}")
  return cls


def available_executors() -> list[str]:
  load_builtins("executors")
  return available_names("executors", _EXECUTOR_REGISTRY)
