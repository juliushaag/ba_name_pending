"""Explicit, idempotent discovery of built-in, plugin, and external
implementations. Three ways to name an implementation, in the order every
`Registry.get` tries them:

1. A built-in name (`Registry.load`) -- always tried first, so an
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


class Registry:
  """One named family of implementations (components, executors, policies,
  environments): built-ins registered by import side effect, installed plugins,
  and `module:Class` paths, resolved in that order."""

  def __init__(self, group: str, kind: str, base: str, modules: tuple[str, ...]):
    self.group, self.kind, self._base, self._modules = group, kind, base, modules
    self.classes: dict[str, type] = {}
    self._loaded = False

  def register(self, cls: type) -> type:
    if cls.__name__ in self.classes:
      raise ValueError(f"{self.kind.capitalize()} {cls.__name__} already registered")
    self.classes[cls.__name__] = cls
    return cls

  def load(self) -> None:
    with _LOCK:
      if self._loaded:
        return
      for module in self._modules:
        import_module(f"barl_jax.{module}")
      self._loaded = True

  def get(self, name: str) -> type:
    self.load()
    base = import_type(self._base)
    cls = import_type(name) if ":" in name else resolve_named(self.group, name, self.classes, base)
    if cls is None:
      raise KeyError(f"{name!r} is not a registered {self.kind} or plugin "
                     f"({', '.join(self.available())})")
    if not issubclass(cls, base):
      raise TypeError(f"{name!r} must implement {base.__name__}")
    return cls

  def available(self) -> list[str]:
    self.load()
    return available_names(self.group, self.classes)


COMPONENTS = Registry("components", "component", "barl_jax.components:Component",
                      ("critics", "actors", "networks.squash", "networks.regulator",
                       "networks.schedule"))
EXECUTORS = Registry("executors", "executor", "barl_jax.executors.base:ActionExecutor",
                     ("executors",))
# Per-algorithm Policy classes (AMDP, DIME, NCLQL, SAC, SDAC) and their ModelConfig
# schemas live beside their actor, so importing "actors" completes both registries.
POLICIES = Registry("policies", "policy", "barl_jax.policy:Policy", ("policies", "actors"))
ENVIRONMENTS = Registry("environments", "environment", "barl_jax.env:Environment",
                        tuple(f"environments.{name}" for name in
                              ("mujoco_gym", "dm_control", "mani_skill", "humanoid_bench")))

register_component = COMPONENTS.register
register_executor = EXECUTORS.register
register_policy = POLICIES.register
register_environment = ENVIRONMENTS.register
get_component = COMPONENTS.get
get_executor = EXECUTORS.get
available_components = COMPONENTS.available
available_executors = EXECUTORS.available


def load_builtins(group: str) -> None:
  {r.group: r for r in (COMPONENTS, EXECUTORS, POLICIES, ENVIRONMENTS)}[group].load()
