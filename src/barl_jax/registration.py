"""Explicit, idempotent discovery of built-in and external implementations."""
from importlib import import_module
from threading import RLock

_LOCK = RLock()
_LOADED = set()


def load_builtins(group):
  modules = {
    "components": ("critics", "actors", "networks.squash", "networks.regulator", "networks.schedule"),
    "policies": ("policies",),
    "environments": tuple(f"environments.{name}" for name in
                          ("mujoco_gym", "dm_control", "mani_skill", "humanoid_bench")),
  }
  with _LOCK:
    if group in _LOADED:
      return
    for module in modules[group]:
      import_module(f"barl_jax.{module}")
    _LOADED.add(group)


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
