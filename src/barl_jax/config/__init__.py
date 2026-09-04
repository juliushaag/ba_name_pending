"""Public config package surface with lazy re-exports.

This package exists as the canonical home for config code, but it must not
eagerly import `run.py`/`tree.py` on package import: `barl_jax.configs` is a
compatibility shim that imports `barl_jax.config.toolkit`, and Python loads the
package `__init__` first when resolving that submodule. Eager re-exports here
would therefore pull `hparam.py` through `run.py` while `barl_jax.configs` is
still only partially initialized, creating an import cycle.
"""

from importlib import import_module

__all__ = [
  "CONFIG_DIR",
  "Config",
  "ConfigMixin",
  "ConfigTree",
  "INJECTED_KEYS",
  "RUN_CONFIGS",
  "TRAINING_KEYS",
  "_check_type",
  "_format_type",
  "available_models",
  "deep_merge",
  "env_stats",
  "explain",
  "get_overrides",
  "get_snapshot",
  "load_config",
  "load_yaml_with_includes",
  "resolve_includes",
  "resolve_model_name",
  "resolve_params",
  "resolve_slot",
  "resolve_training",
  "resolved_flat",
  "run_defaults",
  "run_keys",
  "set_overrides",
  "set_snapshot",
  "snapshot_for",
  "structure_problems",
  "tree",
  "unknown_component_defaults",
  "validate",
]

_EXPORTS = {
  "CONFIG_DIR": ("barl_jax.config.toolkit", "CONFIG_DIR"),
  "Config": ("barl_jax.config.toolkit", "Config"),
  "ConfigMixin": ("barl_jax.config.toolkit", "ConfigMixin"),
  "_check_type": ("barl_jax.config.toolkit", "_check_type"),
  "_format_type": ("barl_jax.config.toolkit", "_format_type"),
  "deep_merge": ("barl_jax.config.toolkit", "deep_merge"),
  "load_yaml_with_includes": ("barl_jax.config.toolkit", "load_yaml_with_includes"),
  "resolve_includes": ("barl_jax.config.toolkit", "resolve_includes"),
  "INJECTED_KEYS": ("barl_jax.config.run", "INJECTED_KEYS"),
  "RUN_CONFIGS": ("barl_jax.config.run", "RUN_CONFIGS"),
  "load_config": ("barl_jax.config.run", "load_config"),
  "run_defaults": ("barl_jax.config.run", "run_defaults"),
  "run_keys": ("barl_jax.config.run", "run_keys"),
  "TRAINING_KEYS": ("barl_jax.config.tree", "TRAINING_KEYS"),
  "ConfigTree": ("barl_jax.config.tree", "ConfigTree"),
  "available_models": ("barl_jax.config.tree", "available_models"),
  "env_stats": ("barl_jax.config.tree", "env_stats"),
  "explain": ("barl_jax.config.tree", "explain"),
  "get_overrides": ("barl_jax.config.tree", "get_overrides"),
  "get_snapshot": ("barl_jax.config.tree", "get_snapshot"),
  "resolve_model_name": ("barl_jax.config.tree", "resolve_model_name"),
  "resolve_params": ("barl_jax.config.tree", "resolve_params"),
  "resolve_slot": ("barl_jax.config.tree", "resolve_slot"),
  "resolve_training": ("barl_jax.config.tree", "resolve_training"),
  "resolved_flat": ("barl_jax.config.tree", "resolved_flat"),
  "set_overrides": ("barl_jax.config.tree", "set_overrides"),
  "set_snapshot": ("barl_jax.config.tree", "set_snapshot"),
  "snapshot_for": ("barl_jax.config.tree", "snapshot_for"),
  "structure_problems": ("barl_jax.config.tree", "structure_problems"),
  "tree": ("barl_jax.config.tree", "tree"),
  "unknown_component_defaults": ("barl_jax.config.tree", "unknown_component_defaults"),
  "validate": ("barl_jax.config.tree", "validate"),
}


def __getattr__(name: str):
  try:
    module_name, attr_name = _EXPORTS[name]
  except KeyError as exc:
    raise AttributeError(name) from exc
  value = getattr(import_module(module_name), attr_name)
  globals()[name] = value
  return value
