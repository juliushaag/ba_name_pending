"""General-purpose YAML config toolkit: `$()` file inclusion with deep merge,
plus a typed `.get(key, type)` accessor that fails with a message naming the
exact key, the available alternatives and (for a type mismatch) what was
expected vs. what was there.

Ported from a sibling project's `core/configs.py` and adapted to this repo's
fixed config root (`CONFIG_DIR`, see `tree.py`) instead of guessing it
from a config file's directory depth. Deliberately dependency-free (only
`yaml`/`pathlib`/`re`/`typing`), the same discipline `tree.py` holds
itself to, since it is the substrate `tree.py` builds its `$()`-include
/ deep-merge / typed-access layering on top of -- this module knows nothing
about that layering, only about YAML, includes and types.

`$()` includes:
    key: $(/path/from/config/root.yaml)
    key: $(relative/to/this/file.yaml)

are resolved and deep-merged recursively wherever they appear in a loaded
YAML tree, via `resolve_includes()` (`tree._load_yaml` routes every
file it reads through this, so `$()` works anywhere under `config/`).
"""

from __future__ import annotations

import os
import re
import types
from importlib.resources import files
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union, get_args, get_origin

import yaml

_UNION_ORIGINS = (Union, types.UnionType)

T = TypeVar("T")

_MISSING = object()

CONFIG_DIR = Path(os.environ["BARL_CONFIG_DIR"]).expanduser().resolve() if "BARL_CONFIG_DIR" in os.environ else Path(str(files("barl_jax") / "defaults"))

INCLUDE_PATTERN = re.compile(r"\$\(([^)]+)\)")


def deep_merge(base: Dict, override: Dict) -> Dict:
  result = dict(base)
  for key, value in override.items():
    if key in result and isinstance(result[key], dict) and isinstance(value, dict):
      result[key] = deep_merge(result[key], value)
    else:
      result[key] = value
  return result


def _load_include(include_path: str, current_dir: Path, config_root: Path) -> Any:
  if include_path.startswith("/"):
    file_path = config_root / include_path[1:]
  else:
    file_path = current_dir / include_path
  file_path = file_path.resolve()

  if not file_path.exists():
    raise FileNotFoundError(f"Included config not found: {file_path}")

  with open(file_path) as f:
    included = yaml.safe_load(f) or {}

  return resolve_includes(included, file_path.parent, config_root)


def resolve_includes(data: Any, current_dir: Path, config_root: Path) -> Any:
  if isinstance(data, str):
    match = INCLUDE_PATTERN.fullmatch(data.strip())
    return _load_include(match.group(1), current_dir, config_root) if match else data

  if isinstance(data, dict):
    return {key: resolve_includes(value, current_dir, config_root) for key, value in data.items()}

  if isinstance(data, list):
    return [resolve_includes(item, current_dir, config_root) for item in data]

  return data


def load_yaml_with_includes(path: str | Path, config_root: str | Path = CONFIG_DIR) -> Dict:
  path = Path(path).resolve()
  with open(path) as f:
    data = yaml.safe_load(f) or {}
  return resolve_includes(data, path.parent, Path(config_root).resolve())


def parse_set_overrides(assignments: list[str] | None) -> Dict[str, Any]:
  """`["components.critic.tier=large", ...]` (a repeated `--set PATH=VALUE`
  flag) -> `{"components.critic.tier": "large", ...}`, the dotted-override
  dict both `print_config.py` and `config_diff.py` resolve a model against.
  Each value is parsed as YAML scalar/collection syntax, so `--set
  params.lr=1e-4` and `--set params.normalize_obs=false` land as a float and
  a bool, not the literal strings."""
  overrides: Dict[str, Any] = {}
  for assignment in assignments or []:
    path, sep, value = assignment.partition("=")
    if not sep:
      raise SystemExit(f"--set {assignment!r} is not of the form path=value")
    overrides[path.strip()] = yaml.safe_load(value)
  return overrides


def _format_type(tp: Type) -> str:
  origin = get_origin(tp)
  if origin is None:
    return tp.__name__ if hasattr(tp, "__name__") else str(tp)
  if origin in _UNION_ORIGINS:
    return " | ".join(_format_type(a) for a in get_args(tp))
  args = get_args(tp)
  if not args:
    return str(tp)
  origin_name = origin.__name__ if hasattr(origin, "__name__") else str(origin)
  return f"{origin_name}[{', '.join(_format_type(a) for a in args)}]"


def _format_value(value: Any, max_len: int = 50) -> str:
  s = repr(value)
  return s if len(s) <= max_len else s[:max_len - 3] + "..."


def _check_type(value: Any, expected_type: Type, key: str, section: str) -> None:
  origin = get_origin(expected_type)
  full_key = f"{section}.{key}" if section else key

  if value is None:
    if origin in _UNION_ORIGINS and type(None) in get_args(expected_type):
      return
    raise TypeError(f"Config key '{full_key}' is None, expected {_format_type(expected_type)}")

  if origin in _UNION_ORIGINS:
    for arg in get_args(expected_type):
      if arg is type(None):
        continue
      try:
        _check_type(value, arg, key, section)
        return
      except TypeError:
        continue
    names = " | ".join(_format_type(a) for a in get_args(expected_type) if a is not type(None))
    raise TypeError(f"Config key '{full_key}' has wrong type\n"
                    f"  Expected: {names}\n"
                    f"  Got: {type(value).__name__} = {_format_value(value)}")

  if origin is not None:
    if not isinstance(value, origin):
      raise TypeError(f"Config key '{full_key}' has wrong type\n"
                      f"  Expected: {_format_type(expected_type)}\n"
                      f"  Got: {type(value).__name__} = {_format_value(value)}")
    args = get_args(expected_type)
    if not args:
      return
    if origin is list:
      for i, elem in enumerate(value):
        _check_type(elem, args[0], f"{key}[{i}]", section)
    elif origin is dict and len(args) >= 2:
      key_type, val_type = args
      for k, v in value.items():
        _check_type(k, key_type, f"{key}.{{key}}", section)
        _check_type(v, val_type, f"{key}.{k}", section)
    return

  if expected_type is float and isinstance(value, int) and not isinstance(value, bool):
    return
  if expected_type is int and isinstance(value, bool):
    raise TypeError(f"Config key '{full_key}' has wrong type\n  Expected: int\n  Got: bool = {value}")
  if expected_type is bool and isinstance(value, int) and not isinstance(value, bool):
    raise TypeError(f"Config key '{full_key}' has wrong type\n  Expected: bool\n  Got: int = {value}")
  if not isinstance(value, expected_type):
    raise TypeError(f"Config key '{full_key}' has wrong type\n"
                    f"  Expected: {_format_type(expected_type)}\n"
                    f"  Got: {type(value).__name__} = {_format_value(value)}")


class ConfigMixin:
  """Typed, dotted-path access into a section of a raw nested dict."""

  def __init__(self, section: str = "", data: Optional[Dict] = None, config_path: Optional[str] = None):
    self._section = section
    self._root_data = data if data is not None else {}
    self._config_path = config_path
    self._consumed: set[str] = set()

    if section and data:
      current = data
      for part in section.split("."):
        if not isinstance(current, dict) or part not in current:
          available = list(current.keys()) if isinstance(current, dict) else []
          raise ValueError(
            f"Config section '{section}' not found\n"
            f"  Missing '{part}'\n"
            f"  Available keys: {available}\n"
            f"  Config: {config_path or 'unknown'}")
        current = current[part]
      self._data = current if isinstance(current, dict) else {}
    else:
      self._data = data if data is not None else {}

  def get(self, key: str, expected_type: Optional[Type[T]] = None, *, default=_MISSING) -> T:
    parts = key.split(".")
    current = self._data
    for i, part in enumerate(parts):
      if not isinstance(current, dict):
        path = ".".join(parts[:i + 1])
        full_path = f"{self._section}.{path}" if self._section else path
        raise KeyError(f"Config key '{full_path}' not found\n"
                       f"  '{parts[i - 1] if i > 0 else self._section}' is not a dict\n"
                       f"  Config: {self._config_path or 'unknown'}")
      if part not in current:
        if default is not _MISSING:
          return default
        path = ".".join(parts[:i + 1])
        full_path = f"{self._section}.{path}" if self._section else path
        raise KeyError(f"Config key '{full_path}' not found\n"
                       f"  Available keys at '{'.'.join(parts[:i]) or self._section or 'root'}': "
                       f"{list(current.keys())}\n"
                       f"  Config: {self._config_path or 'unknown'}")
      current = current[part]

    if expected_type is not None:
      _check_type(current, expected_type, key, self._section)
      if expected_type is float and isinstance(current, int) and not isinstance(current, bool):
        self._consumed.add(parts[0])
        return current * 1.0  # type: ignore
    self._consumed.add(parts[0])
    return current  # type: ignore

  def has(self, key: str) -> bool:
    current = self._data
    for part in key.split("."):
      if not isinstance(current, dict) or part not in current:
        return False
      current = current[part]
    return True

  def resolve(self, key: str, decl, *, owner: str = None):
    from barl_jax.hparam import Config
    if not isinstance(decl, Config):
      raise TypeError(f"resolve() needs a Config for {key!r}, got {type(decl).__name__}")
    if not self.has(key):
      return decl.default
    value = self._data[key]
    self._consumed.add(key.split(".", 1)[0])
    if decl.type is not None:
      _check_type(value, decl.type, key, self._section)
    return decl.coerce(value)

  def unused_keys(self) -> set[str]:
    return set(self._data) - self._consumed

  def assert_all_consumed(self, *, owner: str = None) -> None:
    unused = sorted(self.unused_keys())
    if unused:
      where = owner or self._section or self._config_path or "config"
      raise KeyError(f"{where}: unused config keys {unused}. These values were resolved "
                     "but no declaration or constructor consumed them.")

  def get_optional(self, key: str, default: T = None, expected_type: Optional[Type[T]] = None) -> T:
    try:
      return self.get(key, expected_type)
    except KeyError:
      return default  # type: ignore

  def get_section(self, key: str) -> Dict:
    value = self.get(key)
    if not isinstance(value, dict):
      full_key = f"{self._section}.{key}" if self._section else key
      raise TypeError(f"Config key '{full_key}' is not a section (dict)\n"
                      f"  Got: {type(value).__name__} = {_format_value(value)}")
    return value

  @property
  def root(self) -> Dict:
    return self._root_data

  @property
  def section_data(self) -> Dict:
    return self._data


class Config(ConfigMixin):
  """A whole config file, `$()`-resolved, with typed dotted-path access."""

  @classmethod
  def load(cls, path: str | Path, config_root: Optional[str | Path] = None) -> "Config":
    data = load_yaml_with_includes(path, config_root if config_root is not None else CONFIG_DIR)
    return cls("", data, config_path=str(Path(path).resolve()))

  def __repr__(self) -> str:
    section = f"[{self._section}]" if self._section else ""
    path = f" from {self._config_path}" if self._config_path else ""
    return f"{self.__class__.__name__}{section}{path}"
