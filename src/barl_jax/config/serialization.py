"""Small dataclass construction and serialization primitives for config classes."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
from types import UnionType
from typing import Any, Mapping, Union, get_args, get_origin, get_type_hints


def type_name(expected: Any) -> str:
  origin = get_origin(expected)
  if origin in (Union, UnionType):
    return " | ".join(type_name(x) for x in get_args(expected))
  if origin is not None:
    args = get_args(expected)
    return f"{getattr(origin, '__name__', origin)}[{', '.join(type_name(x) for x in args)}]"
  return getattr(expected, "__name__", str(expected))


def is_instance(value: Any, expected: Any) -> bool:
  origin = get_origin(expected)
  if origin in (Union, UnionType):
    return any(is_instance(value, arg) for arg in get_args(expected))
  if origin is list:
    return isinstance(value, list) and all(is_instance(x, get_args(expected)[0]) for x in value)
  if origin is tuple:
    args = get_args(expected)
    return isinstance(value, tuple) and (not args or len(args) == len(value)) and all(
      is_instance(x, args[min(i, len(args) - 1)]) for i, x in enumerate(value))
  if origin is dict:
    args = get_args(expected)
    return isinstance(value, dict) and (not args or all(
      is_instance(k, args[0]) and is_instance(v, args[1]) for k, v in value.items()))
  if expected is Any:
    return True
  if expected is float:
    return isinstance(value, (float, int)) and not isinstance(value, bool)
  if expected is int:
    return isinstance(value, int) and not isinstance(value, bool)
  if expected is bool:
    return isinstance(value, bool)
  try:
    return isinstance(value, expected)
  except TypeError:
    return True


def _dataclass_type(expected: Any) -> type | None:
  if isinstance(expected, type) and is_dataclass(expected):
    return expected
  origin = get_origin(expected)
  if origin in (Union, UnionType):
    for arg in get_args(expected):
      found = _dataclass_type(arg)
      if found is not None:
        return found
  return None


def variant_registry():
  """(registry, register) for a polymorphic config family: subclasses of its base
  register under their `class_name`, and the base exposes the dict as `variants`."""
  registry: dict[str, type] = {}

  def register(cls: type) -> type:
    registry[cls.class_name] = cls
    return cls

  return registry, register


def _variant(cls: type, name: str, path: str) -> type:
  """The registered subclass of a polymorphic config (`cls.variants`) named `name`."""
  try:
    return cls.variants[name]
  except KeyError:
    raise KeyError(f"{path or cls.__name__}: unknown class_name {name!r}; "
                   f"available: {sorted(cls.variants)}") from None


def construct(cls: type, values: Mapping[str, Any], *, path: str = ""):
  """Construct ``cls`` recursively, rejecting unknown keys and bad types. A config
  with a `variants` registry is built as the variant its `class_name` names."""
  if not isinstance(values, Mapping):
    raise TypeError(f"{path or 'config'} must be a mapping, got {type(values).__name__}")
  if hasattr(cls, "variants") and "class_name" in values:
    cls = _variant(cls, values["class_name"], path)
  declared = {f.name for f in fields(cls) if f.init}
  unknown = sorted(set(values) - declared)
  if unknown:
    raise KeyError(f"{path or cls.__name__}: unknown keys {unknown}; "
                   f"available keys: {sorted(declared)}")
  hints = get_type_hints(cls)
  kwargs = {}
  for f in fields(cls):
    if not f.init or f.name not in values:
      continue
    key = f"{path}.{f.name}" if path else f.name
    value = values[f.name]
    expected = hints.get(f.name, f.type)
    nested = _dataclass_type(expected)
    if nested is not None:
      kwargs[f.name] = construct(nested, value, path=key)
    elif not is_instance(value, expected):
      raise TypeError(f"{key} has wrong type: expected {type_name(expected)}, "
                      f"got {type(value).__name__} = {value!r}")
    else:
      kwargs[f.name] = deepcopy(value)
  return cls(**kwargs)


def merge(instance, values: Mapping[str, Any], *, path: str = ""):
  """Strictly apply a nested mapping to an immutable dataclass."""
  if not isinstance(values, Mapping):
    raise TypeError(f"{path or 'config'} must be a mapping, got {type(values).__name__}")
  if hasattr(instance, "variants") and values.get("class_name", instance.class_name) != instance.class_name:
    # Switching variants starts from the new variant's defaults; the old one's
    # fields do not carry over.
    return construct(_variant(type(instance), values["class_name"], path), values, path=path)
  declared = {f.name for f in fields(instance) if f.init}
  unknown = sorted(set(values) - declared)
  if unknown:
    raise KeyError(f"{path or type(instance).__name__}: unknown keys {unknown}; "
                   f"available keys: {sorted(declared)}")
  hints = get_type_hints(type(instance))
  updates = {}
  for f in fields(instance):
    if not f.init or f.name not in values:
      continue
    key = f"{path}.{f.name}" if path else f.name
    value = values[f.name]
    current = getattr(instance, f.name)
    if is_dataclass(current) and isinstance(value, Mapping):
      updates[f.name] = merge(current, value, path=key)
    elif not is_instance(value, hints.get(f.name, f.type)):
      raise TypeError(f"{key} has wrong type: expected {type_name(hints.get(f.name, f.type))}, "
                      f"got {type(value).__name__} = {value!r}")
    else:
      updates[f.name] = deepcopy(value)
  return replace(instance, **updates)


def to_dict(value: Any) -> Any:
  """Return a detached JSON-safe tree for a dataclass or builtin container."""
  if is_dataclass(value):
    return {f.name: to_dict(getattr(value, f.name)) for f in fields(value) if f.init}
  if isinstance(value, Mapping):
    return {str(k): to_dict(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [to_dict(v) for v in value]
  if isinstance(value, (str, int, float, bool)) or value is None:
    return value
  # Numpy scalars and similar plugin values commonly expose item().
  item = getattr(value, "item", None)
  if callable(item):
    return to_dict(item())
  raise TypeError(f"{type(value).__name__} is not JSON-safe")


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
  """Flatten a dataclass mapping for legacy runtime/checkpoint boundaries."""
  tree = to_dict(value)
  out = {}
  def visit(node, path):
    if isinstance(node, Mapping):
      for key, child in node.items():
        visit(child, f"{path}.{key}" if path else key)
    else:
      out[f"{prefix}.{path}" if prefix and path else prefix or path] = node
  visit(tree, "")
  return out
