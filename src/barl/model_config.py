"""YAML-driven architecture configuration (component classes + kwargs, all
in config/models.yaml with no Python-level default class) -- a separate axis
from HParamMixin's algorithm-tuning knobs; see resolve_slot/resolve_component
for the override precedence."""

import os

import yaml

from barl.networks.registry import get_component

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "config")
_MODELS_PATH = os.path.join(_CONFIG_DIR, "models.yaml")

_models_cache: dict = None


def _load_yaml(path: str) -> dict:
  if not os.path.exists(path):
    return {}
  with open(path) as f:
    return yaml.safe_load(f) or {}


def _models() -> dict:
  global _models_cache
  if _models_cache is None:
    _models_cache = _load_yaml(_MODELS_PATH)
  return _models_cache


def resolve_model_name(name: str) -> tuple[str, str | None]:
  """Resolves a `model:` string to (policy_class_name, profile) via
  models.yaml's `aliases:` map; anything not listed there is assumed to
  already be a literal registered Policy class name."""
  alias = _models().get("aliases", {}).get(name)
  if alias:
    return alias["policy"], alias.get("profile")
  return name, None


def resolve_slot(policy_name: str, slot: str, profile: str | None) -> tuple[type, dict]:
  """See module docstring for precedence. `class:` is required -- no
  Python-level fallback -- every other key in the slot's dict is a kwarg."""

  policy_cfg = _models().get(policy_name) or {}

  slot_cfg = dict(policy_cfg.get("components", {}).get(slot, {}))
  cls_name = slot_cfg.pop("class", None)
  if cls_name is None:
    raise KeyError(
      f"{policy_name!r} has no `class:` for its {slot!r} slot in {_MODELS_PATH} -- "
      f"every component must be named explicitly, there is no Python-level default."
    )
  cls = get_component(cls_name)

  resolved = dict(slot_cfg)

  if profile is not None:
    profile_cfg = policy_cfg.get("profiles", {}).get(profile)
    if profile_cfg is None:
      raise KeyError(f"{policy_name!r} has no profile {profile!r} in {_MODELS_PATH}")
    profile_slot_cfg = dict(profile_cfg.get(slot, {}))
    profile_slot_cfg.pop("class", None)  # profiles override kwargs only, not the component class
    resolved.update(profile_slot_cfg)

  return cls, resolved
