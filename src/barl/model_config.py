"""YAML-driven configuration: component classes + kwargs (resolve_slot) and
each Policy's own top-level tunables (resolve_params), all in
config/models.yaml with no Python-level defaults -- see resolve_slot/
resolve_component/resolve_params for the override precedence, and
set_models_override()/snapshot_for() for how a sweep run's embedded `models:`
snapshot reproduces its exact resolved config later."""

import os

import yaml

from barl.networks.registry import get_component

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "config")
_MODELS_PATH = os.path.join(_CONFIG_DIR, "models.yaml")

_models_cache: dict = None
_models_override: dict | None = None


def _load_yaml(path: str) -> dict:
  if not os.path.exists(path):
    return {}
  with open(path) as f:
    return yaml.safe_load(f) or {}


def _models() -> dict:
  global _models_cache
  if _models_cache is None:
    _models_cache = _load_yaml(_MODELS_PATH)
  return {**_models_cache, **_models_override} if _models_override else _models_cache


def set_models_override(override: dict | None):
  """Layers `override` on top of config/models.yaml for the rest of the
  process -- lets a saved per-run YAML's embedded `models:` snapshot (see
  scripts/run_sweep.py / snapshot_for()) reproduce its exact resolved
  architecture/params even if config/models.yaml has since changed. Called
  once per run by scripts/online_learning.py's main()."""
  global _models_override
  _models_override = override


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


def resolve_params(policy_name: str, profile: str | None) -> dict:
  """Like resolve_slot but for a Policy's own top-level tunables
  (models.yaml's `params:` block) rather than a named actor/critic/squash/...
  slot -- no `class:` involved, just a flat kwargs dict, profile-overridable
  the same way."""
  policy_cfg = _models().get(policy_name) or {}
  params = dict(policy_cfg.get("params", {}))

  if profile is not None:
    profile_cfg = policy_cfg.get("profiles", {}).get(profile)
    if profile_cfg is None:
      raise KeyError(f"{policy_name!r} has no profile {profile!r} in {_MODELS_PATH}")
    params.update(profile_cfg.get("params", {}))

  return params


def snapshot_for(name: str) -> dict:
  """Minimal models.yaml sub-dict needed to reproduce resolving `name`
  later, regardless of subsequent edits to config/models.yaml: the aliases:
  entry (if `name` names one) plus the resolved policy's full block. Embedded
  verbatim by scripts/run_sweep.py into each run's per-run YAML, and logged
  to wandb by scripts/online_learning.py."""
  models = _models()
  out: dict = {}
  alias = models.get("aliases", {}).get(name)
  if alias:
    out["aliases"] = {name: alias}
  policy_name, _ = resolve_model_name(name)
  if policy_name in models:
    out[policy_name] = models[policy_name]
  return out
