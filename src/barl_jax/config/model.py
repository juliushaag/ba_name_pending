"""Checkpoint model specifications derived from typed model configs."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from barl_jax.config.serialization import to_dict


@dataclass(frozen=True)
class ModelSpec:
  """Stable, JSON-only model architecture boundary for checkpoints."""

  name: str
  profile: str | None
  policy_class: str
  payload: str
  schema_version: int = 1

  @classmethod
  def resolve(cls, name: str, config=None, profile: str | None = None):
    # Import lazily: a caller asking for config defaults should not load this
    # checkpoint boundary or any runtime module.
    from barl_jax.config.run import RunConfig
    if isinstance(config, RunConfig):
      run = config
      if name != run.model:
        raise ValueError(f"model spec name {name!r} does not match run model {run.model!r}")
      if profile is not None and profile != run.profile:
        raise ValueError(f"model spec profile {profile!r} does not match run profile {run.profile!r}")
      model = run.model_config
      selected_profile = run.profile
    elif isinstance(config, dict):
      # Small Python callers (tests, notebooks) pass a nested model mapping
      # directly; accept the canonical shape with `policy`/`params`/`components`.
      raise TypeError("ModelSpec.resolve needs a RunConfig or model name, not a plain dict; "
                      "use RunConfig.from_dict for mappings")
    else:
      # ModelSpec is also useful before a run exists.  This uses the typed
      # model registry only and leaves env-specific fields out of its payload.
      from barl_jax.config.schemas import resolve_model
      canonical, alias_profile, model_cls = resolve_model(name)
      selected_profile = profile if profile is not None else alias_profile
      model = model_cls()
      if selected_profile is not None:
        from barl_jax.config.run import _profile_model
        model = _profile_model(canonical, model, selected_profile)
      name = canonical
    # Canonical nested form: `model.to_dict()` emits `policy`/`params`/
    # `components` with `class` as the public selector key, so a saved file
    # loads without any package default tables.
    if hasattr(model, "to_dict"):
      model_dict = model.to_dict()
    else:
      model_dict = to_dict(model)
    data = {"schema_version": cls.schema_version, "name": name,
            "profile": selected_profile, "model": model_dict}
    return cls(name, selected_profile, model.policy_class if hasattr(model, "policy_class") else model_dict["policy"],
               json.dumps(data, sort_keys=True, separators=(",", ":")))

  def data(self) -> dict[str, Any]:
    return json.loads(self.payload)

  def model(self) -> dict[str, Any]:
    return self.data()["model"]

  def params(self) -> dict[str, Any]:
    return self.model().get("params", {})

  def component(self, slot: str):
    from barl_jax.registration import get_component
    try:
      component = self.model()["components"][slot]
    except KeyError as exc:
      raise KeyError(f"model {self.name!r} has no component slot {slot!r}") from exc
    # Canonical form uses `class`; older generic payloads used `class_name`.
    class_name = component.get("class", component.get("class_name"))
    if class_name is None:
      raise KeyError(f"model {self.name!r} component {slot!r} names no class")
    return get_component(class_name), component


_EXPERIMENT_IDENTITY_KEYS = frozenset({"seed", "num_seeds", "run_dir", "sweep_tag",
                                       "snapshot", "checkpoint"})


def _fingerprint(payload: dict) -> str:
  return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


def model_fingerprint(resolved: dict) -> str:
  return _fingerprint(resolved)


def experiment_fingerprint(resolved: dict, run_cfg: dict) -> str:
  payload = {**{f"model.{k}": v for k, v in resolved.items()},
             **{f"run.{k}": v for k, v in run_cfg.items()
                if k not in _EXPERIMENT_IDENTITY_KEYS}}
  return _fingerprint(payload)


__all__ = ["ModelSpec", "model_fingerprint", "experiment_fingerprint"]
