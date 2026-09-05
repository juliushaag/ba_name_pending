"""Immutable construction specifications, independent of active config state."""
import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
  name: str
  profile: str | None
  policy_class: str
  payload: str

  @classmethod
  def resolve(cls, name, config, profile=None):
    name, alias_profile = config.resolve_model_name(name)
    profile = profile if profile is not None else alias_profile
    data = {
      "params": config.resolve_params(name, profile),
      "slots": {slot: config.resolve_slot(name, slot, profile) for slot in config.slots(name)},
      "snapshot": config.snapshot_for(name, profile),
      "overrides": config.overrides,
    }
    return cls(name, profile, config.policy_class_name(name), json.dumps(data))

  def data(self):
    return json.loads(self.payload)

  def component(self, slot):
    from barl_jax.registration import get_component
    name, kwargs = self.data()["slots"][slot]
    return get_component(name), kwargs


# `resolved` (a model's `snapshot["resolved"]`, `model.*` -> value) identifies
# an architecture and its hyperparameters alone. A run additionally pins
# batch_size, utd_ratio, chunk_size, ... -- values that change what an
# experiment *is* without changing what model it trained. The two fingerprints
# below are deliberately different digests, not one function serving both:
# two checkpoints of the same model at different run settings must compare
# equal on model_fingerprint and differ on experiment_fingerprint.

# Excluded from the experiment fingerprint because they identify *which*
# instance of the experiment this is, not the experiment itself: two seeds of
# the same run, or the same run resumed under a different sweep_tag/run_dir/
# checkpoint path, are one experiment in wandb's grouping sense.
_EXPERIMENT_IDENTITY_KEYS = frozenset({"seed", "num_seeds", "run_dir", "sweep_tag",
                                       "snapshot", "checkpoint"})


def _fingerprint(payload: dict) -> str:
  return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


def model_fingerprint(resolved: dict) -> str:
  """Stable digest of a resolved model config alone -- "this architecture at
  these hyperparameters", independent of which run trained it. Comparable
  across any two checkpoints/snapshots that resolved the same model."""
  return _fingerprint(resolved)


def experiment_fingerprint(resolved: dict, run_cfg: dict) -> str:
  """Stable digest of a resolved model plus the run settings that define the
  experiment it ran in (batch_size, utd_ratio, ... -- everything but the
  per-instance identity keys in `_EXPERIMENT_IDENTITY_KEYS`). Narrower than
  `model_fingerprint`: the same model at a different run setting is a
  different experiment fingerprint even though it is the same model."""
  payload = {**{f"model.{k}": v for k, v in resolved.items()},
             **{f"run.{k}": v for k, v in run_cfg.items()
                if k not in _EXPERIMENT_IDENTITY_KEYS}}
  return _fingerprint(payload)
