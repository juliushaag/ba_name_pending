"""Immutable construction specifications, independent of active config state."""
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
