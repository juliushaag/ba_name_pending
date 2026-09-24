"""Generic model-configuration scaffolding.

Component, policy-parameter, and model-variant defaults live in typed frozen
dataclasses beside their implementations -- `StochasticActorConfig` next to
`StochasticActor` (`barl_jax.actors.sac`), `PolicyConfig` next to `RLPolicy`
(`barl_jax.policies.rl`), `SACModelConfig` next to the `SAC` policy class. Each
model variant self-registers into `MODEL_SCHEMAS` via `register_model()` when
its module is imported (`registration.load_builtins("policies")`).

This module owns only what has no single implementation to sit beside: the
model/components composition shape and the model registry.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, fields, replace
from typing import TYPE_CHECKING, Any, ClassVar, Mapping

from barl_jax.config.serialization import construct
from barl_jax.config.serialization import merge as merge_dataclass

if TYPE_CHECKING:
  from barl_jax.policies.rl import PolicyConfig

# Syntax choices for `RunConfig`'s own fields, not capabilities of a
# constructed policy.
LOGGER_KINDS = ("wandb", "local", "none")
CHECKPOINT_MODES = ("weights", "resume")
LR_DECAYS = ("sine", "linear", "cosine", "constant")


def component_to_dict(component: Any) -> dict[str, Any]:
  """Canonical nested form, with ``class`` as the public selector key."""
  out = asdict(component)
  out.pop("class_name", None)
  out["class"] = component.class_name
  return deepcopy(out)


def component_from_dict(values: Mapping[str, Any], *, path: str = "component"):
  """Inverse of `component_to_dict`, built strictly from `values` alone."""
  from barl_jax.registration import get_component
  values = _selector_to_class_name(values)
  if "class_name" not in values:
    raise KeyError(f"{path}: names no 'class'")
  return construct(get_component(values["class_name"]).config_type, values, path=path)


def _selector_to_class_name(values: Mapping[str, Any]) -> dict[str, Any]:
  values = dict(values)
  if "class" in values:
    if "class_name" in values:
      raise KeyError("component mapping cannot contain both 'class' and 'class_name'")
    values["class_name"] = values.pop("class")
  return values


@dataclass(frozen=True)
class ComponentsConfig:
  """The named component slots one model variant builds.

  Slots are typed loosely: each model variant names its own actor/critic/
  regulator config class and there is no common ancestor to name here without
  importing every implementation. `Policy.build_component`'s `gate_cls` check
  enforces a slot's contract at construction time."""

  actor: Any
  critic: Any
  entropy: Any | None = None

  def present(self) -> dict[str, Any]:
    return {f.name: getattr(self, f.name) for f in fields(self)
            if getattr(self, f.name) is not None}

  def to_dict(self) -> dict[str, Any]:
    return {slot: component_to_dict(value) for slot, value in self.present().items()}


@dataclass(frozen=True)
class ModelConfig:
  """One algorithm's complete typed defaults: a policy class name, its own
  params, and the component slots it builds."""

  policy_class: str
  params: "PolicyConfig"
  components: ComponentsConfig

  schema_name: ClassVar[str] = "ModelConfig"

  def to_dict(self) -> dict[str, Any]:
    return {
      "policy": self.policy_class,
      "params": deepcopy(asdict(self.params)),
      "components": self.components.to_dict(),
    }

  @classmethod
  def from_dict(cls, values: Mapping[str, Any]) -> "ModelConfig":
    """Rebuild a model from a complete `to_dict()` mapping, without consulting
    current defaults -- a saved model is authoritative."""
    unknown = sorted(set(values) - {"policy", "params", "components"})
    if unknown:
      raise KeyError(f"model: unknown keys {unknown}; available keys: "
                     "['policy', 'params', 'components']")
    default = cls()
    if values.get("policy", default.policy_class) != default.policy_class:
      raise ValueError(f"model policy {values['policy']!r} does not match {default.policy_class!r}")
    params = construct(cls.__dataclass_fields__["params"].type, values.get("params", {}),
                       path="params")
    slots = values.get("components", {})
    known = {f.name for f in fields(ComponentsConfig)}
    if set(slots) - known:
      raise KeyError(f"model.components: unknown slots {sorted(set(slots) - known)}; "
                     f"available slots: {sorted(known)}")
    components = ComponentsConfig(**{slot: component_from_dict(value, path=f"components.{slot}")
                                     for slot, value in slots.items()})
    return cls(policy_class=default.policy_class, params=params, components=components)

  def overridden(self, overrides: Mapping[str, Any]) -> "ModelConfig":
    """Apply a partial `model_overrides` mapping on top of this model."""
    unknown = sorted(set(overrides) - {"policy", "params", "components"})
    if unknown:
      raise KeyError(f"model_overrides: unknown keys {unknown}; available keys: "
                     "['policy', 'params', 'components']")
    if "policy" in overrides and overrides["policy"] != self.policy_class:
      raise ValueError(f"model_overrides.policy={overrides['policy']!r} does not match "
                       f"selected policy {self.policy_class!r}")
    result = self
    if "params" in overrides:
      result = replace(result, params=merge_dataclass(
        result.params, overrides["params"], path="model_overrides.params"))
    if "components" in overrides:
      entries = overrides["components"]
      if not isinstance(entries, Mapping):
        raise TypeError("model_overrides.components must be a mapping")
      slots = {f.name for f in fields(result.components)}
      if set(entries) - slots:
        raise KeyError(f"model_overrides.components: unknown slots {sorted(set(entries) - slots)}; "
                       f"available slots: {sorted(slots)}")
      values = {}
      for slot, override in entries.items():
        current = getattr(result.components, slot)
        if current is None:
          raise KeyError(f"model_overrides.components.{slot}: slot is not present in {self.schema_name}")
        values[slot] = _override_component(current, override, path=f"model_overrides.components.{slot}")
      result = replace(result, components=replace(result.components, **values))
    return result


def _override_component(base: Any, override: Mapping[str, Any], *, path: str) -> Any:
  if not isinstance(override, Mapping):
    raise TypeError(f"{path} must be a mapping")
  from barl_jax.registration import get_component
  raw = _selector_to_class_name(override)
  selected = raw.pop("class_name", base.class_name)
  if selected != base.class_name:
    base = get_component(selected).config_type()
  return merge_dataclass(base, raw, path=path)


# Populated by `register_model()` as a side effect of importing each algorithm's
# module -- never a second table of values next to the schema classes.
MODEL_SCHEMAS: dict[str, type[ModelConfig]] = {}


def register_model(schema: type[ModelConfig]) -> type[ModelConfig]:
  """Register `schema` under its own `schema_name`, as a decorator beside the
  class it registers."""
  if not isinstance(schema, type) or not issubclass(schema, ModelConfig):
    raise TypeError("model schema must derive from ModelConfig")
  name = schema.schema_name
  if not name or name == ModelConfig.schema_name:
    raise ValueError(f"{schema.__name__} must set a distinct schema_name before registering")
  MODEL_SCHEMAS[name] = schema
  return schema


def model_schema(name: str) -> type[ModelConfig]:
  from barl_jax.registration import load_builtins
  load_builtins("policies")
  try:
    return MODEL_SCHEMAS[name]
  except KeyError as exc:
    raise KeyError(f"unknown model {name!r}; available models: {sorted(MODEL_SCHEMAS)}") from exc
