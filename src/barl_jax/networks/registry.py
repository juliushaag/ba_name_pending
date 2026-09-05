"""String-keyed registry for concrete Actor/Critic/Squash classes -- lets
config/models.yaml name a class (`class: HLGaussCritic`) instead of a Python
import path. Same shape as Policy._classes/Environment._classes, but callers
construct the resolved class themselves (see model_config.resolve_slot)."""

_COMPONENT_REGISTRY: dict[str, type] = {}


def register_component(cls: type) -> type:
  name = cls.__name__
  if name in _COMPONENT_REGISTRY:
    raise ValueError(f"Component {name} already registered")
  _COMPONENT_REGISTRY[name] = cls
  return cls


def get_component(name: str) -> type:
  from barl_jax.registration import available_names, import_type, load_builtins, resolve_named
  load_builtins("components")
  if ":" in name:
    return import_type(name)
  from barl_jax.components import Component
  cls = resolve_named("components", name, _COMPONENT_REGISTRY, Component)
  if cls is None:
    raise KeyError(f"{name} not in registered components: "
                   f"{available_names('components', _COMPONENT_REGISTRY)}")
  return cls


def available_components() -> list[str]:
  from barl_jax.registration import available_names, load_builtins
  load_builtins("components")
  return available_names("components", _COMPONENT_REGISTRY)
