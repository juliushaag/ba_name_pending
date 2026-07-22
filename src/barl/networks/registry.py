"""String-keyed registry for concrete Actor/Critic/Squash classes -- lets
config/models.yaml name a class (`class: QRDistributionalCritic`) instead of a
Python import path. Same shape as Policy._classes/Environment._classes, but
callers construct the resolved class themselves (see model_config.resolve_slot)."""

_COMPONENT_REGISTRY: dict[str, type] = {}


def register_component(cls: type) -> type:
  name = cls.__name__
  if name in _COMPONENT_REGISTRY:
    raise ValueError(f"Component {name} already registered")
  _COMPONENT_REGISTRY[name] = cls
  return cls


def get_component(name: str) -> type:
  if name not in _COMPONENT_REGISTRY:
    raise KeyError(f"{name} not in registered components: {list(_COMPONENT_REGISTRY.keys())}")
  return _COMPONENT_REGISTRY[name]


def available_components() -> list[str]:
  return list(_COMPONENT_REGISTRY.keys())
