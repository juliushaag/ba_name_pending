"""Run-level and environment configuration schemas.

Component, policy-parameter, and model-variant defaults live in typed frozen
dataclasses beside their implementations -- `StochasticActorConfig` next to
`StochasticActor` (`barl_jax.actors.sac`), `PolicyConfig` next to `RLPolicy`
(`barl_jax.policies.rl`), `SACModelConfig` next to the `SAC` policy class that
composes them. Each model variant self-registers into `MODEL_SCHEMAS` via
`register_model()` when its module is imported (`Policy.class_for`/
`RunConfig.defaults` trigger that import through `registration.load_builtins`,
the same mechanism `get_component`/`Policy.class_for` already use).

This module owns only the generic scaffolding that has no single
implementation to sit beside: environment metadata, the model/components
composition shape, and the registries themselves.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
  from barl_jax.policies.rl import PolicyConfig

# Deliberately kept here instead of importing execution/optimization/etc:
# syntax choices for `RunConfig`'s own fields, not capabilities of a
# constructed policy.
LOGGER_KINDS = ("wandb", "local", "none")
CHECKPOINT_MODES = ("weights", "resume")
LR_DECAYS = ("sine", "linear", "cosine", "constant")


@dataclass(frozen=True)
class ComponentsConfig:
  """The named component slots one model variant builds.

  `actor`/`critic`/`entropy` are typed loosely: each concrete model variant
  names its own actor/critic/regulator config class (e.g.
  `StochasticActorConfig`), and there is no single common ancestor to name
  here without re-importing every implementation this module deliberately
  stays independent of. `Policy.build_component`'s `gate_cls` check is what
  actually enforces a slot's contract at construction time.
  """

  actor: Any
  critic: Any
  entropy: Any | None = None

  def to_dict(self) -> dict[str, Any]:
    return {name: component_to_dict(value) for name, value in (
      ("actor", self.actor), ("critic", self.critic), ("entropy", self.entropy))
      if value is not None}


@dataclass(frozen=True)
class ModelConfig:
  """One algorithm's complete typed defaults: a policy class name, its own
  params, and the component slots it builds. Concrete variants (`SACModelConfig`
  beside `SAC`, ...) set `schema_name` and materialize defaults for all three."""

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

  def flat(self) -> dict[str, Any]:
    out = {"policy": self.policy_class}
    out.update({f"params.{k}": deepcopy(v)
                for k, v in asdict(self.params).items()})
    for slot, component in (("actor", self.components.actor),
                            ("critic", self.components.critic),
                            ("entropy", self.components.entropy)):
      if component is None:
        continue
      out.update({f"components.{slot}.{k}": deepcopy(v)
                  for k, v in component_to_runtime_dict(component).items()})
    return out


@dataclass(frozen=True)
class EnvironmentConfig:
  """Static benchmark facts. Live spaces and dimensions are runtime facts."""

  name: str
  return_min: float | None = None
  return_max: float | None = None
  defaults: dict[str, Any] = field(default_factory=dict)


def _env_class(name: str, return_min: float | None = None,
               return_max: float | None = None) -> type[EnvironmentConfig]:
  """Create a named metadata class without importing a simulator module."""
  return dataclass(frozen=True)(type(
    f"{name}EnvironmentConfig", (EnvironmentConfig,),
    {"__annotations__": {"name": str, "return_min": float | None,
                          "return_max": float | None},
     "name": name, "return_min": return_min, "return_max": return_max}))


# Names are intentionally explicit.  The registry is metadata only; resolving
# one of these names never imports gymnasium, MuJoCo, dm-control, or JAX.
ENVIRONMENT_SCHEMAS: dict[str, type[EnvironmentConfig]] = {
  name: _env_class(name, lo, hi)
  for name, lo, hi in (
    ("HalfCheetah", -3200.0, 3200.0), ("Hopper", -900.0, 900.0),
    ("Walker2d", -1400.0, 1400.0), ("Ant", -1600.0, 1600.0),
    ("Humanoid", -1600.0, 1600.0), ("HumanoidStandup", -80000.0, 80000.0),
    ("InvertedPendulum", -200.0, 200.0), ("InvertedDoublePendulum", -2000.0, 2000.0),
    ("Reacher", -160.0, 160.0), ("Swimmer", -80.0, 80.0), ("Pusher", -380.0, 380.0),
    ("DogRun", -200.0, 200.0), ("DogStand", -200.0, 200.0),
    ("DogWalk", -200.0, 200.0), ("DogTrot", -200.0, 200.0),
    ("HumanoidWalk", -200.0, 200.0), ("HumanoidRun", -200.0, 200.0),
    ("HumanoidStand", -200.0, 200.0), ("CartpoleSwingup", -200.0, 200.0),
    ("CartpoleBalance", -200.0, 200.0), ("CheetahRun", -200.0, 200.0),
    ("WalkerWalk", -200.0, 200.0), ("WalkerRun", -200.0, 200.0),
    ("HopperHop", -200.0, 200.0), ("HopperStand", -200.0, 200.0),
    ("QuadrupedWalk", -200.0, 200.0), ("QuadrupedRun", -200.0, 200.0),
    ("FingerSpin", -200.0, 200.0), ("FishSwim", -200.0, 200.0),
    ("AcrobotSwingup", -200.0, 200.0), ("PendulumSwingup", -200.0, 200.0),
    ("BallInCupCatch", -200.0, 200.0), ("ReacherEasy", -200.0, 200.0),
    ("ReacherHard", -200.0, 200.0), ("PushCube", -79.0, 79.0),
    ("PickCube", -79.0, 79.0), ("StackCube", -79.0, 79.0),
    ("PickSingleYCB", -79.0, 79.0), ("PegInsertionSide", -126.79, 126.79),
    ("H1HandWalk", -200.0, 200.0), ("H1HandStand", -200.0, 200.0),
    ("H1HandRun", -200.0, 200.0), ("H1HandCrawl", -200.0, 200.0),
    ("H1HandHurdle", -200.0, 200.0), ("H1HandMaze", -2200.0, 2200.0),
    ("H1HandSitSimple", -200.0, 200.0), ("H1HandSitHard", -200.0, 200.0),
    ("H1HandBalanceSimple", -200.0, 200.0), ("H1HandBalanceHard", -200.0, 200.0),
    ("H1HandStair", -200.0, 200.0), ("H1HandSlide", -200.0, 200.0),
    ("H1HandPole", -200.0, 200.0), ("H1HandHighbarSimple", -200.0, 200.0),
    ("H1HandHighbarHard", -200.0, 200.0), ("H1HandReach", -4000.0, 4000.0),
    ("H1HandPush", -2000.0, 2000.0), ("H1HandCabinet", -2200.0, 2200.0),
    ("H1HandDoor", -200.0, 200.0), ("H1HandTruck", -2800.0, 2800.0),
    ("H1HandCube", -200.0, 200.0), ("H1HandBookshelfSimple", -3000.0, 3000.0),
    ("H1HandBookshelfHard", -3000.0, 3000.0), ("H1HandBasketball", -2000.0, 2000.0),
    ("H1HandWindow", -200.0, 200.0), ("H1HandSpoon", -200.0, 200.0),
    ("H1HandKitchen", -8.0, 8.0), ("H1HandPackage", -2000.0, 2000.0),
    ("H1HandPowerlift", -200.0, 200.0), ("H1HandRoom", -200.0, 200.0),
    ("H1HandInsertNormal", -200.0, 200.0), ("H1HandInsertSmall", -200.0, 200.0),
  )
}


# Populated by `register_model()` as a side effect of importing each algorithm's
# module (see `registration.load_builtins("policies")`) -- never a second table
# of values next to the schema classes themselves.
MODEL_SCHEMAS: dict[str, type[ModelConfig]] = {}
MODEL_ALIASES: dict[str, tuple[str, str | None]] = {}


def register_model(schema: type[ModelConfig], *, aliases: tuple[str, ...] = ()) -> type[ModelConfig]:
  """Register `schema` under its own `schema_name`. Use as a decorator beside
  the class it registers, e.g. `@register_model` above `SACModelConfig`."""
  if not isinstance(schema, type) or not issubclass(schema, ModelConfig):
    raise TypeError("model schema must derive from ModelConfig")
  name = schema.schema_name
  if not name or name == ModelConfig.schema_name:
    raise ValueError(f"{schema.__name__} must set a distinct schema_name before registering")
  MODEL_SCHEMAS[name] = schema
  for alias in aliases:
    MODEL_ALIASES[alias] = (name, None)
  return schema


def register_environment_schema(name: str, schema: type[EnvironmentConfig]) -> None:
  if not isinstance(schema, type) or not issubclass(schema, EnvironmentConfig):
    raise TypeError("environment schema must derive from EnvironmentConfig")
  ENVIRONMENT_SCHEMAS[name] = schema


def model_schema(name: str) -> type[ModelConfig]:
  from barl_jax.registration import load_builtins
  load_builtins("policies")
  canonical, _ = MODEL_ALIASES.get(name, (name, None))
  try:
    return MODEL_SCHEMAS[canonical]
  except KeyError as exc:
    raise KeyError(f"unknown model {name!r}; available models: {sorted(MODEL_SCHEMAS)}") from exc


def resolve_model(name: str) -> tuple[str, str | None, type[ModelConfig]]:
  from barl_jax.registration import load_builtins
  load_builtins("policies")
  canonical, profile = MODEL_ALIASES.get(name, (name, None))
  return canonical, profile, model_schema(canonical)


def environment_schema(name: str) -> type[EnvironmentConfig]:
  try:
    return ENVIRONMENT_SCHEMAS[name]
  except KeyError as exc:
    raise KeyError(f"unknown environment {name!r}; available environments: "
                   f"{sorted(ENVIRONMENT_SCHEMAS)}") from exc


def component_to_dict(component: Any) -> dict[str, Any]:
  """Canonical nested form, with ``class`` as the public selector key."""
  out = asdict(component)
  out.pop("class_name", None)
  out["class"] = component.class_name
  return deepcopy(out)


def component_to_runtime_dict(component: Any) -> dict[str, Any]:
  """Flat legacy form used for `RunConfig.flat_model()`/wandb config logging --
  architecture-shape fields only (size, MLP surface, time coder, noise
  schedule). The complete value tree already travels alongside it as
  `model_overrides`; this is a projection for logging, not the source of
  truth (see `docs/plans/config-class-restructure.md`)."""
  out: dict[str, Any] = {"class": component.class_name}
  for key in ("latent_dim", "layers", "n_models"):
    value = getattr(component, key, None)
    if value is not None:
      out[key] = value
  mlp = getattr(component, "mlp", None)
  if mlp is not None:
    out.update(asdict(mlp))
  time = getattr(component, "time", None)
  if time is not None:
    out.update(time.to_runtime_dict())
  schedule = getattr(component, "noise_schedule", None)
  if schedule is not None:
    out["noise_schedule"] = component_to_dict(schedule)
  return deepcopy(out)


def model_to_runtime_dict(model: ModelConfig) -> dict[str, Any]:
  """Convert a typed model to the flat mapping consumed by legacy builders."""
  return model.flat()


__all__ = [name for name in globals() if not name.startswith("_")]
