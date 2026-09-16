"""Typed training-run configuration and the deliberately small YAML loader."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, get_type_hints

if TYPE_CHECKING:
  from barl_jax.policies.rl import PolicyConfig

from barl_jax.config.schemas import (
  CHECKPOINT_MODES,
  LOGGER_KINDS,
  LR_DECAYS,
  ComponentsConfig,
  EnvironmentConfig,
  ModelConfig,
  component_to_dict,
  environment_schema,
  model_schema,
  model_to_runtime_dict,
  resolve_model,
)
from barl_jax.config.serialization import merge as deep_merge_dataclass
from barl_jax.executors import DirectExecutorConfig, ExecutorConfig
from barl_jax.optimization import CLIP_MODES
from barl_jax.registration import available_executors, get_component, get_executor


@dataclass(frozen=True)
class RunConfig:
  """One complete immutable training run.

  ``model_config`` is materialized from a schema class and contains all model
  values, including values hidden behind tier/style selectors.  The public
  input spelling is ``model_overrides``; its canonical output is the complete
  nested model object so a serialized run is self-contained.
  """

  model: str
  env: str
  profile: str | None = None
  checkpoint: str | None = None
  checkpoint_mode: str = "weights"
  seed: int = 0
  num_seeds: int = 3
  validate_config: bool = False
  run_dir: str | None = None
  sweep_tag: str | None = None
  logger: str = "wandb"
  total_env_steps: int = 1_000_000
  random_steps: int = 5000
  rollout_steps: int = 1
  num_envs: int = 1
  chunk_size: int = 1
  buffer_size: int = 1_000_000
  batch_size: int = 256
  utd_ratio: float = 2.0
  lr_warmup_steps: int = 0
  lr_decay: str = "constant"
  eval_every: int = 25_000
  eval_runs: int = 10
  eval_video_every: int | None = 4
  log_every: int = 1000
  debug_every: int = 10000
  wandb_finish_timeout: int = 300
  compile_cache_dir: str | None = "~/.cache/barl_jax/jax"
  resume_every: int | None = 50_000
  resume_keep: int = 2
  collection_executor: ExecutorConfig = field(default_factory=DirectExecutorConfig)
  eval_executor: ExecutorConfig = field(default_factory=DirectExecutorConfig)
  snapshot: dict[str, Any] | None = None
  model_config: ModelConfig = field(default_factory=lambda: model_schema("SAC")())
  env_config: EnvironmentConfig = field(default_factory=lambda: environment_schema("HalfCheetah")())

  @classmethod
  def defaults(cls, model: str, env: str, profile: str | None = None) -> "RunConfig":
    selected, alias_profile, model_cls = resolve_model(model)
    if profile is not None and alias_profile is not None and profile != alias_profile:
      raise ValueError(f"model {model!r} selects profile {alias_profile!r}, "
                       f"cannot also use profile {profile!r}")
    profile = profile if profile is not None else alias_profile
    try:
      env_cls = environment_schema(env)
    except KeyError:
      # External environments (`module:Class` paths, e.g. the CPU quickstart's
      # tiny pendulum) have no static benchmark metadata; they still get a
      # complete run object with empty defaults. Live spaces/dimensions are
      # runtime facts from EnvStats, never config defaults.
      if ":" not in env:
        raise
      from barl_jax.config.schemas import EnvironmentConfig as _EnvConfig
      env_cls = _EnvConfig
      return cls(model=model, env=env, profile=profile,
                 model_config=model_cls() if profile is None else _profile_model(selected, model_cls(), profile),
                 env_config=_EnvConfig(name=env))
    # A schema class is the only source of built-in model defaults.  Calling it
    # allocates ordinary Python dataclasses only; it does not touch JAX or a
    # simulator registry.
    model_value = model_cls()
    if profile is not None:
      model_value = _profile_model(selected, model_value, profile)
    return cls(model=model, env=env, profile=profile,
               model_config=model_value, env_config=env_cls())

  @classmethod
  def from_dict(cls, mapping: Mapping[str, Any]) -> "RunConfig":
    if not isinstance(mapping, Mapping):
      raise TypeError(f"run config must be a mapping, got {type(mapping).__name__}")
    if "runs" in mapping or "defaults" in mapping:
      raise ValueError("training config must describe one run; `runs` and `defaults` "
                       "envelopes are not supported")
    for key in ("model", "env"):
      if key not in mapping:
        raise KeyError(f"run config is missing required selector {key!r}")
    if not isinstance(mapping["model"], str) or not isinstance(mapping["env"], str):
      raise TypeError("model and env selectors must be strings")

    base = cls.defaults(mapping["model"], mapping["env"], mapping.get("profile"))
    allowed = {f.name for f in fields(cls)} - {"model_config", "env_config", "validate_config"}
    allowed |= {"validate", "model_overrides", "env_metadata"}
    unknown = sorted(set(mapping) - allowed)
    if unknown:
      available = sorted(allowed - {"model_overrides"}) + ["model_overrides"]
      raise KeyError(f"run config: unknown keys {unknown}; available keys: {available}")

    result = base
    run_updates = {}
    for f in fields(cls):
      if f.name in ("model", "env", "profile", "snapshot", "model_config", "env_config",
                    "collection_executor", "eval_executor"):
        continue
      input_name = "validate" if f.name == "validate_config" else f.name
      if input_name not in mapping:
        continue
      value = mapping[input_name]
      expected = get_type_hints(cls)[f.name]
      from barl_jax.config.serialization import is_instance as _is_instance
      from barl_jax.config.serialization import type_name as _type_name
      if not _is_instance(value, expected):
        raise TypeError(f"{input_name} has wrong type: expected {_type_name(expected)}, "
                        f"got {type(value).__name__} = {value!r}")
      run_updates[f.name] = deepcopy(value)
    if "snapshot" in mapping:
      if mapping["snapshot"] is not None and not isinstance(mapping["snapshot"], Mapping):
        raise TypeError("snapshot must be a mapping or null")
      run_updates["snapshot"] = (None if mapping["snapshot"] is None
                                  else deepcopy(dict(mapping["snapshot"])))
    if run_updates:
      result = replace(result, **run_updates)

    executor_updates = {}
    for key in ("collection_executor", "eval_executor"):
      if key in mapping:
        executor_updates[key] = _resolve_executor_config(mapping[key], path=key)
    if executor_updates:
      result = replace(result, **executor_updates)

    if "env_metadata" in mapping:
      metadata = mapping["env_metadata"]
      if not isinstance(metadata, Mapping):
        raise TypeError("env_metadata must be a mapping")
    overrides = mapping.get("model_overrides")
    if overrides is not None:
      if not isinstance(overrides, Mapping):
        raise TypeError("model_overrides must be a mapping")
      result = replace(result, model_config=_apply_model_overrides(
        result.model_config, overrides))
    result.validate()
    return result

  @property
  def params(self) -> PolicyConfig:
    return self.model_config.params

  @property
  def components(self) -> ComponentsConfig:
    return self.model_config.components

  @property
  def policy(self) -> str:
    return self.model_config.policy_class

  def to_dict(self) -> dict[str, Any]:
    out = {}
    for f in fields(self):
      if f.name in ("model_config", "env_config", "collection_executor", "eval_executor"):
        continue
      output_name = "validate" if f.name == "validate_config" else f.name
      out[output_name] = deepcopy(getattr(self, f.name))
    # Plain-dict, not the live dataclass instance: `training_checkpoint.py`'s
    # validate_resume() diffs this against a JSON-round-tripped saved copy
    # field by field, and a dataclass instance never compares equal to the
    # dict json.loads() hands back for the same value.
    out["collection_executor"] = component_to_dict(self.collection_executor)
    out["eval_executor"] = component_to_dict(self.eval_executor)
    # Keep the input section name, but write the complete effective model.  A
    # saved file can therefore be loaded without any package default tables.
    out["model_overrides"] = self.model_config.to_dict()
    out["env_metadata"] = {
      "name": self.env_config.name,
      "return_min": self.env_config.return_min,
      "return_max": self.env_config.return_max,
      "defaults": deepcopy(self.env_config.defaults),
    }
    return deepcopy(out)

  def to_runtime_dict(self) -> dict[str, Any]:
    out = {k: deepcopy(v) for k, v in self.to_dict().items()
           if k not in ("model_overrides", "env_metadata")}
    out.update(model_to_runtime_dict(self.model_config))
    out["model_overrides"] = self.model_config.to_dict()
    return deepcopy(out)

  def flat_model(self) -> dict[str, Any]:
    return model_to_runtime_dict(self.model_config)

  def validate(self) -> None:
    problems = _problems(self)
    if problems:
      raise ValueError("invalid run config\n" + "\n".join(f"- {p}" for p in problems))

  def problems(self) -> list[str]:
    return _problems(self)

  def replace(self, **changes) -> "RunConfig":
    """Dataclass replacement with selector-aware rebuilding."""
    selectors = {k for k in ("model", "env", "profile") if k in changes}
    if selectors:
      selected_model = changes.get("model", self.model)
      selected_env = changes.get("env", self.env)
      selected_profile = changes.get("profile", self.profile)
      rebuilt = type(self).defaults(selected_model, selected_env, selected_profile)
      for f in fields(self):
        if f.name not in selectors and f.name not in ("model_config", "env_config"):
          changes.setdefault(f.name, getattr(self, f.name))
      changes.setdefault("model_config", rebuilt.model_config)
      changes.setdefault("env_config", rebuilt.env_config)
    return replace(self, **changes)


# Run fields are retained as a plain mapping for small legacy callers.  They no
# longer carry Config declarations and are never consulted by RunConfig.defaults.
# collection_executor/eval_executor are `default_factory` fields -- `f.default`
# is `dataclasses.MISSING` for those, so they're excluded from the comprehension
# and given their real (canonical-dict) default explicitly instead.
RUN_CONFIGS = {("validate" if f.name == "validate_config" else f.name): f.default for f in fields(RunConfig)
               if f.name not in {"model", "env", "profile", "snapshot", "model_config", "env_config",
                                 "collection_executor", "eval_executor"}}
RUN_CONFIGS["collection_executor"] = {"class": "DirectExecutor"}
RUN_CONFIGS["eval_executor"] = {"class": "DirectExecutor"}
INJECTED_KEYS = frozenset({"model", "env", "profile", "model_overrides", "snapshot"})


def run_defaults() -> dict[str, Any]:
  return deepcopy(RUN_CONFIGS)


def run_keys(root=None) -> set[str]:
  return set(RUN_CONFIGS)


def validate_run(cfg: RunConfig | Mapping[str, Any], *, num_seeds: int | None = None) -> list[str]:
  """Return diagnostics for callers that aggregate preflight errors."""
  if isinstance(cfg, RunConfig):
    problems = cfg.problems()
  else:
    problems = _mapping_problems(cfg)
  if num_seeds is not None:
    if not isinstance(num_seeds, int) or isinstance(num_seeds, bool) or num_seeds < 1:
      problems.append(f"num_seeds={num_seeds!r}: must be an int >= 1")
  return problems


def load_config(path: str | Path, root=None) -> list[dict[str, Any]]:
  """Compatibility boundary returning one runtime mapping for one YAML run."""
  from barl_jax.config import from_yaml
  run = from_yaml(path)
  return [run.to_runtime_dict()]


def _profile_model(name: str, value: ModelConfig, profile: str) -> ModelConfig:
  """Resolve a typed profile class if a plugin registered one.

  Built-ins currently have no profiles.  A model schema can expose
  ``profiles`` as a class-level mapping of profile names to schema classes;
  this keeps profile lookup class-owned instead of adding a defaults table.
  """
  profiles = getattr(type(value), "profiles", {})
  if profile not in profiles:
    available = sorted(profiles)
    raise KeyError(f"model {name!r} has no profile {profile!r}; available profiles: {available}")
  profile_schema = profiles[profile]
  if isinstance(profile_schema, type) and issubclass(profile_schema, ModelConfig):
    return profile_schema()
  if isinstance(profile_schema, ModelConfig):
    return deepcopy(profile_schema)
  raise TypeError(f"profile {profile!r} for {name!r} is not a ModelConfig schema")


def _normal_component_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
  value = dict(value)
  if "class" in value:
    if "class_name" in value:
      raise KeyError("component override cannot contain both 'class' and 'class_name'")
    value["class_name"] = value.pop("class")
  return value


def _resolve_executor_config(raw: Mapping[str, Any], *, path: str) -> ExecutorConfig:
  """A `class`/`class_name`-tagged mapping -> a fresh, fully typed executor
  config. Unlike `model_overrides.components.<slot>` (`_apply_component_override`
  below) there is no prior instance to merge onto here -- `collection_executor`/
  `eval_executor` are always constructed from scratch, so a mapping that names
  the field at all must name its class too."""
  if not isinstance(raw, Mapping):
    raise TypeError(f"{path} must be a mapping, got {type(raw).__name__}")
  value = _normal_component_mapping(raw)
  if "class_name" not in value:
    raise KeyError(f"{path}: must name a 'class' (available executors: {available_executors()})")
  name = value.pop("class_name")
  config_type = get_executor(name).config_type
  from barl_jax.config.serialization import construct
  try:
    return construct(config_type, value, path=path)
  except (KeyError, TypeError) as exc:
    raise KeyError(f"{path}: cannot materialize {config_type.__name__}: {exc}") from exc


def _selected_time_style(name: str):
  """A named time-coder preset as a complete `TimeEmbeddingConfig` -- the
  `time_emb:` override shortcut. Lazily imported: only an override that
  actually names a `time_emb` style needs the (jax-touching) time-embedding
  module at all."""
  from barl_jax.networks.time_embedding import TimeEmbeddingConfig
  styles = {
    "sin_16": TimeEmbeddingConfig(kind="sin", time_dim=8, num_freqs=16),
    "sin_20": TimeEmbeddingConfig(kind="sin", time_dim=8, num_freqs=20),
    "fourier": TimeEmbeddingConfig(kind="fourier", time_dim=16),
    "ctrl": TimeEmbeddingConfig(kind="fourier", time_dim=16),
    "pisgrad": TimeEmbeddingConfig(kind="fourier", time_dim=256, hidden=256,
                                    num_freqs=256, concat_input=False, spacing="linear"),
  }
  try:
    return styles[name]
  except KeyError as exc:
    raise KeyError(f"unknown time_emb {name!r}; available styles: {list(styles)}") from exc


def _apply_component_override(base: Any, override: Mapping[str, Any], *, path: str) -> Any:
  if not isinstance(override, Mapping):
    raise TypeError(f"{path} must be a mapping")
  raw = _normal_component_mapping(override)
  selected_name = raw.get("class_name", base.class_name)
  # A class selector is resolved first, from its own config_type -- the same
  # typed default a fresh Policy.build_component() would materialize.
  if selected_name != base.class_name:
    base = get_component(selected_name).config_type()
  if "time_emb" in raw:
    raw_time = _selected_time_style(raw["time_emb"])
    if hasattr(base, "time"):
      base = replace(base, time=raw_time)
  # Selector keys are now represented by the selected dataclass.  Apply the
  # ordinary nested override after selector surfaces have been expanded.
  raw.pop("class_name", None)
  result = deep_merge_dataclass(base, raw, path=path)
  return result


def _apply_model_overrides(base: ModelConfig, overrides: Mapping[str, Any]) -> ModelConfig:
  unknown = sorted(set(overrides) - {"policy", "policy_class", "params", "components"})
  if unknown:
    raise KeyError(f"model_overrides: unknown keys {unknown}; available keys: "
                   "['policy', 'params', 'components']")
  if "policy" in overrides and overrides["policy"] != base.policy_class:
    raise ValueError(f"model_overrides.policy={overrides['policy']!r} does not match "
                     f"selected policy {base.policy_class!r}")
  if "policy_class" in overrides and overrides["policy_class"] != base.policy_class:
    raise ValueError(f"model_overrides.policy_class={overrides['policy_class']!r} does not match "
                     f"selected policy {base.policy_class!r}")
  result = base
  if "params" in overrides:
    if not isinstance(overrides["params"], Mapping):
      raise TypeError("model_overrides.params must be a mapping")
    result = replace(result, params=deep_merge_dataclass(
      result.params, overrides["params"], path="model_overrides.params"))
  if "components" in overrides:
    entries = overrides["components"]
    if not isinstance(entries, Mapping):
      raise TypeError("model_overrides.components must be a mapping")
    slots = {f.name for f in fields(result.components)}
    unknown_slots = sorted(set(entries) - slots)
    if unknown_slots:
      raise KeyError(f"model_overrides.components: unknown slots {unknown_slots}; "
                     f"available slots: {sorted(slots)}")
    component_values = {}
    for slot, override in entries.items():
      current = getattr(result.components, slot)
      if current is None:
        raise KeyError(f"model_overrides.components.{slot}: slot is not present in {base.schema_name}")
      component_values[slot] = _apply_component_override(
        current, override, path=f"model_overrides.components.{slot}")
    if component_values:
      result = replace(result, components=replace(result.components, **component_values))
  return result


def _mapping_problems(cfg: Mapping[str, Any]) -> list[str]:
  if not isinstance(cfg, Mapping):
    return [f"config has type {type(cfg).__name__}, expected mapping"]
  # A mapping passed by an older direct trainer caller cannot carry a typed
  # model object, but run-level diagnostics remain useful and side-effect free.
  problems: list[str] = []
  from barl_jax.config.serialization import is_instance as _is_instance
  from barl_jax.config.serialization import type_name as _type_name
  hints = get_type_hints(RunConfig)
  for key, expected in hints.items():
    input_key = "validate" if key == "validate_config" else key
    if key in ("model_config", "env_config", "collection_executor", "eval_executor") \
       or input_key not in cfg:
      continue
    if not _is_instance(cfg[input_key], expected):
      problems.append(f"{input_key}={cfg[input_key]!r}: expected {_type_name(expected)}")
  problems.extend(_cross_run_problems(dict(cfg)))
  return problems


def _problems(config: RunConfig) -> list[str]:
  problems = _mapping_problems(config.to_runtime_dict())
  # to_runtime_dict includes flat model keys that _mapping_problems ignores;
  # validate the selected model and nested config directly here.
  try:
    resolve_model(config.model)
  except KeyError as exc:
    problems.append(str(exc))
  if ":" not in config.env:
    try:
      environment_schema(config.env)
    except KeyError as exc:
      problems.append(str(exc))
  problems.extend(_model_problems(config.model_config))
  return problems


def _cross_run_problems(cfg: Mapping[str, Any]) -> list[str]:
  problems: list[str] = []
  def fail(key, value, relation):
    problems.append(f"{key}={value!r}: {relation}")
  positive = ("total_env_steps", "rollout_steps", "num_envs", "chunk_size",
              "buffer_size", "batch_size", "eval_every", "eval_runs",
              "log_every", "debug_every", "resume_keep")
  nonnegative = ("random_steps", "lr_warmup_steps", "wandb_finish_timeout")
  optional_positive = ("eval_video_every", "resume_every")
  for key in positive:
    value = cfg.get(key)
    if isinstance(value, int) and not isinstance(value, bool) and value < 1:
      fail(key, value, "must be >= 1")
  for key in nonnegative:
    value = cfg.get(key)
    if isinstance(value, int) and not isinstance(value, bool) and value < 0:
      fail(key, value, "must be >= 0")
  for key in optional_positive:
    value = cfg.get(key)
    if value is not None and isinstance(value, int) and not isinstance(value, bool) and value < 1:
      fail(key, value, "must be >= 1 or null")
  utd = cfg.get("utd_ratio")
  if isinstance(utd, (int, float)) and not isinstance(utd, bool) and (not math.isfinite(utd) or utd <= 0):
    fail("utd_ratio", utd, "must be a finite number > 0")
  mode = cfg.get("checkpoint_mode", "weights")
  if mode not in CHECKPOINT_MODES:
    fail("checkpoint_mode", mode, f"must be one of {CHECKPOINT_MODES}")
  elif mode == "resume" and not cfg.get("checkpoint"):
    fail("checkpoint_mode", mode, "requires a non-null 'checkpoint'")
  logger = cfg.get("logger", "wandb")
  if logger not in LOGGER_KINDS:
    fail("logger", logger, f"must be one of {LOGGER_KINDS}")
  decay = cfg.get("lr_decay", "constant")
  if decay not in LR_DECAYS:
    fail("lr_decay", decay, f"must be one of {LR_DECAYS}")
  seeds = cfg.get("num_seeds")
  if seeds is not None and (not isinstance(seeds, int) or isinstance(seeds, bool) or seeds < 1):
    fail("num_seeds", seeds, "must be an int >= 1")
  num_envs = cfg.get("num_envs")
  if isinstance(num_envs, int) and not isinstance(num_envs, bool) and num_envs >= 1:
    for key in ("rollout_steps", "random_steps"):
      steps = cfg.get(key)
      if isinstance(steps, int) and not isinstance(steps, bool) and steps % num_envs:
        fail(key, steps, f"must be divisible by num_envs ({num_envs})")
  for key in ("collection_executor", "eval_executor"):
    value = cfg.get(key)
    if value is not None:
      try:
        _resolve_executor_config(value, path=key)
      except (KeyError, TypeError, ValueError) as exc:
        problems.append(str(exc))
  return problems


def _model_problems(model: ModelConfig) -> list[str]:
  problems: list[str] = []
  params = model.params
  if hasattr(params, "alpha_granularity"):
    from barl_jax.policies.rl import ALPHA_GRANULARITIES
    if params.alpha_granularity not in ALPHA_GRANULARITIES:
      problems.append(f"params.alpha_granularity={params.alpha_granularity!r}: must be one of {ALPHA_GRANULARITIES}")
  # Slot-objective compatibility (e.g. SAC's `td` critic vs NCLQL's `noise_td`).
  # Needs runtime classes for `learning_objective`/`SLOT_OBJECTIVES`; keep it
  # best-effort here so RunConfig stays importable without the registry.
  try:
    from barl_jax.policy import Policy as _Policy
    from barl_jax.registration import get_component as _get_component
    _policy_cls = _Policy.class_for(model.policy_class)
    _actor_cls = None
    try:
      _actor_cls, _ = _get_component(model.components.actor.class_name), None
    except Exception:
      _actor_cls = None
    for _slot in ("actor", "critic", "entropy"):
      _comp = getattr(model.components, _slot, None)
      if _comp is None:
        continue
      try:
        _cls = _get_component(_comp.class_name)
      except Exception as exc:
        problems.append(str(exc)); continue
      _objective = _policy_cls.slot_objective(
        _slot, _actor_cls if "actor" in getattr(_policy_cls, "SLOTS", ()) else None)
      if _objective is not None and getattr(_cls, "learning_objective", None) != _objective:
        problems.append(f"{model.policy_class}.{_slot}: {_cls.__name__} does not support {_objective!r}")
  except Exception:
    pass
  for slot, component in (("actor", model.components.actor),
                          ("critic", model.components.critic),
                          ("entropy", model.components.entropy)):
    if component is None:
      continue
    try:
      schema = get_component(component.class_name).config_type
    except KeyError as exc:
      problems.append(str(exc)); continue
    if schema is not None and not isinstance(component, schema):
      problems.append(f"components.{slot}.class={component.class_name!r}: "
                     f"materialized value is {type(component).__name__}, expected {schema.__name__}")
    mlp = getattr(component, "mlp", None)
    if mlp is not None:
      if mlp.activation not in {"silu", "relu", "gelu", "tanh", "relu6", "leakyrelu", "mish"}:
        problems.append(f"components.{slot}.mlp.activation={mlp.activation!r}: unknown activation")
      if mlp.dtype not in {"float32", "bfloat16", "float16"}:
        problems.append(f"components.{slot}.mlp.dtype={mlp.dtype!r}: unknown dtype")
    time_name = getattr(component, "time_emb", None)
    if time_name is not None:
      try:
        _selected_time_style(time_name)
      except KeyError as exc:
        problems.append(str(exc))
    from barl_jax.networks.schedule import Schedule as _Schedule
    def _is_schedule(name: str) -> bool:
      try:
        return issubclass(get_component(name), _Schedule)
      except KeyError:
        return False
    schedule = getattr(component, "noise_schedule", None)
    if schedule is not None:
      if not _is_schedule(schedule.class_name):
        problems.append(f"components.{slot}.noise_schedule.class={schedule.class_name!r}: unknown schedule")
      if schedule.steps < 1 or schedule.warmup < 0 or schedule.warmup >= schedule.steps:
        problems.append(f"components.{slot}.noise_schedule: steps must be >= 1 and warmup in [0, steps)")
    if hasattr(component, "ladder_cls") and not _is_schedule(component.ladder_cls):
      problems.append(f"components.{slot}.ladder_cls={component.ladder_cls!r}: unknown schedule")
    if hasattr(component, "ladder_kind"):
      from barl_jax.actors.sdac import LADDER_KINDS
      if component.ladder_kind not in LADDER_KINDS:
        problems.append(f"components.{slot}.ladder_kind={component.ladder_kind!r}: must be one of {LADDER_KINDS}")
    if hasattr(component, "flow_steps") and component.flow_steps < 1:
      problems.append(f"components.{slot}.flow_steps={component.flow_steps!r}: must be >= 1")
    if hasattr(component, "num_bins") and component.num_bins < 2:
      problems.append(f"components.{slot}.num_bins={component.num_bins!r}: must be >= 2")
    mode = getattr(component, "grad_clip_mode", None)
    if mode is not None and mode not in CLIP_MODES:
      problems.append(f"components.{slot}.grad_clip_mode={mode!r}: unknown clip mode")
  return problems


__all__ = ["RunConfig", "RUN_CONFIGS", "INJECTED_KEYS", "run_defaults", "run_keys",
           "validate_run", "load_config"]
