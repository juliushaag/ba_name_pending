"""Typed training-run configuration."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field, fields, replace
from typing import TYPE_CHECKING, Any, Mapping, get_type_hints

if TYPE_CHECKING:
  from barl_jax.policies.rl import PolicyConfig

from barl_jax.config.schemas import (
  CHECKPOINT_MODES,
  LOGGER_KINDS,
  LR_DECAYS,
  ComponentsConfig,
  ModelConfig,
  component_to_dict,
  model_schema,
)
from barl_jax.config.serialization import construct, is_instance, merge, type_name
from barl_jax.executors import DirectExecutorConfig, ExecutorConfig
from barl_jax.registration import available_executors, get_component, get_executor

_NESTED = ("model_config", "collection_executor", "eval_executor")
_INT_POSITIVE = ("total_env_steps", "rollout_steps", "num_envs", "buffer_size", "batch_size",
                 "eval_every", "eval_runs", "log_every", "debug_every", "resume_keep")
_INT_NONNEGATIVE = ("random_steps", "lr_warmup_steps", "wandb_finish_timeout")
_INT_POSITIVE_OR_NULL = ("eval_video_every", "resume_every")


@dataclass(frozen=True)
class RunConfig:
  """One complete immutable training run.

  `model_config` is materialized from a model schema class and holds every model
  value. Its input spelling is `model_overrides`; the canonical output is the
  complete nested model so a serialized run is self-contained."""

  model: str
  env: str
  checkpoint: str | None = None
  checkpoint_mode: str = "weights"
  seed: int = 0
  num_seeds: int = 3
  validate: bool = False
  run_dir: str | None = None
  sweep_tag: str | None = None
  logger: str = "wandb"
  total_env_steps: int = 1_000_000
  random_steps: int = 5000
  rollout_steps: int = 1
  num_envs: int = 1
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

  @classmethod
  def defaults(cls, model: str, env: str) -> "RunConfig":
    return cls(model=model, env=env, model_config=model_schema(model)())

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
      if not isinstance(mapping[key], str):
        raise TypeError("model and env selectors must be strings")

    names = {f.name for f in fields(cls)} - {"model_config"}
    unknown = sorted(set(mapping) - names - {"model_overrides"})
    if unknown:
      raise KeyError(f"run config: unknown keys {unknown}; available keys: "
                     f"{sorted(names - set(_NESTED) | {'model_overrides'})}")

    result = cls.defaults(mapping["model"], mapping["env"])
    plain = {k: v for k, v in mapping.items()
             if k in names and k not in ("model", "env", *_NESTED)}
    result = merge(result, plain)
    for key in ("collection_executor", "eval_executor"):
      if key in mapping:
        result = replace(result, **{key: _executor_config(mapping[key], path=key)})
    overrides = mapping.get("model_overrides")
    if overrides is not None:
      if not isinstance(overrides, Mapping):
        raise TypeError("model_overrides must be a mapping")
      result = replace(result, model_config=result.model_config.overridden(overrides))
    result.check()
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
    out = {f.name: deepcopy(getattr(self, f.name)) for f in fields(self) if f.name not in _NESTED}
    # Plain dicts, not dataclass instances: `validate_resume` diffs this against
    # a JSON-round-tripped copy, and a dataclass never equals the dict json hands back.
    out["collection_executor"] = component_to_dict(self.collection_executor)
    out["eval_executor"] = component_to_dict(self.eval_executor)
    out["model_overrides"] = self.model_config.to_dict()
    return out

  def check(self) -> None:
    problems = self.problems()
    if problems:
      raise ValueError("invalid run config\n" + "\n".join(f"- {p}" for p in problems))

  def problems(self) -> list[str]:
    problems = _run_problems(self.to_dict())
    if ":" not in self.env:
      from barl_jax.env import Environment
      if self.env not in Environment.available():
        problems.append(f"env={self.env!r}: unknown environment; "
                        f"available: {Environment.available()}")
    return problems + _model_problems(self.model_config)

  def replace(self, **changes) -> "RunConfig":
    """Dataclass replacement; changing `model` rebuilds that model's defaults."""
    if "model" in changes and "model_config" not in changes:
      changes["model_config"] = model_schema(changes["model"])()
    return replace(self, **changes)


def run_defaults() -> dict[str, Any]:
  return RunConfig.defaults("SAC", "HalfCheetah").to_dict()


def validate_run(cfg: RunConfig | Mapping[str, Any], *, num_seeds: int | None = None) -> list[str]:
  """Run-level diagnostics for callers that aggregate preflight errors."""
  problems = cfg.problems() if isinstance(cfg, RunConfig) else _run_problems(cfg)
  if num_seeds is not None and not _is_int(num_seeds, 1):
    problems.append(f"num_seeds={num_seeds!r}: must be an int >= 1")
  return problems


def _executor_config(raw: Mapping[str, Any], *, path: str) -> ExecutorConfig:
  """A `class`-tagged mapping -> a fresh, typed executor config. There is no
  prior instance to merge onto, so the mapping must name its class."""
  if not isinstance(raw, Mapping):
    raise TypeError(f"{path} must be a mapping, got {type(raw).__name__}")
  value = dict(raw)
  name = value.pop("class", value.pop("class_name", None))
  if name is None:
    raise KeyError(f"{path}: must name a 'class' (available executors: {available_executors()})")
  return construct(get_executor(name).config_type, value, path=path)


def _is_int(value, minimum: int) -> bool:
  return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _run_problems(cfg: Mapping[str, Any]) -> list[str]:
  """Type and range problems in a run mapping (plain values only)."""
  if not isinstance(cfg, Mapping):
    return [f"config has type {type(cfg).__name__}, expected mapping"]
  problems: list[str] = []

  def fail(key, relation):
    problems.append(f"{key}={cfg[key]!r}: {relation}")

  for key, expected in get_type_hints(RunConfig).items():
    if key not in _NESTED and key in cfg and not is_instance(cfg[key], expected):
      problems.append(f"{key}={cfg[key]!r}: expected {type_name(expected)}")

  def ranged(keys, minimum, relation):
    for key in keys:
      value = cfg.get(key)
      if isinstance(value, int) and not isinstance(value, bool) and value < minimum:
        fail(key, relation)

  ranged(_INT_POSITIVE, 1, "must be >= 1")
  ranged(_INT_NONNEGATIVE, 0, "must be >= 0")
  ranged(_INT_POSITIVE_OR_NULL, 1, "must be >= 1 or null")
  utd = cfg.get("utd_ratio")
  if isinstance(utd, (int, float)) and not isinstance(utd, bool) and (not math.isfinite(utd) or utd <= 0):
    fail("utd_ratio", "must be a finite number > 0")
  for key, allowed, default in (("checkpoint_mode", CHECKPOINT_MODES, "weights"),
                                ("logger", LOGGER_KINDS, "wandb"),
                                ("lr_decay", LR_DECAYS, "constant")):
    if cfg.get(key, default) not in allowed:
      fail(key, f"must be one of {allowed}")
  if cfg.get("checkpoint_mode") == "resume" and not cfg.get("checkpoint"):
    fail("checkpoint_mode", "requires a non-null 'checkpoint'")
  if cfg.get("num_seeds") is not None and not _is_int(cfg["num_seeds"], 1):
    fail("num_seeds", "must be an int >= 1")
  num_envs = cfg.get("num_envs")
  if _is_int(num_envs, 1):
    for key in ("rollout_steps", "random_steps"):
      if _is_int(cfg.get(key), 0) and cfg[key] % num_envs:
        fail(key, f"must be divisible by num_envs ({num_envs})")
  for key in ("collection_executor", "eval_executor"):
    if cfg.get(key) is not None:
      try:
        _executor_config(cfg[key], path=key)
      except (KeyError, TypeError, ValueError) as exc:
        problems.append(str(exc))
  return problems


def _model_problems(model: ModelConfig) -> list[str]:
  from barl_jax.policy import Policy
  policy_cls = Policy.class_for(model.policy_class)
  problems = [f"params: {p}" for p in policy_cls.config_problems(model.params)]
  for slot, component in model.components.present().items():
    try:
      cls = get_component(component.class_name)
    except KeyError as exc:
      problems.append(str(exc))
      continue
    if cls.config_type is not None and not isinstance(component, cls.config_type):
      problems.append(f"components.{slot}.class={component.class_name!r}: materialized value is "
                      f"{type(component).__name__}, expected {cls.config_type.__name__}")
    problem = policy_cls.slot_problem(model, slot)
    if problem:
      problems.append(problem)
    problems.extend(f"components.{slot}: {p}" for p in cls.config_problems(component))
  return problems


__all__ = ["RunConfig", "run_defaults", "validate_run"]
