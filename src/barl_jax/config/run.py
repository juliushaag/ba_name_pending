"""Run-config declarations and loading."""

import math

from barl_jax.config.toolkit import CONFIG_DIR, load_yaml_with_includes
from barl_jax.hparam import Config

RUN_CONFIGS: dict[str, Config] = {
  "checkpoint": Config(None),
  "checkpoint_mode": Config("weights"),
  "seed": Config(0),
  "num_seeds": Config(3),
  "validate": Config(False),
  "run_dir": Config(None),
  "sweep_tag": Config(None),
  "logger": Config("wandb"),
  "total_env_steps": Config(1_000_000),
  "random_steps": Config(5000),
  "rollout_steps": Config(1),
  "num_envs": Config(1),
  "chunk_size": Config(1),
  "buffer_size": Config(1_000_000),
  "batch_size": Config(256),
  "utd_ratio": Config(2.0),
  "epochs_per_rollout": Config(32),
  "lr_warmup_steps": Config(0),
  "lr_decay": Config("constant"),
  "eval_every": Config(25000),
  "eval_runs": Config(10),
  "eval_videos": Config(1),
  "eval_video_every": Config(200000),
  "log_every": Config(250),
  "debug_every": Config(1000),
  "wandb_finish_timeout": Config(300),
  "compile_cache_dir": Config("~/.cache/barl_jax/jax"),
}

INJECTED_KEYS = frozenset({
  "model", "env",
  "model_overrides",
  "snapshot",
})


def run_defaults() -> dict:
  return {name: cfg.default for name, cfg in RUN_CONFIGS.items()}


# Keys read as a modulo denominator or a `// ` divisor somewhere in trainer.py
# -- zero there is not "off", it is a ZeroDivisionError or a silent 0-length
# iteration, so these must be >= 1 rather than merely >= 0.
_POSITIVE_INT_KEYS = ("total_env_steps", "rollout_steps", "num_envs", "chunk_size",
                      "buffer_size", "batch_size", "eval_every", "eval_runs",
                      "log_every", "debug_every")
# Explicitly allowed to be zero -- see each key's own comment in RUN_CONFIGS'
# consumers (e.g. random_steps: 0 skips the warmup fill; lr_warmup_steps: 0 is
# no warmup).
_NONNEGATIVE_INT_KEYS = ("random_steps", "lr_warmup_steps", "wandb_finish_timeout")
# `None` means "never" for these (see trainer.py's `_train`); a stated value
# still goes through the same >= 1 rule as the always-on cadence keys above.
_POSITIVE_OR_NONE_INT_KEYS = ("eval_video_every",)
_NONNEGATIVE_OR_NONE_INT_KEYS = ("eval_videos",)

LOGGER_KINDS = ("wandb", "local", "none")


def validate_run(cfg: dict, *, num_seeds: int = None) -> list[str]:
  """Preflight checks over a resolved run config, independent of the model it
  names (barl_jax.config.tree.validate covers that half) -- the boundary both
  YAML/CLI-loaded configs and every direct `trainer.setup()` caller share, so
  a bad run config fails before `_setup` allocates an environment, a policy,
  or a JAX device buffer. Each problem names the key, the value, and the
  relation it violates."""
  problems = []

  def fail(key, value, relation):
    problems.append(f"{key}={value!r}: {relation}")

  for key, decl in RUN_CONFIGS.items():
    if key not in cfg:
      continue
    value = cfg[key]
    # A default of None (checkpoint, run_dir, sweep_tag, ...) infers
    # `type(None)`, i.e. "no declared type" -- these are validated by their
    # own specific rules below instead, not by Config.coerce().
    if decl.default is None:
      continue
    # bool is an int subclass -- Config.coerce()'s isinstance check would
    # accept True/False for an int-typed key silently.
    if decl.type in (int, float) and isinstance(value, bool):
      fail(key, value, f"must be a {decl.type.__name__}, not bool")
      continue
    try:
      decl.coerce(value)
    except TypeError as e:
      fail(key, value, str(e))

  for key in _POSITIVE_INT_KEYS:
    if key in cfg and isinstance(cfg[key], int) and not isinstance(cfg[key], bool) and cfg[key] < 1:
      fail(key, cfg[key], "must be >= 1")
  for key in _NONNEGATIVE_INT_KEYS:
    if key in cfg and isinstance(cfg[key], int) and not isinstance(cfg[key], bool) and cfg[key] < 0:
      fail(key, cfg[key], "must be >= 0")
  for key in _POSITIVE_OR_NONE_INT_KEYS:
    v = cfg.get(key)
    if v is not None and isinstance(v, int) and not isinstance(v, bool) and v < 1:
      fail(key, v, "must be >= 1 or null")
  for key in _NONNEGATIVE_OR_NONE_INT_KEYS:
    v = cfg.get(key)
    if v is not None and isinstance(v, int) and not isinstance(v, bool) and v < 0:
      fail(key, v, "must be >= 0 or null")

  utd = cfg.get("utd_ratio")
  if utd is not None and isinstance(utd, (int, float)) and not isinstance(utd, bool):
    if not math.isfinite(utd) or utd <= 0:
      fail("utd_ratio", utd, "must be a finite number > 0")

  mode = cfg.get("checkpoint_mode", "weights")
  if mode not in ("weights", "resume"):
    fail("checkpoint_mode", mode, "must be 'weights' or 'resume'")
  elif mode == "resume" and not cfg.get("checkpoint"):
    fail("checkpoint_mode", mode, "requires a non-null 'checkpoint'")

  decay = cfg.get("lr_decay")
  if decay is not None:
    from barl_jax.optimization import OptimizerMixin
    if decay not in OptimizerMixin.DECAYS:
      fail("lr_decay", decay, f"must be one of {sorted(OptimizerMixin.DECAYS)}")

  logger_kind = cfg.get("logger", "wandb")
  if logger_kind not in LOGGER_KINDS:
    fail("logger", logger_kind, f"must be one of {LOGGER_KINDS}")

  seeds = num_seeds if num_seeds is not None else cfg.get("num_seeds")
  if seeds is not None and (not isinstance(seeds, int) or isinstance(seeds, bool) or seeds < 1):
    fail("num_seeds", seeds, "must be an int >= 1")

  # trainer._train computes `rollout_steps // num_envs` as the env-step count
  # collected per iteration; below num_envs that floors to 0, so the loop
  # collects nothing and env_steps never advances (an infinite loop, not a
  # small one).
  rollout_steps, num_envs = cfg.get("rollout_steps"), cfg.get("num_envs")
  if (isinstance(rollout_steps, int) and isinstance(num_envs, int)
      and not isinstance(rollout_steps, bool) and not isinstance(num_envs, bool)
      and num_envs >= 1 and rollout_steps < num_envs):
    fail("rollout_steps", rollout_steps, f"must be >= num_envs ({num_envs}), "
        "or every iteration collects zero env steps and training never advances")

  return problems


def run_keys(root=CONFIG_DIR) -> set:
  from barl_jax.config.tree import ConfigTree
  return set(ConfigTree(root).run)


def _env_defaults(env_name: str) -> dict:
  from barl_jax.env import Environment
  try:
    cls = Environment.class_for(env_name)
  except KeyError:
    cls = None
  return cls.merged_defaults() if cls is not None else {}


def load_config(path: str, root=CONFIG_DIR) -> list[dict]:
  from barl_jax.config.tree import ConfigTree
  raw = load_yaml_with_includes(path, config_root=root)

  snapshot = raw.get("snapshot")
  runs = raw.get("runs")
  if not runs:
    raise ValueError(f"{path}: no `runs` entries -- need at least one {{model, env}} run spec")

  tree = ConfigTree(root, snapshot=snapshot)
  declared = set(tree.run)

  resolved = []
  for i, run in enumerate(runs):
    if "model" not in run or "env" not in run:
      raise ValueError(f"{path}: runs[{i}] is missing a required 'model' or 'env' key")

    levels = [tree.run, raw.get("defaults") or {}, tree.resolve_training(run["model"]),
              _env_defaults(run["env"]), run]

    entry = {k: v for level in levels for k, v in level.items()}
    overrides = {k: v for level in levels for k, v in (level.get("model_overrides") or {}).items()}
    if overrides:
      entry["model_overrides"] = overrides
    if snapshot:
      entry["snapshot"] = snapshot

    unknown = sorted(set(entry) - declared - INJECTED_KEYS)
    if unknown:
      raise ValueError(
        f"{path}: runs[{i}] resolves keys declared nowhere in config/run/defaults.yaml: "
        f"{', '.join(unknown)} -- add them there (a run key with no declaration would be a "
        f"silent no-op, and would never appear in the run's wandb config)")

    resolved.append(entry)

  return resolved
