"""Run-config declarations and loading."""

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
