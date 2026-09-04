"""Run-level keys, declared as `Config`s in code -- the defaults that used
to live in config/run/defaults.yaml. Every resolved run dict is now complete by
construction: the code declaration provides a value for every key, and a YAML
layer (`defaults:`, a model's `training:`, an env's `defaults:`, a run entry)
overrides it. Because the defaults are declared in code, the resolved config the run
logs can attribute every value to a layer and nothing is silent.

Pure module (imports only barl_jax.hparam) so the launcher can read it without
pulling jax -- the same discipline config_tree.py holds itself to. The six
`TRAINING_KEYS` a model may set from its own `training:` block are unchanged
(see config_tree.py) -- they are a subset of the keys below."""

from barl_jax.hparam import Config

RUN_CONFIGS: dict[str, Config] = {
  # --- what to run ---------------------------------------------------------
  "checkpoint": Config(None),
  "seed":       Config(0),
  "num_seeds":  Config(3),
  "validate":   Config(False),
  "run_dir":    Config(None),
  "sweep_tag":  Config(None),

  # --- budget --------------------------------------------------------------
  "total_env_steps": Config(1_000_000),
  "random_steps":    Config(5000),

  # --- rollout -------------------------------------------------------------
  "rollout_steps": Config(1),
  "num_envs":      Config(1),
  "chunk_size":    Config(1),
  "buffer_size":   Config(1_000_000),

  # --- optimization ---------------------------------------------------------
  "batch_size":        Config(256),
  "utd_ratio":         Config(2.0),
  "epochs_per_rollout": Config(32),
  "lr_warmup_steps":   Config(0),
  "lr_decay":          Config("constant"),

  # --- evaluation ------------------------------------------------------------
  "eval_every":      Config(25000),
  "eval_runs":       Config(10),
  "eval_videos":     Config(1),
  "eval_video_every": Config(200000),

  # --- logging ---------------------------------------------------------------
  "log_every":   Config(250),
  "debug_every": Config(1000),
  "wandb_finish_timeout": Config(300),

  # --- compilation ------------------------------------------------------------
  "compile_cache_dir": Config("~/.cache/barl_jax/jax"),
}

# Back-compat for call sites/tests that still import the old name. The values
# are ordinary Config declarations now, not searchable HParams.
RUN_HPARAMS = RUN_CONFIGS


def run_defaults() -> dict:
  """Every run key's declaration default, as a plain name -> value dict."""
  return {name: cfg.default for name, cfg in RUN_CONFIGS.items()}
