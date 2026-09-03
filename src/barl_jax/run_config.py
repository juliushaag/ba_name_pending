"""Run-level tunables, declared as `HParam`s in code -- the defaults that used
to live in config/run/defaults.yaml. Every resolved run dict is now complete by
construction: the code declaration provides a value for every key, and a YAML
layer (`defaults:`, a model's `training:`, an env's `defaults:`, a run entry)
overrides it. Because the defaults are HParams, the resolved config the run
logs can attribute every value to a layer and nothing is silent.

Pure module (imports only barl_jax.hparam) so the launcher can read it without
pulling jax -- the same discipline config_tree.py holds itself to. The six
`TRAINING_KEYS` a model may set from its own `training:` block are unchanged
(see config_tree.py) -- they are a subset of the keys below."""

from barl_jax.hparam import HParam

RUN_HPARAMS: dict[str, HParam] = {
  # --- what to run ---------------------------------------------------------
  "checkpoint": HParam(None, descr="path to resume from, or null to train from scratch"),
  "seed":       HParam(0, descr="base seed; a multi-seed run uses seed..seed+num_seeds-1"),
  "num_seeds":  HParam(3, descr="seeds vmapped into one process (one CUDA context, one "
                               "compiled step)"),
  "validate":   HParam(False, descr="smoke-test mode: one rollout chunk + one train_step"),
  "run_dir":    HParam(None, descr="a sweep's per-run directory; null = the global "
                                   "checkpoints/<model>/<env>/<timestamp>/ tree"),
  "sweep_tag":  HParam(None, descr="wandb tag grouping one sweep's runs"),

  # --- budget --------------------------------------------------------------
  "total_env_steps": HParam(1_000_000, descr="stop once this many env steps have been collected"),
  "random_steps":    HParam(5000, descr="random-action buffer fill before training starts"),

  # --- rollout -------------------------------------------------------------
  "rollout_steps": HParam(1, descr="env steps collected per outer iteration (across all "
                                   "num_envs). Keep equal to num_envs."),
  "num_envs":      HParam(1, descr="parallel rollout envs"),
  "buffer_size":   HParam(1_000_000, descr="max transitions kept, per seed"),

  # --- optimization ---------------------------------------------------------
  "batch_size":        HParam(256),
  "utd_ratio":         HParam(2.0, descr="off-policy: grad steps per outer iteration = "
                                         "round(rollout_steps * utd_ratio)"),
  "epochs_per_rollout": HParam(32, descr="on-policy only -- currently unread"),
  "lr_warmup_steps":   HParam(0, descr="grad steps of linear 0 -> base_lr ramp"),
  "lr_decay":          HParam("constant", descr="shape of the post-warmup decay (Policy.DECAYS)"),

  # --- evaluation ------------------------------------------------------------
  "eval_every":      HParam(25000, descr="env steps between evals"),
  "eval_runs":       HParam(10, descr="episodes per eval, run in parallel"),
  "eval_videos":     HParam(1, descr="of those runs, how many render frames; null = all"),
  "eval_video_every": HParam(200000, descr="env steps between rendered evals; null = never"),

  # --- logging ---------------------------------------------------------------
  "log_every":   HParam(250, descr="grad steps between primary-tier logs"),
  "debug_every": HParam(1000, descr="grad steps between debug-tier logs"),
  "wandb_finish_timeout": HParam(300, descr="seconds run.finish() may spend uploading"),

  # --- compilation ------------------------------------------------------------
  "compile_cache_dir": HParam("~/.cache/barl_jax/jax",
                              descr="where jax keeps its persistent compilation cache; null disables"),
}


def run_defaults() -> dict:
  """Every run key's declaration default, as a plain name -> value dict."""
  return {name: hp.default for name, hp in RUN_HPARAMS.items()}