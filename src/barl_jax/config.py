import yaml


def load_config(path: str) -> list[dict]:
  """Merges each `runs:` entry onto `defaults:`, the model's own `training:` block
  and `env_defaults:` for that entry's env, and returns the resolved list of
  run dicts. Precedence, lowest first:

      defaults: < <the model's `training:` block> < env_defaults[<env>] < the run entry

  The model layer is what lets a run knob that is really a property of the
  *algorithm* (batch_size, utd_ratio, rollout_steps -- see
  model_config.TRAINING_KEYS) live next to that policy in config/models.yaml instead
  of being restated per run. It sits above `defaults:` so a model that declares
  one gets it, and below env_defaults/the run entry so an explicit statement in
  this file -- including any scripts/jax/run_sweep.py `run.` axis, which writes
  into the run entry -- still wins.

  `model_overrides:` is the one key merged key-wise across the levels instead
  of replaced wholesale, since a per-env and a per-run override normally mean
  to stack. A top-level `models:` key (see scripts/jax/run_sweep.py's embedded
  snapshot) rides along under `cfg["models"]` on every resolved run, consumed
  by model_config.set_models_override()."""
  with open(path) as f:
    raw = yaml.safe_load(f) or {}

  defaults = raw.get("defaults", {})
  env_defaults = raw.get("env_defaults") or {}
  models = raw.get("models")
  runs = raw.get("runs")
  if not runs:
    raise ValueError(f"{path}: no `runs` entries -- need at least one {{model, env}} run spec")

  # Imported here, not at module scope: this resolves against config/models.yaml
  # (plus any embedded `models:` snapshot), which drags in the component
  # registry -- and load_config is also read by tooling that has no business
  # importing the whole package.
  from barl_jax import model_config
  # A sweep's per-run YAML carries its own frozen copy of models.yaml; the
  # model layer has to come from *that*, not from whatever the file says now.
  # setup() sets the same override again per run, so this is not a second
  # source of truth, just an earlier application of the same one.
  model_config.set_models_override(models)

  resolved = []
  for i, run in enumerate(runs):
    if "model" not in run or "env" not in run:
      raise ValueError(f"{path}: runs[{i}] is missing a required 'model' or 'env' key")

    levels = [defaults, model_config.resolve_training(run["model"]),
              env_defaults.get(run["env"], {}), run]

    entry = {k: v for level in levels for k, v in level.items()}
    overrides = {k: v for level in levels for k, v in (level.get("model_overrides") or {}).items()}
    if overrides:
      entry["model_overrides"] = overrides
    if models:
      entry["models"] = models
    resolved.append(entry)

  return resolved
