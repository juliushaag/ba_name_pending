import yaml


def load_config(path: str) -> list[dict]:
  """Merges each `runs:` entry onto `defaults:` and onto `env_defaults:` for
  that entry's env (run-specific keys win, env_defaults beat plain defaults)
  and returns the resolved list of run dicts. `model_overrides:` is the one
  key merged key-wise across those three levels instead of replaced wholesale,
  since a per-env and a per-run override normally mean to stack. A top-level
  `models:` key (see scripts/run_sweep.py's embedded snapshot) rides along
  under `cfg["models"]` on every resolved run, consumed by
  model_config.set_models_override()."""
  with open(path) as f:
    raw = yaml.safe_load(f) or {}

  defaults = raw.get("defaults", {})
  env_defaults = raw.get("env_defaults") or {}
  models = raw.get("models")
  runs = raw.get("runs")
  if not runs:
    raise ValueError(f"{path}: no `runs` entries -- need at least one {{model, env}} run spec")

  resolved = []
  for i, run in enumerate(runs):
    if "model" not in run or "env" not in run:
      raise ValueError(f"{path}: runs[{i}] is missing a required 'model' or 'env' key")

    levels = [defaults, env_defaults.get(run["env"], {}), run]

    entry = {k: v for level in levels for k, v in level.items()}
    overrides = {k: v for level in levels for k, v in (level.get("model_overrides") or {}).items()}
    if overrides:
      entry["model_overrides"] = overrides
    if models:
      entry["models"] = models
    resolved.append(entry)

  return resolved
