import yaml


def load_config(path: str) -> list[dict]:
  """Merges each `runs:` entry onto `defaults:` (run-specific keys win) and
  returns the resolved list of run dicts. A top-level `models:` key (see
  scripts/run_sweep.py's embedded snapshot) rides along under `cfg["models"]`
  on every resolved run, consumed by model_config.set_models_override()."""
  with open(path) as f:
    raw = yaml.safe_load(f) or {}

  defaults = raw.get("defaults", {})
  models = raw.get("models")
  runs = raw.get("runs")
  if not runs:
    raise ValueError(f"{path}: no `runs` entries -- need at least one {{model, env}} run spec")

  resolved = []
  for i, run in enumerate(runs):
    if "model" not in run or "env" not in run:
      raise ValueError(f"{path}: runs[{i}] is missing a required 'model' or 'env' key")
    entry = {**defaults, **run}
    if models:
      entry["models"] = models
    resolved.append(entry)

  return resolved
