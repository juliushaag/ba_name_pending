import yaml


def load_config(path: str) -> list[dict]:
  """Merges each `runs:` entry onto `defaults:` (run-specific keys win) and
  returns the resolved list of run dicts."""
  with open(path) as f:
    raw = yaml.safe_load(f) or {}

  defaults = raw.get("defaults", {})
  runs = raw.get("runs")
  if not runs:
    raise ValueError(f"{path}: no `runs` entries -- need at least one {{model, env}} run spec")

  resolved = []
  for i, run in enumerate(runs):
    if "model" not in run or "env" not in run:
      raise ValueError(f"{path}: runs[{i}] is missing a required 'model' or 'env' key")
    resolved.append({**defaults, **run})

  return resolved
