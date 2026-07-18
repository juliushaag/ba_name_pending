import yaml


def load_config(path: str) -> list[dict]:
  """
  Loads a YAML config shaped like:

      defaults:
        num_iterations: 100
        ...
      runs:
        - model: FlowQLPolicy
          env: HalfCheetah
        - model: SACPolicy
          env: HalfCheetah
          num_iterations: 50   # overrides defaults for this run only

  Each entry in `runs` is merged onto `defaults` (run-specific keys win),
  so a single config can list several algorithm/env combinations to run in
  series while only writing down what differs between them. Returns the
  resolved list of run dicts -- callers just iterate and run each in turn.
  """
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
