"""Run-config loading: composes config/run/defaults.yaml, config/run/envs.yaml,
each model's own `training:` block and an experiment file's `runs:` list into one
resolved dict per run.

Every run key is declared in config/run/defaults.yaml -- RUN_KEYS is read from
that file, not written out here, and a key in a resolved run that it doesn't
declare is an error. That is what lets trainer.py index `cfg[...]` directly
instead of carrying a `cfg.get(key, default)` fallback the wandb config would
never show."""

import yaml

from barl_jax.config_tree import CONFIG_DIR, ConfigTree

# Keys a run carries that are set by the machinery rather than declared as
# tunables: what to run, where to put it, and the frozen config it runs under.
INJECTED_KEYS = frozenset({
  "model", "env",          # the {algo, env} combination itself
  "model_overrides",       # dotted-path overrides layered over the model config
  "snapshot",              # this run's frozen copy of the config tree
})


def run_keys(root=CONFIG_DIR) -> set:
  """The declared run keys -- the contents of config/run/defaults.yaml."""
  return set(ConfigTree(root).run)


def load_config(path: str, root=CONFIG_DIR) -> list[dict]:
  """Resolves an experiment file into one dict per run. Precedence, lowest first:

      config/run/defaults.yaml
        < this file's `defaults:`
        < the model's own `training:` block (config/models/<policy>.yaml)
        < config/run/envs.yaml's `defaults:` for that run's env
        < the `runs:` entry itself

  The model layer is what lets a run knob that is really a property of the
  *algorithm* (batch_size, utd_ratio, rollout_steps -- config_tree.TRAINING_KEYS)
  live next to that policy instead of being restated per run. It sits above this
  file's `defaults:` so a model that declares one gets it, and below the env/run
  levels so an explicit statement -- including any scripts/submit.py `run.`
  axis, which writes into the run entry -- still wins.

  `model_overrides:` is the one key merged key-wise across the levels instead of
  replaced wholesale, since a per-env and a per-run override normally mean to
  stack. A top-level `snapshot:` key (see scripts/grid.py) rides along on
  every resolved run and is applied by model_config.set_snapshot()."""
  with open(path) as f:
    raw = yaml.safe_load(f) or {}

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
              tree.env_defaults(run["env"]), run]

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
