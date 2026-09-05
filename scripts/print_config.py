import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import barl_jax  # noqa: F401  -- populates the policy/component registries
import barl_jax.config.tree as model_config
from barl_jax.config.toolkit import parse_set_overrides


def print_model(name: str, config) -> None:
  cfg = config.explain(name)

  header = f"{cfg['model']}  ->  {cfg['policy']}"
  if cfg["profile"]:
    header += f", profile {cfg['profile']}"
  print(f"\n{header}\n{'=' * len(header)}")

  def print_block(title: str, entries: dict) -> None:
    if not entries:
      return
    print(f"\n  {title}")
    width = max(len(k) for k in entries)
    for key, info in entries.items():
      shadowed = ", ".join(f"{o['value']!r} from {o['source']}" for o in reversed(info["overrides"]))
      print(f"    {key:<{width}}  {info['value']!r:<12}  {info['source']}"
            + (f"   (shadows {shadowed})" if shadowed else ""))

  print_block("training", cfg["training"])
  print_block("params", cfg["params"])
  for slot, slot_cfg in cfg["components"].items():
    title = f"components.{slot}  [{slot_cfg['class']}]"
    if slot_cfg.get("tier"):
      title += f"  tier: {slot_cfg['tier']}"
    print_block(title, slot_cfg["kwargs"])


if __name__ == "__main__":
  args = argparse.ArgumentParser(
    description="Print a model's fully resolved config -- presets < component baseline "
                "< slot tier < slot kwargs < profile < run overrides -- annotating every "
                "value with the layer it came from and what it shadows.")
  args.add_argument("models", nargs="*",
                    help="Policy class names and/or config/models/ aliases (default: all of them)")
  args.add_argument("--set", action="append", metavar="PATH=VALUE",
                    help="Preview under a dotted override, exactly as a run's model_overrides: "
                         "or a scripts/submit.py `model.` axis would apply it, e.g. "
                         "--set components.critic.tier=large. Repeatable.")
  args.add_argument("--json", action="store_true",
                    help="Print the flat resolved config as JSON instead -- exactly what a run "
                         "snapshots and logs to wandb.")
  kwargs = args.parse_args()

  overrides = parse_set_overrides(kwargs.set)
  config = model_config.ConfigTree(overrides=overrides)

  names = kwargs.models or config.available_models()

  if kwargs.json:
    import json
    print(json.dumps({name: config.resolved(name) for name in names},
                     indent=2, sort_keys=True, default=str))
    raise SystemExit(0)

  problems = []
  for name in names:
    print_model(name, config)
    problems += [f"{name}: {p}" for p in model_config.validate(name, config)]

  if problems:
    print("\nPROBLEMS (these abort a run -- see model_config.validate)")
    for problem in problems:
      print(f"  {problem}")
