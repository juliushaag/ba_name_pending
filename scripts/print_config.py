import argparse
import os

import yaml

os.environ.setdefault("MUJOCO_GL", "egl")

import barl_jax  # noqa: F401  -- populates the policy/component registries
import barl_jax.model_config as model_config


def print_model(name: str) -> None:
  cfg = model_config.explain(name)

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
    if slot_cfg.get("size"):
      title += f"  size: {slot_cfg['size']}"
    print_block(title, slot_cfg["kwargs"])


if __name__ == "__main__":
  args = argparse.ArgumentParser(
    description="Print a model's fully resolved config -- presets < component baseline "
                "< slot size < slot kwargs < profile < run overrides -- annotating every "
                "value with the layer it came from and what it shadows.")
  args.add_argument("models", nargs="*",
                    help="Policy class names and/or config/models/ aliases (default: all of them)")
  args.add_argument("--set", action="append", metavar="PATH=VALUE",
                    help="Preview under a dotted override, exactly as a run's model_overrides: "
                         "or a scripts/run_sweep.py `model.` axis would apply it, e.g. "
                         "--set components.critic.size=large. Repeatable.")
  args.add_argument("--json", action="store_true",
                    help="Print the flat resolved config as JSON instead -- exactly what a run "
                         "snapshots and logs to wandb.")
  kwargs = args.parse_args()

  overrides = {}
  for assignment in kwargs.set or []:
    path, sep, value = assignment.partition("=")
    if not sep:
      raise SystemExit(f"--set {assignment!r} is not of the form path=value")
    overrides[path.strip()] = yaml.safe_load(value)
  model_config.set_overrides(overrides)

  names = kwargs.models or model_config.available_models()

  if kwargs.json:
    import json
    print(json.dumps({name: model_config.resolved_flat(name) for name in names},
                     indent=2, sort_keys=True, default=str))
    raise SystemExit(0)

  problems = []
  for name in names:
    print_model(name)
    problems += [f"{name}: {p}" for p in model_config.validate(name)]

  if problems:
    print("\nPROBLEMS (these abort a run -- see model_config.validate)")
    for problem in problems:
      print(f"  {problem}")
