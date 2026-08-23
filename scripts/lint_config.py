"""Whole-tree config lint -- the gate that keeps `config/` the only place a
tunable lives.

model_config.validate() runs the per-model subset of these before every training
run; this script is the whole-tree view, for CI and for editing config/ by hand.
It fails on anything that would make a run's wandb config an incomplete account
of what actually ran:

  * a registered component with a Python default or a positional tunable
  * a `config/components/` entry naming no registered component, or missing `kind:`
  * a size ladder whose entries declare different keys, or a component/slot
    setting a key a ladder owns
  * a slot of a sized kind with no `size:`
  * an `extends:` path resolving to nothing, or a cycle
  * a preset nothing references
  * a run key read by the code but declared in no config/run/defaults.yaml

Exit code is the number of problems, so it composes in a shell.

    uv run python scripts/lint_config.py
"""

import argparse
import os
import re
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import barl_jax  # noqa: F401  -- populates the policy/component registries
import barl_jax.model_config as model_config
from barl_jax.config import INJECTED_KEYS, run_keys
from barl_jax.networks.registry import available_components, get_component

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Where run keys are read from a config dict. Anything matching `cfg["x"]` in
# these must be declared, or it is a key nothing can ever set.
RUN_KEY_SOURCES = ("src/barl_jax/trainer.py", "scripts/online_learning.py")
RUN_KEY_PATTERN = re.compile(r"""cfg\[["'](\w+)["']\]""")


def component_problems() -> list[str]:
  """Every registered component: no Python defaults, and an entry to resolve."""
  problems = []
  tree = model_config.tree()
  for name in sorted(available_components()):
    problems += [f"{name}: {p}" for p in model_config.no_default_problems(get_component(name))]
    if name not in tree.components:
      problems.append(f"{name} is registered but has no config/components/ entry, so every "
                      f"kwarg it needs must be spelled out on each slot that names it")
  return problems


def slot_problems() -> list[str]:
  """Every slot of every model: a size where the kind has a ladder."""
  problems = []
  tree = model_config.tree()
  for name in sorted(tree.available_models()):
    policy_name, profile = tree.resolve_model_name(name)
    for slot in tree.slots(policy_name):
      try:
        cls_name = tree.slot_class(policy_name, slot)
      except KeyError as e:
        problems.append(str(e))
        continue
      kind = tree.component_kind(cls_name)
      if tree.size_table(kind) and tree.slot_size(policy_name, slot, profile) is None:
        problems.append(f"{policy_name}.{slot} [{cls_name}]: no `size:` -- every {kind} slot "
                        f"must name one of {', '.join(sorted(tree.size_table(kind)))}")
  return problems


def unused_presets() -> list[str]:
  """A preset nothing extends is either a typo at the reference site or dead
  weight; either way it is not what the tree says it is."""
  tree = model_config.tree()
  referenced = set()

  def walk(block):
    for ref in (block or {}).get("extends") or []:
      referenced.add(ref)
      try:
        walk(tree.preset(ref))
      except (KeyError, ValueError):
        pass

  for entry in tree.components.values():
    walk(entry if isinstance(entry, dict) else {})
  for policy_name in tree.available_models():
    for slot, slot_cfg in (tree.policy_cfg(policy_name).get("components") or {}).items():
      walk(slot_cfg)
      size = slot_cfg.get("size")
      if size is not None:
        referenced.add(f"sizes.{tree.component_kind(slot_cfg.get('class'))}.{size}")

  problems = []
  for stem, block in tree.presets.items():
    if stem == model_config.SIZES:
      continue   # a ladder's unused rungs are the point: they are what you switch to
    for key in block:
      if f"{stem}.{key}" not in referenced:
        problems.append(f"config/presets/{stem}.yaml: {key!r} is referenced by no "
                        f"`extends:` anywhere")
  return problems


def undeclared_run_keys() -> list[str]:
  declared = run_keys() | INJECTED_KEYS
  problems = []
  for rel in RUN_KEY_SOURCES:
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
      continue
    with open(path) as f:
      for key in sorted(set(RUN_KEY_PATTERN.findall(f.read()))):
        if key not in declared:
          problems.append(f"{rel}: reads cfg[{key!r}], declared in no "
                          f"config/run/defaults.yaml -- nothing could ever set it")
  return problems


CHECKS = {
  "components":  component_problems,
  "structure":   model_config.structure_problems,
  "slots":       slot_problems,
  "presets":     unused_presets,
  "run_keys":    undeclared_run_keys,
}


if __name__ == "__main__":
  args = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  args.add_argument("--only", choices=sorted(CHECKS), action="append",
                    help="Run just this check. Repeatable.")
  kwargs = args.parse_args()

  total = 0
  for name, check in CHECKS.items():
    if kwargs.only and name not in kwargs.only:
      continue
    problems = check()
    total += len(problems)
    status = "ok" if not problems else f"{len(problems)} problem(s)"
    print(f"{name:<12} {status}")
    for problem in problems:
      print(f"  - {problem}")

  print(f"\n{'clean' if not total else f'{total} problem(s)'}")
  sys.exit(min(total, 100))
