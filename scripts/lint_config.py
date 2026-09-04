"""Whole-tree config lint -- the gate that keeps defaults coupled to code and
keeps the composed config an honest account of a run.

config_tree.validate() runs the per-model subset of these before every training
run; this script is the whole-tree view, for CI and for editing config/ by hand.
It fails on anything that would make a run's wandb config incomplete:

  * a registered component without a `kind` (no tier ladder to resolve against)
  * a tier ladder whose entries declare different keys, or a component declaring
    a key a ladder owns
  * a slot of a tiered kind with no `tier:`, or a stated `tier:` that isn't a rung
  * a model whose `policy:` names an unregistered class, or whose `components:`
    disagree with that class's declared slots
  * a run key read by the code but declared in no code declaration (barl_jax.config)

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
import barl_jax.config.tree as model_config
from barl_jax.config import INJECTED_KEYS, run_keys
from barl_jax.networks.registry import available_components, get_component
from barl_jax.policy import Policy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Where run keys are read from a config dict. Anything matching `cfg["x"]` in
# these must be declared, or it is a key nothing can ever set.
RUN_KEY_SOURCES = ("src/barl_jax/trainer.py", "scripts/online_learning.py")
RUN_KEY_PATTERN = re.compile(r"""cfg\[["'](\w+)["']\]""")


def component_problems() -> list[str]:
  """Every registered component: its class declares what makes it resolvable.
  With config/components/*.yaml gone, the class IS the baseline -- `kind`
  selects its tier ladder (a tiered component must name one), and a `tier:` a
  slot states must resolve to an actual rung."""
  problems = []
  tree = model_config.tree()
  for name in sorted(available_components()):
    cls = get_component(name)
    if getattr(cls, "kind", None) is None:
      problems.append(f"{name} declares no `kind` -- a slot's `tier:` has no ladder to resolve against")
  return problems


def policy_classes_problems() -> list[str]:
  """Every model block's `policy:` names a registered Policy class. A model
  naming a missing class passes every YAML-only check and then fails at
  `trainer.setup` on the compute node -- check the registry here, where the
  error is still one message."""
  tree = model_config.tree()
  problems = []
  for name in sorted(tree.available_models()):
    policy_name, _ = tree.resolve_model_name(name)
    class_name = tree.policy_class_name(policy_name)
    if class_name not in Policy._classes:
      problems.append(f"config/models/: {name}'s policy: names {class_name!r}, "
                      f"which is not a registered Policy ({', '.join(sorted(Policy._classes))})")
  return problems


def slot_problems() -> list[str]:
  """Every slot of every model: a tier where the kind has a ladder, and a
  `components:` block that disagrees with its policy class's SLOTS."""
  problems = []
  tree = model_config.tree()
  for name in sorted(tree.available_models()):
    policy_name, profile = tree.resolve_model_name(name)
    slots = tree.slots(policy_name)

    # The policy builds SLOTS; the model YAML declares `components:`. A slot
    # declared but not built resolves silently into the snapshot and the wandb
    # config with nothing behind it; a built slot not declared fails later at
    # build_component -- both are worth saying here, where the model files sit.
    class_name = tree.policy_class_name(policy_name)
    policy_cls = Policy._classes.get(class_name)
    if policy_cls is not None:
      declared, expected = set(slots), set(policy_cls.SLOTS)
      problems += [f"{policy_name}: YAML declares slot {s!r} that {class_name} does not "
                   f"build (it builds {sorted(expected)})" for s in sorted(declared - expected)]
      problems += [f"{policy_name}: {class_name} builds slot {s!r} but the YAML declares "
                   f"none ({sorted(declared)})" for s in sorted(expected - declared)]

    for slot in slots:
      try:
        cls_name = tree.slot_class(policy_name, slot)
      except KeyError as e:
        problems.append(str(e))
        continue
      kind = tree.component_kind(cls_name)
      table = tree.tier_table(kind)
      if not table:
        continue
      tier = tree.slot_tier(policy_name, slot, profile)
      if tier is None:
        problems.append(f"{policy_name}.{slot} [{cls_name}]: no `tier:` -- every {kind} slot "
                        f"must name one of {', '.join(sorted(table))}")
      elif tier not in table:
        problems.append(f"{policy_name}.{slot} [{cls_name}]: `tier: {tier}` is not a {kind} "
                        f"capacity ({', '.join(sorted(table))})")
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
  "policies":    policy_classes_problems,
  "slots":       slot_problems,
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
