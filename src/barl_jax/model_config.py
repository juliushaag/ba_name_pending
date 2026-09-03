"""Registry-aware facade over config_tree.ConfigTree.

config_tree.py owns the tree and the whole precedence chain and imports nothing
but yaml, so scripts/submit.py can use it without dragging in jax. This module
adds the parts that need the component registry -- turning a `class:` string into
a class, checking resolved kwargs against that class's __init__ signature, and
the lint that keeps tunables out of Python -- plus a process-wide default tree
for the callers that want one rather than passing an instance around.

See config_tree.ConfigTree for the layering; see validate() for what is checked
before a run is allowed to start."""

from barl_jax.config_tree import (CONFIG_DIR, SIZES, TRAINING_KEYS, ConfigTree,  # noqa: F401
                                  merge)
from barl_jax.networks.registry import get_component
from barl_jax.policy import Policy

_tree: ConfigTree | None = None
_overrides: dict = {}
_snapshot: dict | None = None


def tree() -> ConfigTree:
  """The process-wide tree, rebuilt whenever the overrides or the frozen
  snapshot change."""
  global _tree
  if _tree is None:
    _tree = ConfigTree(overrides=_overrides, snapshot=_snapshot)
  return _tree


def _invalidate():
  global _tree
  _tree = None


def set_snapshot(snapshot: dict | None):
  """Freezes the tree to a run's embedded snapshot (see
  ConfigTree.snapshot_for), so a saved run or checkpoint resolves exactly what it
  resolved originally even if config/ has since changed. Called once per run by
  trainer.setup()."""
  global _snapshot
  _snapshot = snapshot
  _invalidate()


def set_overrides(overrides: dict | None):
  """Per-run dotted-path overrides -- `{"components.critic.size": "large",
  "params.entropy_scale": -2.0}` -- sitting above the whole tree and below only
  the kwargs a Policy passes itself. This is the layer a sweep varies, so nothing
  has to be edited in place to sweep a tunable."""
  global _overrides
  _overrides = dict(overrides or {})
  _invalidate()


def get_overrides() -> dict:
  return dict(_overrides)


def resolve_model_name(name: str) -> tuple[str, str | None]:
  return tree().resolve_model_name(name)


def resolve_training(name: str) -> dict:
  return tree().resolve_training(name)


def resolve_slot(policy_name: str, slot: str, profile: str | None) -> tuple[type, dict]:
  """The resolved component *class* plus its merged kwargs. `class:` is
  required -- there is no Python-level default."""
  cls_name, kwargs = tree().resolve_slot(policy_name, slot, profile)
  return get_component(cls_name), kwargs


def resolve_params(policy_name: str, profile: str | None) -> dict:
  return tree().resolve_params(policy_name, profile)


def explain(name: str) -> dict:
  return tree().explain(name)


def available_models() -> list[str]:
  return tree().available_models()


def resolved_flat(name: str, profile: str | None = None) -> dict:
  return tree().resolved(name, profile)


def snapshot_for(name: str, profile: str | None = None) -> dict:
  return tree().snapshot_for(name, profile)


def get_snapshot() -> dict | None:
  return _snapshot


def env_stats(env_name: str) -> dict:
  return tree().env_stats(env_name)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def unknown_component_defaults(policy_name: str = None) -> list[str]:
  """config/components/ entries that don't name a registered component -- typos
  whose baseline kwargs would silently never apply. Only meaningful once
  `barl_jax` is fully imported (that's what populates the registry).

  With `policy_name`, narrowed to the classes that policy's slots actually name
  (plus any an override swaps in), which is what validate() gates a run on;
  without it, the whole-tree lint."""
  from barl_jax.networks.registry import available_components
  known = set(available_components())
  candidates = dict(tree().components)
  if policy_name is not None:
    named = {tree().slot_class(policy_name, slot) for slot in tree().slots(policy_name)}
    candidates = {k: v for k, v in candidates.items() if k in named}
  return sorted(k for k in candidates if k not in known)


def structure_problems() -> list[str]:
  """Whole-tree checks that don't depend on which model is being run: a size
  ladder's rungs set the same keys (so `size: large` means one thing
  everywhere). These are what keep a `size:` comparison honest."""
  t = tree()
  if t.replaying:
    # A replay serves one past run's resolved values; the tree it came from was
    # checked when the snapshot was made, and the code here is not what ran.
    return []
  problems = []

  from barl_jax import hparam
  kinds = sorted({name.split(".")[1] for name in hparam.BUNDLES if name.startswith("sizes.")})
  for kind in kinds:
    table = t.size_table(kind)
    key_sets = {name: set(entry) for name, entry in table.items()}
    if len(set(map(frozenset, key_sets.values()))) > 1:
      shape = ", ".join(f"{n}={sorted(k)}" for n, k in key_sets.items())
      problems.append(f"sizes.{kind}: entries declare different keys "
                      f"({shape}) -- every size of a kind must set the same ones, or a size "
                      f"comparison silently varies which keys are set")

  return problems


def validate(name: str) -> list[str]:
  """Everything about `name`'s resolved config that is wrong in a way YAML alone
  can't catch: unregistered classes, a slot missing its `size:`, and overrides
  that land nowhere. A component missing a required config key or getting a
  value of the wrong type is a `ConfigMixin.get()` error raised at construction
  instead -- not caught here. Returns problem strings; empty means clean.
  Called by trainer.setup() before any training starts."""
  t = tree()
  policy_name, profile = t.resolve_model_name(name)

  # Only the classes *this* model resolves to: barl_jax implements a subset of
  # the policies, so a baseline belonging to an unported component would
  # otherwise abort a perfectly valid SAC run. structure_problems() is the
  # whole-tree view.
  problems = [f"config/components/: {key!r} names no registered component, "
              f"so its baseline kwargs never apply"
              for key in unknown_component_defaults(policy_name)]
  problems += structure_problems()

  if not t.policy_cfg(policy_name):
    return problems + [f"config/models/ has no block for {policy_name!r}"]

  slots = t.slots(policy_name)

  # A model whose `policy:` names a class that isn't in the registry would pass
  # every YAML check, then fail at `trainer.setup`'s build_component on the
  # compute node seconds after the GPU is allocated -- catch it here instead.
  class_name = t.policy_class_name(policy_name)
  policy_cls = Policy._classes.get(class_name)
  if policy_cls is None:
    problems.append(f"config/models/: {policy_name}'s policy: names {class_name!r}, "
                    f"which is not a registered Policy ({', '.join(sorted(Policy._classes))})")
  else:
    # A slot the policy builds but the YAML omits already raises at
    # build_component; an *extra* slot would resolve silently into the snapshot
    # and the wandb config with no component behind it -- check both directions.
    declared, expected = set(slots), set(policy_cls.SLOTS)
    problems += [f"{policy_name}: YAML declares slot {s!r} that {class_name} does not "
                 f"build (it builds {sorted(expected)})" for s in sorted(declared - expected)]
    problems += [f"{policy_name}: {class_name} builds slot {s!r} but the YAML declares "
                 f"none ({sorted(declared)})" for s in sorted(expected - declared)]

  # A `training:` key outside TRAINING_KEYS lands in no run config and changes
  # nothing -- the same silent no-op a typo'd sweep axis would be.
  for source, block in t.training_layers(policy_name, profile):
    problems += [f"{source}: {key!r} is not a training key a model may set "
                 f"({', '.join(sorted(TRAINING_KEYS))}) -- run-level knobs outside that set "
                 f"belong in config/run/defaults.yaml"
                 for key in sorted(set(block) - TRAINING_KEYS)]

  for key in _overrides:
    parts = key.split(".")
    if parts[:1] == ["params"] and len(parts) == 2:
      # against the layers *below* the override, which is itself the last one
      if parts[1] not in merge(t.params_layers(policy_name, profile)[:-1]):
        problems.append(f"override {key!r} sets a param {policy_name} doesn't have")
    elif parts[:1] == ["components"] and len(parts) == 3:
      if parts[1] not in slots:
        problems.append(f"override {key!r} names no slot of {policy_name} ({', '.join(slots)})")
      elif parts[2] == "size":
        kind = t.component_kind(t.slot_class(policy_name, parts[1]))
        if _overrides[key] not in t.size_table(kind):
          problems.append(f"override {key!r}={_overrides[key]!r} is not a {kind} size "
                          f"({', '.join(sorted(t.size_table(kind)))})")
    else:
      problems.append(f"override {key!r} must be 'params.<name>' or 'components.<slot>.<name>'")

  for slot in slots:
    try:
      cls, kwargs = resolve_slot(policy_name, slot, profile)
    except KeyError as e:
      problems.append(str(e))
      continue

    kind = t.component_kind(t.slot_class(policy_name, slot))
    if t.size_table(kind) and t.slot_size(policy_name, slot, profile) is None:
      problems.append(f"{policy_name}.{slot} [{cls.__name__}]: no `size:` -- every {kind} slot "
                      f"must name one of {', '.join(sorted(t.size_table(kind)))}")

  return problems
