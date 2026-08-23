"""Registry-aware facade over config_tree.ConfigTree.

config_tree.py owns the tree and the whole precedence chain and imports nothing
but yaml, so scripts/run_sweep.py can use it without dragging in jax. This module
adds the parts that need the component registry -- turning a `class:` string into
a class, checking resolved kwargs against that class's __init__ signature, and
the lint that keeps tunables out of Python -- plus a process-wide default tree
for the callers that want one rather than passing an instance around.

See config_tree.ConfigTree for the layering; see validate() for what is checked
before a run is allowed to start."""

import inspect

from barl_jax.config_tree import (CONFIG_DIR, SIZES, TRAINING_KEYS, ConfigTree,  # noqa: F401
                                  merge)
from barl_jax.networks.registry import get_component

# __init__ parameters that are plumbing rather than tunables: injected by the
# framework, not resolved from config. Everything else a component takes must be
# keyword-only and defaultless, which is what no_default_problems() enforces.
PLUMBING_PARAMS = frozenset({"self", "env_stats", "rngs", "q_function", "support",
                             "cls", "args", "kwargs", "in_dim", "out_dim",
                             "input_dim", "output_dim", "num_features", "n"})

# Params whose default is not a hidden constant but "take it from EnvStats" --
# a critic's support follows the env unless a kwarg pins it. They are exempt from
# the no-defaults rule *because* the value they end up with is reported by the
# component's derived() and logged under `model/derived.*`, so it is still
# visible in the run. Adding a name here without a matching derived() entry
# would reintroduce exactly what this module exists to prevent.
ENV_BACKED_PARAMS = frozenset({"v_min", "v_max"})

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
# signature introspection
# ---------------------------------------------------------------------------

def _accepted_kwargs(cls: type) -> tuple[set[str], bool]:
  """Named __init__ parameters across `cls`'s whole MRO (a subclass routes e.g.
  QCriticBase's `gamma` through its own **kwargs), plus whether anything in that
  chain has a **kwargs catch-all at all."""
  names, catchall = set(), False
  for klass in cls.__mro__:
    init = klass.__dict__.get("__init__")
    if init is None:
      continue
    for param in inspect.signature(init).parameters.values():
      if param.name == "self":
        continue
      if param.kind is param.VAR_KEYWORD:
        catchall = True
      elif param.kind is not param.VAR_POSITIONAL:
        names.add(param.name)
  return names, catchall


def _annotations(cls: type) -> dict:
  out = {}
  for klass in reversed(cls.__mro__):
    init = klass.__dict__.get("__init__")
    if init is None:
      continue
    for param in inspect.signature(init).parameters.values():
      if isinstance(param.annotation, type):
        out[param.name] = param.annotation
  return out


def _type_problem(key: str, value, annotation: type) -> str | None:
  """Only flags the unambiguous YAML-typing traps (a bare `3e-4` or a stray
  `true,` parsing as a str, a float where an int is required); anything whose
  annotation isn't a plain bool/int/float/str is left alone."""
  if annotation not in (bool, int, float, str):
    return None
  if annotation is float and isinstance(value, int) and not isinstance(value, bool):
    return None
  if annotation is not bool and isinstance(value, bool):
    return f"{key}: {value!r} is a bool, expected {annotation.__name__}"
  if not isinstance(value, annotation):
    return f"{key}: {value!r} is a {type(value).__name__}, expected {annotation.__name__}"
  return None


def no_default_problems(cls: type) -> list[str]:
  """Every __init__ parameter of `cls` that is a tunable yet carries a Python
  default or is positional -- i.e. a value that could differ from what config/
  says and would never appear in the run's wandb config. This is the check that
  makes "no defaults in the code" an enforced invariant rather than a convention;
  PLUMBING_PARAMS is the whitelist of framework-injected arguments."""
  problems = []
  for klass in cls.__mro__:
    init = klass.__dict__.get("__init__")
    if init is None:
      continue
    for param in inspect.signature(init).parameters.values():
      if (param.name in PLUMBING_PARAMS or param.name in ENV_BACKED_PARAMS
          or param.kind in (param.VAR_KEYWORD, param.VAR_POSITIONAL)):
        continue
      where = f"{klass.__name__}.__init__"
      if param.default is not param.empty:
        problems.append(f"{where}: {param.name!r} has a Python default "
                        f"({param.default!r}) -- move it into config/")
      elif param.kind is not param.KEYWORD_ONLY:
        problems.append(f"{where}: {param.name!r} is positional -- every tunable must be "
                        f"keyword-only so a missing config value is a TypeError")
  return problems


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
  """Whole-tree checks that don't depend on which model is being run: sizes own
  their keys exclusively, every `extends:` resolves, every component declares a
  `kind:`. These are what keep `size: large` meaning one thing everywhere."""
  t = tree()
  if t.replaying:
    # A replay serves one past run's resolved values; the tree it came from was
    # checked when the snapshot was made, and the files here are not what ran.
    return []
  problems = []

  for kind, table in (t.presets.get(SIZES) or {}).items():
    key_sets = {name: set(entry) for name, entry in table.items()}
    if len(set(map(frozenset, key_sets.values()))) > 1:
      shape = ", ".join(f"{n}={sorted(k)}" for n, k in key_sets.items())
      problems.append(f"config/presets/sizes.yaml: {kind!r} entries declare different keys "
                      f"({shape}) -- every size of a kind must set the same ones, or a size "
                      f"comparison silently varies which keys are set")

  for cls_name, entry in t.components.items():
    if not isinstance(entry, dict):
      continue
    if "kind" not in entry:
      problems.append(f"config/components/: {cls_name!r} declares no `kind:`")
      continue
    owned = t.size_keys(entry["kind"])
    for key in sorted(owned & set(entry)):
      problems.append(f"config/components/: {cls_name!r} sets {key!r}, which the "
                      f"{entry['kind']!r} size ladder owns -- state a `size:` on the slot instead")
    for ref in entry.get("extends") or []:
      try:
        t.preset(ref)
      except (KeyError, ValueError) as e:
        problems.append(f"config/components/: {cls_name}: {e}")

  return problems


def validate(name: str) -> list[str]:
  """Everything about `name`'s resolved config that is wrong in a way YAML alone
  can't catch: unregistered classes, a slot missing its `size:`, overrides that
  land nowhere, kwargs no component accepts (every component has a **kwargs
  catch-all, so these are silently swallowed otherwise -- the expensive failure
  mode in a sweep), tunables still defaulting in Python, and values whose YAML
  type doesn't match the annotation. Returns problem strings; empty means clean.
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

    accepted, catchall = _accepted_kwargs(cls)
    unknown = sorted(set(kwargs) - accepted)
    if unknown and catchall:
      problems += [f"{policy_name}.{slot} [{cls.__name__}]: {k!r} is accepted by no __init__ in "
                   f"{cls.__name__}'s MRO -- it is silently swallowed by **kwargs" for k in unknown]

    problems += [f"{policy_name}.{slot} [{cls.__name__}]: {p}" for p in no_default_problems(cls)]

    annotations = _annotations(cls)
    problems += [f"{policy_name}.{slot} [{cls.__name__}]: {p}"
                 for key, value in kwargs.items()
                 if key in annotations and (p := _type_problem(key, value, annotations[key]))]

  return problems
