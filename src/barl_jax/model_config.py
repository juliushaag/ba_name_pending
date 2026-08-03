"""YAML-driven configuration: component classes + kwargs (resolve_slot) and
each Policy's own top-level tunables (resolve_params), with exactly two files
behind them and no Python-level defaults -- config/components.yaml (per
component *class* baseline kwargs) under config/models.yaml (per *policy*
slots/params/profiles). See resolve_slot/resolve_component/resolve_params for
the override precedence, explain() for per-key provenance, and
set_models_override()/snapshot_for() for how a sweep run's embedded `models:`
snapshot reproduces its exact resolved config later."""

import inspect
import os

import yaml

from barl_jax.networks.registry import get_component

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "config")
_MODELS_PATH = os.path.join(_CONFIG_DIR, "models.yaml")
_COMPONENTS_PATH = os.path.join(_CONFIG_DIR, "components.yaml")

# Reserved key under which a snapshot carries the config/components.yaml
# entries it resolved against, so replaying it never depends on the current
# contents of that file.
COMPONENT_DEFAULTS_KEY = "component_defaults"

_models_cache: dict = None
_components_cache: dict = None
_models_override: dict | None = None
_components_override: dict | None = None
_overrides: dict = {}


def _load_yaml(path: str) -> dict:
  if not os.path.exists(path):
    return {}
  with open(path) as f:
    return yaml.safe_load(f) or {}


def _models() -> dict:
  global _models_cache
  if _models_cache is None:
    _models_cache = _load_yaml(_MODELS_PATH)
  return {**_models_cache, **_models_override} if _models_override else _models_cache


def _components() -> dict:
  global _components_cache
  if _components_cache is None:
    _components_cache = _load_yaml(_COMPONENTS_PATH)
  return {**_components_cache, **_components_override} if _components_override else _components_cache


def set_models_override(override: dict | None):
  """Layers `override` on top of config/models.yaml for the rest of the
  process -- lets a saved per-run YAML's embedded `models:` snapshot (see
  scripts/jax/run_sweep.py / snapshot_for()) reproduce its exact resolved
  architecture/params even if config/models.yaml has since changed. Its
  reserved `component_defaults:` key is split off the same way over
  config/components.yaml. Called once per run by
  scripts/jax/online_learning.py's main()."""
  global _models_override, _components_override
  if override is None:
    _models_override = _components_override = None
    return
  override = dict(override)
  _components_override = override.pop(COMPONENT_DEFAULTS_KEY, None)
  _models_override = override


def set_overrides(overrides: dict | None):
  """Per-run dotted-path overrides -- `{"components.critic.rho": 1.0,
  "params.entropy_scale": -2.0}` -- sitting above both YAML files and any
  profile, below only the kwargs a Policy passes itself. This is the layer a
  sweep varies (config/online_learning.yaml's `model_overrides:`, fed by
  scripts/jax/run_sweep.py's `model.*` grid axes), so nothing has to be edited
  in-place to sweep a tunable. Called once per run by
  scripts/jax/online_learning.py's main(); validate() checks the keys actually
  land somewhere."""
  global _overrides
  _overrides = dict(overrides or {})


def get_overrides() -> dict:
  return dict(_overrides)


def _override_layer(prefix: str) -> dict:
  return {key.rpartition(".")[2]: value
          for key, value in _overrides.items() if key.rpartition(".")[0] == prefix}


def resolve_model_name(name: str) -> tuple[str, str | None]:
  """Resolves a `model:` string to (policy_class_name, profile) via
  models.yaml's `aliases:` map; anything not listed there is assumed to
  already be a literal registered Policy class name."""
  alias = _models().get("aliases", {}).get(name)
  if alias:
    return alias["policy"], alias.get("profile")
  return name, None


def _profile_cfg(policy_name: str, profile: str) -> dict:
  profile_cfg = (_models().get(policy_name) or {}).get("profiles", {}).get(profile)
  if profile_cfg is None:
    raise KeyError(f"{policy_name!r} has no profile {profile!r} in {_MODELS_PATH}")
  return profile_cfg


def _slot_layers(policy_name: str, slot: str, profile: str | None) -> tuple[type, list[tuple[str, dict]]]:
  """The resolved class plus every kwarg layer feeding that slot, ordered
  lowest-precedence first and each tagged with where it came from -- the
  single place the precedence chain is defined. `class:` is required in the
  policy's own slot dict; only an override may change it, never a profile."""
  policy_cfg = _models().get(policy_name) or {}

  slot_cfg = dict(policy_cfg.get("components", {}).get(slot, {}))
  cls_name = slot_cfg.pop("class", None)
  if cls_name is None:
    raise KeyError(
      f"{policy_name!r} has no `class:` for its {slot!r} slot in {_MODELS_PATH} -- "
      f"every component must be named explicitly, there is no Python-level default."
    )
  # An override may swap the class itself (the point of a
  # `model.components.<slot>.class` sweep axis); the baseline layer below then
  # comes from the class actually used, and any now-stale kwarg the policy's
  # own slot dict carried for the old class is reported by validate().
  cls_name = _overrides.get(f"components.{slot}.class", cls_name)
  cls = get_component(cls_name)

  layers = [
    (f"components.yaml: {cls_name}", dict(_components().get(cls_name) or {})),
    (f"models.yaml: {policy_name}.components.{slot}", slot_cfg),
  ]

  if profile is not None:
    profile_slot_cfg = dict(_profile_cfg(policy_name, profile).get(slot, {}))
    profile_slot_cfg.pop("class", None)  # profiles override kwargs only, not the component class
    layers.append((f"models.yaml: {policy_name}.profiles.{profile}.{slot}", profile_slot_cfg))

  override_kwargs = _override_layer(f"components.{slot}")
  override_kwargs.pop("class", None)  # consumed above, not a constructor kwarg
  layers.append((f"override: components.{slot}", override_kwargs))

  return cls, layers


def _params_layers(policy_name: str, profile: str | None) -> list[tuple[str, dict]]:
  policy_cfg = _models().get(policy_name) or {}
  layers = [(f"models.yaml: {policy_name}.params", dict(policy_cfg.get("params", {})))]
  if profile is not None:
    layers.append((f"models.yaml: {policy_name}.profiles.{profile}.params",
                   dict(_profile_cfg(policy_name, profile).get("params", {}))))
  layers.append(("override: params", _override_layer("params")))
  return layers


def _merge(layers: list[tuple[str, dict]]) -> dict:
  merged: dict = {}
  for _, kwargs in layers:
    merged.update(kwargs)
  return merged


def resolve_slot(policy_name: str, slot: str, profile: str | None) -> tuple[type, dict]:
  """See _slot_layers for the precedence. `class:` is required -- no
  Python-level fallback -- every other key in the slot's dict is a kwarg,
  layered on top of config/components.yaml's baseline for that class."""
  cls, layers = _slot_layers(policy_name, slot, profile)
  return cls, _merge(layers)


def resolve_params(policy_name: str, profile: str | None) -> dict:
  """Like resolve_slot but for a Policy's own top-level tunables
  (models.yaml's `params:` block) rather than a named actor/critic/squash/...
  slot -- no `class:` and no components.yaml layer involved, just a flat
  kwargs dict, profile-overridable the same way."""
  return _merge(_params_layers(policy_name, profile))


def explain(name: str) -> dict:
  """Every resolved value for `name` (a policy class name or an alias)
  annotated with the layer it came from and what the layers below it said --
  the readable form of the precedence chain, printed by
  scripts/print_config.py so no default is ever unattributable."""
  policy_name, profile = resolve_model_name(name)
  policy_cfg = _models().get(policy_name)
  if policy_cfg is None:
    raise KeyError(f"{policy_name!r} has no block in {_MODELS_PATH}")

  def annotate(layers):
    keys = {k for _, kwargs in layers for k in kwargs}
    out = {}
    for key in sorted(keys):
      hits = [(source, kwargs[key]) for source, kwargs in layers if key in kwargs]
      source, value = hits[-1]
      out[key] = {"value": value, "source": source,
                  "overrides": [{"value": v, "source": s} for s, v in hits[:-1]]}
    return out

  components = {}
  for slot in policy_cfg.get("components", {}):
    cls, layers = _slot_layers(policy_name, slot, profile)
    components[slot] = {"class": cls.__name__, "kwargs": annotate(layers)}

  return {
    "model": name,
    "policy": policy_name,
    "profile": profile,
    "params": annotate(_params_layers(policy_name, profile)),
    "components": components,
  }


def available_models() -> list[str]:
  """Every `model:` string config/models.yaml understands -- its policy blocks
  plus its aliases."""
  models = _models()
  return [k for k in models if k not in ("aliases", COMPONENT_DEFAULTS_KEY)] + list(models.get("aliases", {}))


def resolved_flat(name: str) -> dict:
  """Every effective value for `name` as a flat `params.x` /
  `components.<slot>.<x>` map, whichever layer set it -- what a sweep run
  logs to wandb so runs stay comparable by their actual hyperparameters
  rather than by which layer happened to define them."""
  policy_name, profile = resolve_model_name(name)
  out = {f"params.{k}": v for k, v in resolve_params(policy_name, profile).items()}
  for slot in (_models().get(policy_name) or {}).get("components", {}):
    cls, kwargs = resolve_slot(policy_name, slot, profile)
    out[f"components.{slot}.class"] = cls.__name__
    out.update({f"components.{slot}.{k}": v for k, v in kwargs.items()})
  return out


def _accepted_kwargs(cls: type) -> tuple[set[str], bool]:
  """Named __init__ parameters across `cls`'s whole MRO (a subclass routes
  e.g. QCriticBase's `polyak_tau` through its own **kwargs), plus whether anything
  in that chain has a **kwargs catch-all at all."""
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


def validate(name: str) -> list[str]:
  """Everything about `name`'s resolved config that is wrong in a way YAML
  alone can't catch: overrides that land nowhere, kwargs no component accepts
  (every component has a **kwargs catch-all, so these are silently swallowed
  otherwise -- the expensive failure mode in a sweep), and values whose YAML
  type doesn't match the annotation. Returns problem strings; empty means
  clean. Called by scripts/jax/online_learning.py's main() before any training
  starts."""
  policy_name, profile = resolve_model_name(name)

  # Only the classes *this* model resolves to. The torch version flagged every
  # unregistered components.yaml key, which is the right global lint but the
  # wrong gate for a run: barl_jax implements a subset of the policies, so
  # every baseline belonging to an unported component (PPOCritic,
  # OUDiffusionActor, ...) would otherwise abort a perfectly valid SAC run.
  # See unknown_component_defaults() for the whole-file view.
  problems = [f"config/components.yaml: {key!r} names no registered component, "
              f"so its baseline kwargs never apply"
              for key in unknown_component_defaults(policy_name)]
  policy_cfg = _models().get(policy_name)
  if policy_cfg is None:
    return problems + [f"config/models.yaml has no block for {policy_name!r}"]

  slots = policy_cfg.get("components", {})

  for key in _overrides:
    parts = key.split(".")
    if parts[:1] == ["params"] and len(parts) == 2:
      # against the layers *below* the override, which is itself the last one
      if parts[1] not in _merge(_params_layers(policy_name, profile)[:-1]):
        problems.append(f"override {key!r} sets a param {policy_name} doesn't have")
    elif parts[:1] == ["components"] and len(parts) == 3:
      if parts[1] not in slots:
        problems.append(f"override {key!r} names no slot of {policy_name} ({', '.join(slots)})")
    else:
      problems.append(f"override {key!r} must be 'params.<name>' or 'components.<slot>.<name>'")

  for slot in slots:
    try:
      cls, kwargs = resolve_slot(policy_name, slot, profile)
    except KeyError as e:
      problems.append(str(e))
      continue

    accepted, catchall = _accepted_kwargs(cls)
    unknown = sorted(set(kwargs) - accepted)
    if unknown and catchall:
      problems += [f"{policy_name}.{slot} [{cls.__name__}]: {k!r} is accepted by no __init__ in "
                   f"{cls.__name__}'s MRO -- it is silently swallowed by **kwargs" for k in unknown]

    annotations = _annotations(cls)
    problems += [f"{policy_name}.{slot} [{cls.__name__}]: {p}"
                 for key, value in kwargs.items()
                 if key in annotations and (p := _type_problem(key, value, annotations[key]))]

  return problems


def unknown_component_defaults(policy_name: str = None) -> list[str]:
  """config/components.yaml keys that don't name a registered component --
  i.e. typos whose baseline kwargs would silently never apply. Only meaningful
  once `barl_jax` is fully imported (that's what populates the registry).

  With `policy_name`, narrowed to the classes that policy's slots actually name
  (plus any an override swaps in), which is what validate() gates a run on;
  without it, the whole-file lint."""
  from barl_jax.networks.registry import available_components
  known = set(available_components())
  candidates = dict(_components())
  if policy_name is not None:
    named = {slot_cfg.get("class")
             for slot_cfg in (_models().get(policy_name) or {}).get("components", {}).values()}
    named |= {v for k, v in _overrides.items() if k.endswith(".class")}
    candidates = {k: v for k, v in candidates.items() if k in named}
  return sorted(k for k in candidates if k not in known)


def snapshot_for(name: str) -> dict:
  """Minimal config sub-dict needed to reproduce resolving `name` later,
  regardless of subsequent edits to config/models.yaml or
  config/components.yaml: the aliases: entry (if `name` names one), the
  resolved policy's full block, and the components.yaml entry for every class
  that policy's slots name. Embedded verbatim by scripts/jax/run_sweep.py into
  each run's per-run YAML, and logged to wandb by
  scripts/jax/online_learning.py."""
  models = _models()
  out: dict = {}
  alias = models.get("aliases", {}).get(name)
  if alias:
    out["aliases"] = {name: alias}
  policy_name, _ = resolve_model_name(name)
  if policy_name in models:
    out[policy_name] = models[policy_name]

  components = _components()
  named = [slot_cfg.get("class")
           for slot_cfg in (models.get(policy_name) or {}).get("components", {}).values()]
  # An override may name a class no slot mentions (a `components.<slot>.class`
  # sweep axis), whose baseline the snapshot needs just as much.
  named += [v for k, v in _overrides.items() if k.endswith(".class")]
  used = {cls_name: components[cls_name] for cls_name in named if cls_name in components}
  if used:
    out[COMPONENT_DEFAULTS_KEY] = used

  return out
