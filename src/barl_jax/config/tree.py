"""The config tree and its resolver -- the single definition of how a value
becomes a kwarg.
"""

import os
from copy import deepcopy
from pathlib import Path

import yaml

from barl_jax.config.run import run_defaults
from barl_jax.config.toolkit import (
  CONFIG_DIR,
  deep_merge,
  resolve_includes,
)
from barl_jax.registration import load_builtins

TRAINING_KEYS = frozenset({
  "rollout_steps",
  "chunk_size",
  "batch_size",
  "utd_ratio",
  "epochs_per_rollout",
  "random_steps",
  "lr_warmup_steps",
})

_META_KEYS = frozenset({"class", "kind", "tier"})
TIERS = "tiers"


def _component_cls(cls_name: str):
  from barl_jax.networks.registry import get_component
  return get_component(cls_name)


def _policy_cls(policy_name: str, tree=None):
  from barl_jax.policy import Policy
  load_builtins("policies")
  class_name = tree.policy_class_name(policy_name) if tree is not None else policy_name
  return Policy.class_for(class_name)


def _hparam_defaults(cls) -> dict:
  return {name: decl.default for name, decl in getattr(cls, "_hparams", {}).items()}


def _config_defaults(cls) -> dict:
  return {name: decl.default for name, decl in getattr(cls, "_configs", {}).items()}


def _decl_layers(cls, chosen: dict = None) -> list[tuple[str, dict]]:
  configs = getattr(cls, "_configs", {})
  layers = []
  for name, decl in configs.items():
    if decl.style is not None:
      eff = (chosen or {}).get(name, decl.default)
      layers.append((f"style: {eff}", decl.style[eff]))
  if configs:
    layers.append((f"decl configs: {cls.__name__}", _config_defaults(cls)))
  layers.append((f"decl hparams: {cls.__name__}", _hparam_defaults(cls)))
  return layers


def _chosen_styles(cls, slot: str, slot_cfg: dict, profile_cfg: dict,
                   overrides: dict) -> dict:
  chosen = {}
  for name, decl in getattr(cls, "_configs", {}).items():
    if getattr(decl, "style", None):
      chosen[name] = (overrides.get(f"components.{slot}.{name}")
                      or profile_cfg.get(name) or slot_cfg.get(name) or decl.default)
  return chosen


def _load_yaml(path: Path) -> dict:
  if not path.exists():
    return {}
  with open(path) as f:
    data = yaml.safe_load(f) or {}
  return resolve_includes(data, path.parent, CONFIG_DIR)


def _load_dir(path: Path) -> dict:
  if not path.is_dir():
    return {}
  return {p.stem: _load_yaml(p) for p in sorted(path.glob("*.yaml"))}


def _load_dir_merged(path: Path) -> dict:
  merged, origin = {}, {}
  for name, content in _load_dir(path).items():
    for key, value in content.items():
      if key in merged:
        raise ValueError(
          f"{path.name}/: {key!r} is defined in both {origin[key]}.yaml and {name}.yaml -- "
          f"an entry must live in exactly one file")
      merged[key], origin[key] = value, name
  return merged


def merge(layers: list[tuple[str, dict]]) -> dict:
  merged: dict = {}
  for _, kwargs in layers:
    merged = deep_merge(merged, kwargs)
  return merged


class ConfigTree:
  def __init__(self, root: str | os.PathLike = CONFIG_DIR, overrides: dict | None = None,
               snapshot: dict | None = None):
    root = Path(root)
    self.root       = root
    self.overrides  = deepcopy(overrides or {})
    self.presets    = _load_dir(root / "presets")
    self.components = _load_dir_merged(root / "components")
    self.models     = _load_dir_merged(root / "models")
    self.run = run_defaults()
    self.run.update(_load_yaml(root / "run" / "defaults.yaml"))
    self.envs       = _load_yaml(root / "run" / "envs.yaml")
    self.snapshot = deepcopy(snapshot)
    if snapshot:
      for attr, key in (("presets", "presets"), ("components", "components"), ("models", "models")):
        setattr(self, attr, deep_merge(getattr(self, attr), snapshot.get(key) or {}))

  def _replayed(self, prefix: str) -> dict:
    resolved = (self.snapshot or {}).get("resolved") or {}
    return {k[len(prefix):]: v for k, v in resolved.items() if k.startswith(prefix)}

  @property
  def replaying(self) -> bool:
    return bool((self.snapshot or {}).get("resolved"))

  def tier_kwargs(self, kind: str, name: str) -> dict:
    from barl_jax import hparam
    return hparam.tier(kind, name)

  def resolve_model_name(self, name: str) -> tuple[str, str | None]:
    alias = (self.models.get("aliases") or {}).get(name)
    if alias:
      return alias["policy"], alias.get("profile")
    return name, None

  def policy_cfg(self, policy_name: str) -> dict:
    return self.models.get(policy_name) or {}

  def policy_class_name(self, policy_name: str) -> str:
    return self.policy_cfg(policy_name).get("policy", policy_name)

  def available_models(self) -> list[str]:
    return ([k for k in self.models if k != "aliases"]
            + list(self.models.get("aliases") or {}))

  def _profile_cfg(self, policy_name: str, profile: str) -> dict:
    profiles = self.policy_cfg(policy_name).get("profiles") or {}
    if profile not in profiles:
      raise KeyError(f"config/models/: {policy_name!r} has no profile {profile!r}\n"
                     f"  Available profiles: {sorted(profiles)}")
    return profiles[profile]

  def _override_layer(self, prefix: str) -> dict:
    return {key.rpartition(".")[2]: value
            for key, value in self.overrides.items() if key.rpartition(".")[0] == prefix}

  def slot_class(self, policy_name: str, slot: str) -> str:
    if self.replaying:
      return self.snapshot["resolved"][f"components.{slot}.class"]
    slot_cfg = self.policy_cfg(policy_name).get("components", {}).get(slot, {})
    cls_name = slot_cfg.get("class")
    if cls_name is None:
      raise KeyError(
        f"config/models/: {policy_name!r} has no `class:` for its {slot!r} slot -- every "
        f"component must be named explicitly, there is no Python-level default.")
    return self.overrides.get(f"components.{slot}.class", cls_name)

  def slot_layers(self, policy_name: str, slot: str,
                  profile: str | None) -> tuple[str, list[tuple[str, dict]]]:
    slot_cfg    = dict(self.policy_cfg(policy_name).get("components", {}).get(slot, {}))
    cls_name    = self.slot_class(policy_name, slot)
    profile_cfg = dict(self._profile_cfg(policy_name, profile).get(slot, {})) if profile else {}
    kind = self.component_kind(cls_name)

    cls = _component_cls(cls_name)
    chosen = _chosen_styles(cls, slot, slot_cfg, profile_cfg,
                            {k: v for k, v in self.overrides.items()
                             if k.startswith(f"components.{slot}.")})
    layers = _decl_layers(cls, chosen=chosen)

    tier = self.slot_tier(policy_name, slot, profile)
    if tier is not None:
      layers.append((f"tier: {kind}.{tier}", self.tier_kwargs(kind, tier)))

    if slot_cfg:
      layers.append((f"models: {policy_name}.components.{slot}",
                     {k: v for k, v in slot_cfg.items() if k not in _META_KEYS}))
    if profile and profile_cfg:
      layers.append((f"models: {policy_name}.profiles.{profile}.{slot}",
                     {k: v for k, v in profile_cfg.items() if k not in _META_KEYS}))

    override_kwargs = self._override_layer(f"components.{slot}")
    for key in ("class", "tier"):
      override_kwargs.pop(key, None)
    layers.append((f"override: components.{slot}", override_kwargs))

    return cls_name, layers

  def slot_tier(self, policy_name: str, slot: str, profile: str | None) -> str | None:
    if self.replaying:
      return (self.snapshot["resolved"] or {}).get(f"components.{slot}.tier")
    slot_cfg = self.policy_cfg(policy_name).get("components", {}).get(slot, {})
    profile_cfg = self._profile_cfg(policy_name, profile).get(slot, {}) if profile else {}
    return (self.overrides.get(f"components.{slot}.tier")
            or profile_cfg.get("tier") or slot_cfg.get("tier"))

  def component_kind(self, cls_name: str) -> str | None:
    entry = self.components.get(cls_name) or {}
    if "kind" in entry:
      return entry["kind"]
    return getattr(_component_cls(cls_name), "kind", None)

  def tier_table(self, kind: str) -> dict:
    from barl_jax import hparam
    return {name[len(f"tiers.{kind}."):]: kwargs
            for name, kwargs in hparam.BUNDLES.items()
            if name.startswith(f"tiers.{kind}.")}

  def tier_keys(self, kind: str) -> set:
    return {key for entry in self.tier_table(kind).values() for key in entry}

  def component_kwargs(self, cls_name: str, tier: str | None = None) -> dict:
    cls = _component_cls(cls_name)
    layers = _decl_layers(cls)
    kind = getattr(cls, "kind", None)
    table = self.tier_table(kind)
    if table:
      if tier is None:
        tier = "low" if "low" in table else sorted(table)[0]
      layers.append((f"tier: {kind}.{tier}", self.tier_kwargs(kind, tier)))
    return merge(layers)

  def resolve_slot(self, policy_name: str, slot: str, profile: str | None) -> tuple[str, dict]:
    cls_name, layers = self.slot_layers(policy_name, slot, profile)
    live = merge(layers)
    if not self.replaying:
      return cls_name, live
    replayed = dict(self._replayed(f"components.{slot}."))
    replayed.pop("tier", None)
    cls_name = replayed.pop("class", cls_name)
    missing = set(live) - set(replayed)
    if missing:
      raise ValueError(f"Snapshot for {policy_name}.{slot} lacks keys: {sorted(missing)}")
    return cls_name, replayed

  def slots(self, policy_name: str) -> list[str]:
    if self.replaying:
      resolved = self.snapshot["resolved"]
      return sorted({k.split(".")[1] for k in resolved if k.startswith("components.")})
    return list(self.policy_cfg(policy_name).get("components", {}))

  def params_layers(self, policy_name: str, profile: str | None) -> list[tuple[str, dict]]:
    cls = _policy_cls(policy_name, self)
    layers = []
    if getattr(cls, "_configs", {}):
      layers.append(("decl params: configs", _config_defaults(cls)))
    if getattr(cls, "_hparams", {}):
      layers.append(("decl params: hparams", _hparam_defaults(cls)))
    layers += [(f"models: {policy_name}.params", dict(self.policy_cfg(policy_name).get("params", {})))]
    if profile:
      layers.append((f"models: {policy_name}.profiles.{profile}.params",
                     dict(self._profile_cfg(policy_name, profile).get("params", {}))))
    layers.append(("override: params", self._override_layer("params")))
    return layers

  def training_layers(self, policy_name: str, profile: str | None) -> list[tuple[str, dict]]:
    layers = [(f"models: {policy_name}.training",
               dict(self.policy_cfg(policy_name).get("training", {})))]
    if profile:
      layers.append((f"models: {policy_name}.profiles.{profile}.training",
                     dict(self._profile_cfg(policy_name, profile).get("training", {}))))
    return layers

  def resolve_params(self, policy_name: str, profile: str | None) -> dict:
    live = merge(self.params_layers(policy_name, profile))
    if not self.replaying:
      return live
    replayed = self._replayed("params.")
    missing = set(live) - set(replayed)
    if missing:
      raise ValueError(f"Snapshot for {policy_name}.params lacks keys: {sorted(missing)}")
    return replayed

  def resolve_training(self, name: str, profile: str | None = None) -> dict:
    if self.replaying:
      return {k: v for k, v in self._replayed("training.").items() if k in TRAINING_KEYS}
    policy_name, alias_profile = self.resolve_model_name(name)
    profile = profile if profile is not None else alias_profile
    return {k: v for k, v in merge(self.training_layers(policy_name, profile)).items()
            if k in TRAINING_KEYS}

  def env_stats(self, env_name: str) -> dict:
    return dict(((self.envs.get("environments") or {}).get(env_name) or {}).get("stats") or {})

  def env_defaults(self, env_name: str) -> dict:
    return ((self.envs.get("environments") or {}).get(env_name) or {}).get("defaults") or {}

  def resolved(self, name: str, profile: str | None = None) -> dict:
    if self.replaying:
      return dict(self.snapshot["resolved"])
    policy_name, alias_profile = self.resolve_model_name(name)
    profile = profile if profile is not None else alias_profile
    out = {"policy": policy_name, "profile": profile}
    out.update({f"params.{k}": v for k, v in self.resolve_params(policy_name, profile).items()})
    out.update({f"training.{k}": v for k, v in self.resolve_training(policy_name, profile).items()})
    for slot in self.slots(policy_name):
      cls_name, kwargs = self.resolve_slot(policy_name, slot, profile)
      out[f"components.{slot}.class"] = cls_name
      tier = self.slot_tier(policy_name, slot, profile)
      if tier is not None:
        out[f"components.{slot}.tier"] = tier
      out.update({f"components.{slot}.{k}": v for k, v in kwargs.items()})
    return out

  def explain(self, name: str) -> dict:
    policy_name, profile = self.resolve_model_name(name)
    if not self.policy_cfg(policy_name):
      raise KeyError(f"config/models/: no block for {policy_name!r}")

    def annotate(layers):
      keys = {k for _, kwargs in layers for k in kwargs}
      out = {}
      for key in sorted(keys):
        hits, running = [], None
        for source, kwargs in layers:
          if key not in kwargs:
            continue
          value = kwargs[key]
          if isinstance(value, dict) and isinstance(running, dict):
            running = deep_merge(running, value)
          else:
            running = value
          hits.append((source, running))
        source, value = hits[-1]
        out[key] = {"value": value, "source": source,
                    "overrides": [{"value": v, "source": s} for s, v in hits[:-1]]}
      return out

    components = {}
    for slot in self.slots(policy_name):
      cls_name, layers = self.slot_layers(policy_name, slot, profile)
      components[slot] = {"class": cls_name, "kwargs": annotate(layers),
                          "tier": self.slot_tier(policy_name, slot, profile),
                          "kind": self.component_kind(cls_name)}

    return {
      "model": name, "policy": policy_name, "profile": profile,
      "params":     annotate(self.params_layers(policy_name, profile)),
      "training":   annotate(self.training_layers(policy_name, profile)),
      "components": components,
    }

  def snapshot_for(self, name: str, profile: str | None = None) -> dict:
    policy_name, alias_profile = self.resolve_model_name(name)
    profile = profile if profile is not None else alias_profile
    models: dict = {}
    alias = (self.models.get("aliases") or {}).get(name)
    if alias:
      models["aliases"] = {name: alias}
    if policy_name in self.models:
      models[policy_name] = self.policy_cfg(policy_name)

    used_presets = {}
    for slot in self.slots(policy_name):
      cls_name = self.slot_class(policy_name, slot)
      tier, kind = self.slot_tier(policy_name, slot, profile), self.component_kind(cls_name)
      if tier is not None:
        used_presets[f"{TIERS}.{kind}.{tier}"] = self.tier_kwargs(kind, tier)

    return {
      "resolved":   self.resolved(name, profile),
      "models":     models,
      "presets":    _nest(used_presets),
    }


def _nest(flat: dict) -> dict:
  out: dict = {}
  for path, value in flat.items():
    node = out
    parts = path.split(".")
    for part in parts[:-1]:
      node = node.setdefault(part, {})
    node[parts[-1]] = value
  return out


def unknown_component_defaults(policy_name: str = None, *, config) -> list[str]:
  from barl_jax.networks.registry import available_components
  known = set(available_components())
  t = config
  candidates = dict(t.components)
  if policy_name is not None:
    named = {t.slot_class(policy_name, slot) for slot in t.slots(policy_name)}
    candidates = {k: v for k, v in candidates.items() if k in named}
  return sorted(k for k in candidates if k not in known)


def structure_problems(config) -> list[str]:
  t = config
  if t.replaying:
    return []
  problems = []

  from barl_jax import hparam
  kinds = sorted({name.split(".")[1] for name in hparam.BUNDLES if name.startswith(f"{TIERS}.")})
  for kind in kinds:
    table = t.tier_table(kind)
    key_sets = {name: set(entry) for name, entry in table.items()}
    if len(set(map(frozenset, key_sets.values()))) > 1:
      shape = ", ".join(f"{n}={sorted(k)}" for n, k in key_sets.items())
      problems.append(f"{TIERS}.{kind}: entries declare different keys "
                      f"({shape}) -- every tier of a kind must set the same ones, or a tier "
                      f"comparison silently varies which keys are set")

  return problems


def validate(name: str, config) -> list[str]:
  from barl_jax.policy import Policy
  load_builtins("policies")

  t = config
  policy_name, profile = t.resolve_model_name(name)

  problems = [f"config/components/: {key!r} names no registered component, "
              f"so its baseline kwargs never apply"
              for key in unknown_component_defaults(policy_name, config=t)]
  problems += structure_problems(t)

  if not t.policy_cfg(policy_name):
    return problems + [f"config/models/ has no block for {policy_name!r}"]

  slots = t.slots(policy_name)
  class_name = t.policy_class_name(policy_name)
  try:
    policy_cls = Policy.class_for(class_name)
  except KeyError:
    policy_cls = None
  if policy_cls is None:
    problems.append(f"config/models/: {policy_name}'s policy: names {class_name!r}, "
                    f"which is not a registered Policy ({', '.join(sorted(Policy._classes))})")
  else:
    declared, expected = set(slots), set(policy_cls.SLOTS)
    problems += [f"{policy_name}: YAML declares slot {s!r} that {class_name} does not "
                 f"build (it builds {sorted(expected)})" for s in sorted(declared - expected)]
    problems += [f"{policy_name}: {class_name} builds slot {s!r} but the YAML declares "
                 f"none ({sorted(declared)})" for s in sorted(expected - declared)]

  for source, block in t.training_layers(policy_name, profile):
    problems += [f"{source}: {key!r} is not a training key a model may set "
                 f"({', '.join(sorted(TRAINING_KEYS))}) -- run-level knobs outside that set "
                 f"belong in config/run/defaults.yaml"
                 for key in sorted(set(block) - TRAINING_KEYS)]

  for key in t.overrides:
    parts = key.split(".")
    if parts[:1] == ["params"] and len(parts) == 2:
      if parts[1] not in merge(t.params_layers(policy_name, profile)[:-1]):
        problems.append(f"override {key!r} sets a param {policy_name} doesn't have")
    elif parts[:1] == ["components"] and len(parts) == 3:
      if parts[1] not in slots:
        problems.append(f"override {key!r} names no slot of {policy_name} ({', '.join(slots)})")
      elif parts[2] == "tier":
        kind = t.component_kind(t.slot_class(policy_name, parts[1]))
        if t.overrides[key] not in t.tier_table(kind):
          problems.append(f"override {key!r}={t.overrides[key]!r} is not a {kind} tier "
                          f"({', '.join(sorted(t.tier_table(kind)))})")
    else:
      problems.append(f"override {key!r} must be 'params.<name>' or 'components.<slot>.<name>'")

  for slot in slots:
    try:
      cls_name, _kwargs = t.resolve_slot(policy_name, slot, profile)
      cls = _component_cls(cls_name)
    except KeyError as e:
      problems.append(str(e))
      continue

    objective = getattr(policy_cls, "SLOT_OBJECTIVES", {}).get(slot)
    if objective is not None and getattr(cls, "learning_objective", None) != objective:
      problems.append(f"{policy_name}.{slot}: {cls.__name__} does not support {objective!r}")

    kind = t.component_kind(t.slot_class(policy_name, slot))
    if t.tier_table(kind) and t.slot_tier(policy_name, slot, profile) is None:
      problems.append(f"{policy_name}.{slot} [{cls.__name__}]: no `tier:` -- every {kind} slot "
                      f"must name one of {', '.join(sorted(t.tier_table(kind)))}")

  return problems
