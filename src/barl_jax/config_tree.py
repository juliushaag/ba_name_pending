"""The config tree and its resolver -- the single definition of how a value in
config/ becomes a kwarg.

Deliberately dependency-free: this module imports `os`/`pathlib`/`yaml` and
nothing else, so scripts/submit.py can resolve and snapshot a model config
without importing barl_jax (and therefore jax/mujoco). model_config.py wraps it
with the parts that do need the component registry -- class lookup, __init__
signature checks, the no-defaults lint.

The tree, and what merges what:

    config/presets/*.yaml      named fragments, referenced as `<stem>.<key path>`
    config/components/*.yaml   per component *class* baselines (`kind:`, `extends:`)
    config/models/*.yaml       per *policy* `components:`/`params:`/`training:`/`profiles:`
    config/run/defaults.yaml   every run key
    config/run/envs.yaml       per-env stats + per-env run overrides

Precedence for a component kwarg, lowest first:

    1  the component entry's `extends:` presets, in list order (recursively)
    2  the component entry's own keys
    3  the slot's `size:` (sugar for `extends: [sizes.<kind>.<name>]`)
    4  the slot's `extends:` presets
    5  the slot's own keys
    6  the active profile's block for that slot (its size/extends/own, same order)
    7  the run's dotted overrides (`components.<slot>.<key>`)

Levels 8-9 -- a Policy's leftover `**kwargs` and the explicit kwargs it passes at
its own build_component() call site -- are applied by policy.py, above everything
here. `params:` and `training:` use the same chain minus the component layers.

Every merge is recorded as a `(source, kwargs)` pair so explain() can attribute
each resolved value to the layer that set it and list what it shadows.
"""

import os
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"

# The run-config keys a policy may set from its own `training:` block -- the ones
# that are a property of the *algorithm* rather than of the experiment.
# Everything else about a run stays in config/run/defaults.yaml, which is the
# whole point of the whitelist: a key outside it in a model's `training:` block
# is a typo or a misplaced knob, and validate() says so rather than letting it
# silently do nothing.
TRAINING_KEYS = frozenset({
  "rollout_steps",       # env steps per outer iteration
  "batch_size",
  "utd_ratio",           # grad steps per env step (off-policy)
  "epochs_per_rollout",  # on-policy passes over the buffer
  "random_steps",        # uniform-action buffer fill before training
  "lr_warmup_steps",
})

# Keys a component block carries that are not constructor kwargs.
_META_KEYS = frozenset({"class", "kind", "extends", "size"})

# The preset file whose tables `size:` resolves against, keyed by component kind.
SIZES = "sizes"


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
  if not path.exists():
    return {}
  with open(path) as f:
    return yaml.safe_load(f) or {}


def _load_dir(path: Path) -> dict:
  """Every *.yaml in a directory, keyed by file stem."""
  if not path.is_dir():
    return {}
  return {p.stem: _load_yaml(p) for p in sorted(path.glob("*.yaml"))}


def _load_dir_merged(path: Path) -> dict:
  """Every *.yaml in a directory merged into one map. A key defined in two files
  is an error rather than a silent last-file-wins -- which file an entry lives in
  is an organizational choice, never a semantic one."""
  merged, origin = {}, {}
  for name, content in _load_dir(path).items():
    for key, value in content.items():
      if key in merged:
        raise ValueError(
          f"{path.name}/: {key!r} is defined in both {origin[key]}.yaml and {name}.yaml -- "
          f"an entry must live in exactly one file")
      merged[key], origin[key] = value, name
  return merged


def _deep_update(base: dict, update: dict) -> dict:
  """dict.update, except a value that is a dict on both sides merges key-wise --
  so a preset can add one key to an inherited `norm_kwargs`/`time_kwargs` without
  restating the rest of it."""
  for key, value in update.items():
    if isinstance(value, dict) and isinstance(base.get(key), dict):
      base[key] = _deep_update(dict(base[key]), value)
    else:
      base[key] = value
  return base


def merge(layers: list[tuple[str, dict]]) -> dict:
  merged: dict = {}
  for _, kwargs in layers:
    _deep_update(merged, kwargs)
  return merged


class ConfigTree:
  """One loaded config/ tree plus the run overrides applied on top of it.

  An instance is immutable state, so two of them can coexist in one process --
  which is what lets a checkpoint reconstruct under its own frozen config
  without mutating anything global (the module-level singleton in
  model_config.py is a convenience over this, not the other way round)."""

  def __init__(self, root: str | os.PathLike = CONFIG_DIR, overrides: dict | None = None,
               snapshot: dict | None = None):
    root = Path(root)
    self.root       = root
    self.overrides  = dict(overrides or {})
    self.presets    = _load_dir(root / "presets")
    self.components = _load_dir_merged(root / "components")
    self.models     = _load_dir_merged(root / "models")
    self.run        = _load_yaml(root / "run" / "defaults.yaml")
    self.envs       = _load_yaml(root / "run" / "envs.yaml")
    # A frozen snapshot makes this tree a replay of one past run: its `resolved:`
    # map is served directly by resolve_slot/resolve_params/resolve_training, so
    # replay re-runs no layering at all and cannot drift with config/. The raw
    # blocks it also carries are deep-merged over the files purely so explain()
    # can still attribute values for that run.
    self.snapshot = snapshot
    if snapshot:
      for attr, key in (("presets", "presets"), ("components", "components"), ("models", "models")):
        setattr(self, attr, _deep_update(dict(getattr(self, attr)), snapshot.get(key) or {}))

  # -- replay ---------------------------------------------------------------

  def _replayed(self, prefix: str) -> dict:
    """The snapshot's resolved values under one prefix, with the prefix stripped."""
    resolved = (self.snapshot or {}).get("resolved") or {}
    return {k[len(prefix):]: v for k, v in resolved.items() if k.startswith(prefix)}

  @property
  def replaying(self) -> bool:
    return bool((self.snapshot or {}).get("resolved"))

  # -- presets --------------------------------------------------------------

  def preset(self, path: str) -> dict:
    """A preset by dotted path: `net.crossq`, `sizes.critic.large`."""
    node, walked = self.presets, []
    for part in path.split("."):
      walked.append(part)
      if not isinstance(node, dict) or part not in node:
        raise KeyError(f"config/presets: no preset at {path!r} "
                       f"(resolved as far as {'.'.join(walked[:-1]) or '<root>'})")
      node = node[part]
    if not isinstance(node, dict):
      raise KeyError(f"config/presets: {path!r} is a {type(node).__name__}, not a block of kwargs")
    return node

  def _extends_layers(self, block: dict, source: str, stack: tuple) -> list[tuple[str, dict]]:
    """A block's `extends:` presets expanded recursively (a preset may itself
    extend another), then the block's own kwargs -- lowest precedence first."""
    layers = []
    for ref in block.get("extends") or []:
      if ref in stack:
        raise ValueError(f"config/presets: `extends` cycle {' -> '.join((*stack, ref))}")
      layers += self._extends_layers(self.preset(ref), f"preset {ref}", (*stack, ref))
    layers.append((source, {k: v for k, v in block.items() if k not in _META_KEYS}))
    return layers

  # -- names ----------------------------------------------------------------

  def resolve_model_name(self, name: str) -> tuple[str, str | None]:
    """A `model:` string -> (policy class name, profile), via the `aliases:` map.
    Anything not listed there is already a literal Policy class name."""
    alias = (self.models.get("aliases") or {}).get(name)
    if alias:
      return alias["policy"], alias.get("profile")
    return name, None

  def policy_cfg(self, policy_name: str) -> dict:
    return self.models.get(policy_name) or {}

  def available_models(self) -> list[str]:
    return ([k for k in self.models if k != "aliases"]
            + list(self.models.get("aliases") or {}))

  def _profile_cfg(self, policy_name: str, profile: str) -> dict:
    cfg = self.policy_cfg(policy_name).get("profiles", {}).get(profile)
    if cfg is None:
      raise KeyError(f"{policy_name!r} has no profile {profile!r} in config/models/")
    return cfg

  def _override_layer(self, prefix: str) -> dict:
    return {key.rpartition(".")[2]: value
            for key, value in self.overrides.items() if key.rpartition(".")[0] == prefix}

  # -- slots ----------------------------------------------------------------

  def slot_class(self, policy_name: str, slot: str) -> str:
    if self.replaying:
      return self.snapshot["resolved"][f"components.{slot}.class"]
    slot_cfg = self.policy_cfg(policy_name).get("components", {}).get(slot, {})
    cls_name = slot_cfg.get("class")
    if cls_name is None:
      raise KeyError(
        f"config/models/: {policy_name!r} has no `class:` for its {slot!r} slot -- every "
        f"component must be named explicitly, there is no Python-level default.")
    # An override may swap the class itself (the point of a
    # `model.components.<slot>.class` sweep axis); the baseline layer then comes
    # from the class actually used.
    return self.overrides.get(f"components.{slot}.class", cls_name)

  def slot_layers(self, policy_name: str, slot: str,
                  profile: str | None) -> tuple[str, list[tuple[str, dict]]]:
    """The resolved class name plus every kwarg layer feeding that slot, ordered
    lowest-precedence first and each tagged with where it came from -- the single
    place the precedence chain is defined."""
    slot_cfg    = dict(self.policy_cfg(policy_name).get("components", {}).get(slot, {}))
    cls_name    = self.slot_class(policy_name, slot)
    component   = self.components.get(cls_name)
    if component is None:
      raise KeyError(f"config/components/: no entry for {cls_name!r} "
                     f"({policy_name}.{slot}); add one or spell its kwargs out on the slot")
    kind        = component.get("kind")
    profile_cfg = dict(self._profile_cfg(policy_name, profile).get(slot, {})) if profile else {}

    layers = self._extends_layers(component, f"components: {cls_name}", (cls_name,))

    # `size:` is sugar for `extends: [sizes.<kind>.<name>]`, resolved once at a
    # fixed rung so an explicit `latent_dim:` on the slot still shadows it.
    size = (self.overrides.get(f"components.{slot}.size")
            or profile_cfg.get("size") or slot_cfg.get("size"))
    if size is not None:
      layers.append((f"preset {SIZES}.{kind}.{size}", self.preset(f"{SIZES}.{kind}.{size}")))

    layers += self._extends_layers(slot_cfg, f"models: {policy_name}.components.{slot}", ())
    if profile:
      layers += self._extends_layers(
        profile_cfg, f"models: {policy_name}.profiles.{profile}.{slot}", ())

    override_kwargs = self._override_layer(f"components.{slot}")
    for key in ("class", "size"):
      override_kwargs.pop(key, None)  # consumed above, not constructor kwargs
    layers.append((f"override: components.{slot}", override_kwargs))

    return cls_name, layers

  def slot_size(self, policy_name: str, slot: str, profile: str | None) -> str | None:
    if self.replaying:
      return (self.snapshot["resolved"] or {}).get(f"components.{slot}.size")
    slot_cfg = self.policy_cfg(policy_name).get("components", {}).get(slot, {})
    profile_cfg = self._profile_cfg(policy_name, profile).get(slot, {}) if profile else {}
    return (self.overrides.get(f"components.{slot}.size")
            or profile_cfg.get("size") or slot_cfg.get("size"))

  def component_kind(self, cls_name: str) -> str | None:
    return (self.components.get(cls_name) or {}).get("kind")

  def size_table(self, kind: str) -> dict:
    return (self.presets.get(SIZES) or {}).get(kind) or {}

  def size_keys(self, kind: str) -> set:
    """The kwargs a kind's size ladder owns. Nothing else may set them."""
    return {key for entry in self.size_table(kind).values() for key in entry}

  def component_kwargs(self, cls_name: str, size: str | None = None) -> dict:
    """A component class's own baseline, expanded, with no policy slot involved:
    its `extends:` presets plus its own keys, optionally at a named size. This is
    what a component built outside a Policy (a test, a notebook) resolves -- so
    such a build sees exactly the layers a real run's first two rungs give it,
    rather than a hand-written dict that drifts."""
    entry = self.components.get(cls_name)
    if entry is None:
      raise KeyError(f"config/components/: no entry for {cls_name!r}")
    layers = self._extends_layers(entry, f"components: {cls_name}", (cls_name,))
    kind = entry.get("kind")
    table = self.size_table(kind)
    if table:
      if size is None:
        size = "small" if "small" in table else sorted(table)[0]
      layers.append((f"preset {SIZES}.{kind}.{size}", self.preset(f"{SIZES}.{kind}.{size}")))
    return merge(layers)

  def resolve_slot(self, policy_name: str, slot: str, profile: str | None) -> tuple[str, dict]:
    if self.replaying:
      kwargs = self._replayed(f"components.{slot}.")
      cls_name = kwargs.pop("class")
      kwargs.pop("size", None)   # metadata, not a constructor kwarg
      return cls_name, kwargs
    cls_name, layers = self.slot_layers(policy_name, slot, profile)
    return cls_name, merge(layers)

  def slots(self, policy_name: str) -> list[str]:
    if self.replaying:
      resolved = self.snapshot["resolved"]
      return sorted({k.split(".")[1] for k in resolved if k.startswith("components.")})
    return list(self.policy_cfg(policy_name).get("components", {}))

  # -- params / training ----------------------------------------------------

  def params_layers(self, policy_name: str, profile: str | None) -> list[tuple[str, dict]]:
    layers = [(f"models: {policy_name}.params", dict(self.policy_cfg(policy_name).get("params", {})))]
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
    if self.replaying:
      return self._replayed("params.")
    return merge(self.params_layers(policy_name, profile))

  def resolve_training(self, name: str) -> dict:
    """The run keys `name` sets for itself, restricted to TRAINING_KEYS. Takes a
    *model* string rather than a policy class name, because its caller only ever
    has the `model:` a run entry named. Sits below the run config's own env/run
    levels: a model states what its algorithm wants, an explicit per-env or
    per-run statement still wins."""
    if self.replaying:
      return {k: v for k, v in self._replayed("training.").items() if k in TRAINING_KEYS}
    policy_name, profile = self.resolve_model_name(name)
    return {k: v for k, v in merge(self.training_layers(policy_name, profile)).items()
            if k in TRAINING_KEYS}

  # -- environments ---------------------------------------------------------

  def env_stats(self, env_name: str) -> dict:
    """config/run/envs.yaml's `stats:` for one env (or one family base -- the
    caller walks the MRO, so a family entry and an env entry compose)."""
    return dict(((self.envs.get("environments") or {}).get(env_name) or {}).get("stats") or {})

  def env_defaults(self, env_name: str) -> dict:
    return ((self.envs.get("environments") or {}).get(env_name) or {}).get("defaults") or {}

  # -- outputs --------------------------------------------------------------

  def resolved(self, name: str, profile: str | None = None) -> dict:
    """Every effective value for `name` as a flat `params.x` / `training.x` /
    `components.<slot>.<x>` map, whichever layer set it -- the form logged to
    wandb and embedded in a run's snapshot, so runs stay comparable by their
    actual hyperparameters rather than by which layer happened to define them."""
    if self.replaying:
      return dict(self.snapshot["resolved"])
    policy_name, alias_profile = self.resolve_model_name(name)
    profile = profile if profile is not None else alias_profile
    out = {"policy": policy_name, "profile": profile}
    out.update({f"params.{k}": v for k, v in self.resolve_params(policy_name, profile).items()})
    out.update({f"training.{k}": v for k, v in self.resolve_training(name).items()})
    for slot in self.slots(policy_name):
      cls_name, kwargs = self.resolve_slot(policy_name, slot, profile)
      out[f"components.{slot}.class"] = cls_name
      size = self.slot_size(policy_name, slot, profile)
      if size is not None:
        out[f"components.{slot}.size"] = size
      out.update({f"components.{slot}.{k}": v for k, v in kwargs.items()})
    return out

  def explain(self, name: str) -> dict:
    """Every resolved value annotated with the layer it came from and what the
    layers below it said -- the readable form of the precedence chain, printed by
    scripts/print_config.py so no value is ever unattributable."""
    policy_name, profile = self.resolve_model_name(name)
    if not self.policy_cfg(policy_name):
      raise KeyError(f"config/models/: no block for {policy_name!r}")

    def annotate(layers):
      keys = {k for _, kwargs in layers for k in kwargs}
      out = {}
      for key in sorted(keys):
        # The value *after* each layer, not the layer's own literal: a dict-valued
        # kwarg (norm_kwargs, time_kwargs) deep-merges, so a layer contributing
        # `{num_freqs: 20}` resolves to the whole inherited dict with that key
        # replaced. Reporting the literal would print a config the run never used.
        hits, running = [], None
        for source, kwargs in layers:
          if key not in kwargs:
            continue
          value = kwargs[key]
          if isinstance(value, dict) and isinstance(running, dict):
            running = _deep_update(dict(running), value)
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
                          "size": self.slot_size(policy_name, slot, profile),
                          "kind": self.component_kind(cls_name)}

    return {
      "model": name, "policy": policy_name, "profile": profile,
      "params":     annotate(self.params_layers(policy_name, profile)),
      "training":   annotate(self.training_layers(policy_name, profile)),
      "components": components,
    }

  def snapshot_for(self, name: str, profile: str | None = None) -> dict:
    """What a run embeds so it replays identically regardless of later edits to
    config/: the fully resolved values under `resolved:` (the authoritative copy
    -- replay reads these and never re-runs the layering), plus the raw blocks it
    resolved against under `presets`/`components`/`models`, kept so a past run's
    provenance is still explainable."""
    policy_name, alias_profile = self.resolve_model_name(name)
    profile = profile if profile is not None else alias_profile
    models: dict = {}
    alias = (self.models.get("aliases") or {}).get(name)
    if alias:
      models["aliases"] = {name: alias}
    if policy_name in self.models:
      models[policy_name] = self.policy_cfg(policy_name)

    used_classes, used_presets = {}, {}

    def note_presets(block):
      for ref in block.get("extends") or []:
        used_presets.setdefault(ref, self.preset(ref))
        note_presets(self.preset(ref))

    for slot in self.slots(policy_name):
      cls_name = self.slot_class(policy_name, slot)
      entry = self.components.get(cls_name)
      if entry is None:
        continue
      used_classes[cls_name] = entry
      note_presets(entry)
      note_presets(self.policy_cfg(policy_name)["components"][slot])
      size, kind = self.slot_size(policy_name, slot, profile), entry.get("kind")
      if size is not None:
        used_presets.setdefault(f"{SIZES}.{kind}.{size}", self.preset(f"{SIZES}.{kind}.{size}"))

    return {
      "resolved":   self.resolved(name, profile),
      "models":     models,
      "components": used_classes,
      "presets":    _nest(used_presets),
    }


def _nest(flat: dict) -> dict:
  """`{"net.crossq": {...}}` -> `{"net": {"crossq": {...}}}`, so a snapshot's
  presets load back into the same shape ConfigTree.preset() walks."""
  out: dict = {}
  for path, value in flat.items():
    node = out
    parts = path.split(".")
    for part in parts[:-1]:
      node = node.setdefault(part, {})
    node[parts[-1]] = value
  return out
