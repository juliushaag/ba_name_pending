"""The config tree and its resolver -- the single definition of how a value
becomes a kwarg.

Defaults now live **on the classes that use them** (see barl_jax/hparam.py:
`HParam` for searchable algorithmic knobs, `Config` for structural defaults),
composed by inheritance from style mixins (networks/mlp.py, networks/flow.py,
networks/optim.py, networks/critic.py). There is no named-bundle or `extends:`
indirection and no config/presets or config/components YAML -- a component's
baseline *is* its class. This module is a resolver over layering only. It
imports only `os`/`pathlib`/`yaml` and `barl_jax.configs` at import time; the
class registry is pulled lazily once a component class / its declared defaults
are needed.

`barl_jax.configs` is the substrate this module is built on: `$()` file
inclusion, deep merge, and the typed `ConfigMixin.get()` accessor.

What merges what:

    a component class             its inherited `Config` (structural) and
                                  `HParam` (searchable) defaults, down the MRO
    a policy class                its own `HParam` params
    barl_jax.hparam               the `sizes.<kind>.<name>` capacity ladders a
                                  slot's `size:` resolves against
    config/models/*.yaml          per-*policy* `components:`/`params:`/`training:`/`profiles:`
    an env class                  `stats`/`defaults` (return range, per-env overrides)
    barl_jax.run_config           every run key's default (`RUN_HPARAMS`)

Precedence for a component kwarg, lowest first:

    1  the class's inherited `Config` structural defaults
    2  the class's inherited `HParam` algorithmic defaults
    3  the slot's `size:` (a capacity-ladder rung, `sizes.<kind>.<name>`)
    4  the slot's own keys in config/models.yaml
    5  the active profile's block for that slot
    6  the run's dotted overrides (`components.<slot>.<key>`)

Explicit runtime kwargs passed by a Policy's build_component() call site are
applied by policy.py above everything here. `params:` uses the same chain minus
the structural rungs, with the policy class's own HParams as its lowest rung.

Every merge is recorded as a `(source, kwargs)` pair so explain() can attribute
each resolved value to the layer that set it and list what it shadows.
"""

import importlib.util
import os
from pathlib import Path

import yaml

# Not `from barl_jax.configs import ...`: scripts/grid.py loads *this* module
# by path precisely so a login node never pays for `barl_jax/__init__.py`'s
# jax/mujoco imports (see its own comment), and an absolute `barl_jax.*`
# import here would trigger exactly that package `__init__` regardless of how
# this module itself was reached. Loading configs.py the same by-path way
# grid.py loads this module keeps that guarantee intact.
_CONFIGS_PATH = Path(__file__).resolve().with_name("configs.py")
_spec = importlib.util.spec_from_file_location("barl_jax_configs", _CONFIGS_PATH)
_configs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_configs)
CONFIG_DIR       = _configs.CONFIG_DIR
ConfigMixin      = _configs.ConfigMixin
_format_value    = _configs._format_value
deep_merge       = _configs.deep_merge
resolve_includes = _configs.resolve_includes

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
_META_KEYS = frozenset({"class", "kind", "size"})

# The size-ladder namespace (barl_jax.hparam) a `size:` resolves against.
SIZES = "sizes"


def _component_cls(cls_name: str):
  """The registered class behind a `class:` name -- what owns the HParam/Config
  declarations, `kind`, and the completeness of the composed
  config."""
  # Lazy: pulling the registry imports barl_jax (jax, mujoco), which this module
  # itself never does at import time; every reader of the resolved tree already
  # runs inside or right after a barl_jax import by now.
  from barl_jax.networks.registry import get_component
  return get_component(cls_name)


def _policy_cls(policy_name: str, tree=None):
  """The registered Policy class behind a model name, for its declared params."""
  from barl_jax.policy import Policy
  class_name = tree.policy_class_name(policy_name) if tree is not None else policy_name
  if class_name not in Policy._classes:
    raise KeyError(f"config/models/: {policy_name!r} names unregistered policy {class_name!r}")
  return Policy._classes[class_name]


def _hparam_defaults(cls) -> dict:
  """The class's declared HParam defaults as the base kwarg layer."""
  return {name: decl.default for name, decl in getattr(cls, "_hparams", {}).items()}


def _config_defaults(cls) -> dict:
  """The class's declared structural Config defaults as a base kwarg layer."""
  return {name: decl.default for name, decl in getattr(cls, "_configs", {}).items()}


def _decl_layers(cls, chosen: dict = None) -> list[tuple[str, dict]]:
  """A component class's own declaration, as the base layers of its slot:
  each style selector's *whole surface*, then the structural `Config` defaults
  (selectors included, so the resolved config records the choice), then the
  searchable `HParam` defaults -- lowest precedence first, merged down the MRO.

  `chosen` maps a style selector to its *effective* name (the class default
  unless a slot/profile/override moved it); expanding the surface from the
  effective value is what makes a `--set components.<slot>.net_style=silu`
  change the whole net."""

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
  """The effective style-selector names for a class's slot: override >
  profile > slot YAML > class default. Style surfaces are expanded from these."""
  chosen = {}
  for name, decl in getattr(cls, "_configs", {}).items():
    if getattr(decl, "style", None):
      chosen[name] = (overrides.get(f"components.{slot}.{name}")
                      or profile_cfg.get(name) or slot_cfg.get(name) or decl.default)
  return chosen


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
  """A YAML file, with any `$()` include it (or anything it includes) contains
  resolved -- see barl_jax.configs.resolve_includes. No file under config/
  currently uses `$()`, so this is a no-op for the tree as it stands; it makes
  the syntax available as an alternative to `extends:` for splicing a whole
  file at a point in the tree, rather than merging a named preset."""
  if not path.exists():
    return {}
  with open(path) as f:
    data = yaml.safe_load(f) or {}
  return resolve_includes(data, path.parent, CONFIG_DIR)


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


def merge(layers: list[tuple[str, dict]]) -> dict:
  """`deep_merge` (barl_jax.configs) folded over every layer, lowest
  precedence first -- so a preset can add one key to an inherited
  `norm_kwargs`/`time_kwargs` without restating the rest of it."""
  merged: dict = {}
  for _, kwargs in layers:
    merged = deep_merge(merged, kwargs)
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
    # Run-key defaults are code-declared (barl_jax.run_config.RUN_HPARAMS);
    # config/run/defaults.yaml is an override layer that predates the move and
    # is read when present so an old tree still resolves identically.
    from barl_jax.run_config import run_defaults
    self.run = run_defaults()
    self.run.update(_load_yaml(root / "run" / "defaults.yaml"))
    self.envs       = _load_yaml(root / "run" / "envs.yaml")
    # A frozen snapshot makes this tree a replay of one past run: its `resolved:`
    # map is served directly by resolve_slot/resolve_params/resolve_training, so
    # replay re-runs no layering at all and cannot drift with config/. The raw
    # blocks it also carries are deep-merged over the files purely so explain()
    # can still attribute values for that run.
    self.snapshot = snapshot
    if snapshot:
      for attr, key in (("presets", "presets"), ("components", "components"), ("models", "models")):
        setattr(self, attr, deep_merge(getattr(self, attr), snapshot.get(key) or {}))

  # -- replay ---------------------------------------------------------------

  def _replayed(self, prefix: str) -> dict:
    """The snapshot's resolved values under one prefix, with the prefix stripped."""
    resolved = (self.snapshot or {}).get("resolved") or {}
    return {k[len(prefix):]: v for k, v in resolved.items() if k.startswith(prefix)}

  @property
  def replaying(self) -> bool:
    return bool((self.snapshot or {}).get("resolved"))

  # -- presets --------------------------------------------------------------

  def ladder_kwargs(self, kind: str, name: str) -> dict:
    """A size ladder rung by dotted path representation: `sizes.critic.large`
    -> its kwargs. Entirely from the code-declared ladders (barl_jax.hparam)."""
    from barl_jax import hparam
    return hparam.ladder(kind, name)

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

  def policy_class_name(self, policy_name: str) -> str:
    """A model's Python class, from its own `policy:` key -- defaulting to the
    model name itself, which is what leaves a 1:1 model (NCLQL, or any model
    before this key existed) untouched. Deliberately YAML-only: scripts/grid.py
    loads this module by path precisely to avoid importing barl_jax, so the
    model->class mapping must not need the Python registry to resolve."""
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
    profile_cfg = dict(self._profile_cfg(policy_name, profile).get(slot, {})) if profile else {}
    kind = self.component_kind(cls_name)

    # The class's own declaration is the baseline: style surfaces (expanded
    # from the effective selector, so an override swaps the whole net), then
    # the structural Config and HParam defaults, composed down the MRO.
    cls = _component_cls(cls_name)
    chosen = _chosen_styles(cls, slot, slot_cfg, profile_cfg,
                            {k: v for k, v in self.overrides.items()
                             if k.startswith(f"components.{slot}.")})
    layers = _decl_layers(cls, chosen=chosen)

    # `size:` resolves one rung of the component kind's capacity ladder, at a
    # fixed rung so an explicit `latent_dim:` on the slot still shadows it.
    size = (self.overrides.get(f"components.{slot}.size")
            or profile_cfg.get("size") or slot_cfg.get("size"))
    if size is not None:
      layers.append((f"size: {kind}.{size}", self.ladder_kwargs(kind, size)))

    # The slot's own YAML kwargs (no extends -- structure lives in the class),
    # then the profile's, then the run's dotted overrides.
    if slot_cfg:
      layers.append((f"models: {policy_name}.components.{slot}",
                     {k: v for k, v in slot_cfg.items() if k not in _META_KEYS}))
    if profile and profile_cfg:
      layers.append((f"models: {policy_name}.profiles.{profile}.{slot}",
                     {k: v for k, v in profile_cfg.items() if k not in _META_KEYS}))

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
    entry = self.components.get(cls_name) or {}
    if "kind" in entry:
      return entry["kind"]
    return getattr(_component_cls(cls_name), "kind", None)

  def size_table(self, kind: str) -> dict:
    from barl_jax import hparam
    return {name[len(f"sizes.{kind}."):]: kwargs
            for name, kwargs in hparam.BUNDLES.items()
            if name.startswith(f"sizes.{kind}.")}

  def size_keys(self, kind: str) -> set:
    """The kwargs a kind's size ladder owns. Nothing else may set them."""
    return {key for entry in self.size_table(kind).values() for key in entry}

  def component_kwargs(self, cls_name: str, size: str | None = None) -> dict:
    """A component class's own baseline, expanded, with no policy slot involved:
    its inherited Config/HParam defaults plus a default-sized ladder. This is
    what a component built outside a Policy (a test, a notebook) resolves -- so
    such a build sees exactly the first rungs a real run's resolution gives it,
    rather than a hand-written dict that drifts."""
    cls = _component_cls(cls_name)
    layers = _decl_layers(cls)
    kind = getattr(cls, "kind", None)
    table = self.size_table(kind)
    if table:
      if size is None:
        size = "small" if "small" in table else sorted(table)[0]
      layers.append((f"size: {kind}.{size}", self.ladder_kwargs(kind, size)))
    return merge(layers)

  def resolve_slot(self, policy_name: str, slot: str, profile: str | None) -> tuple[str, dict]:
    cls_name, layers = self.slot_layers(policy_name, slot, profile)
    live = merge(layers)
    if not self.replaying:
      return cls_name, live
    # Overlay, not replace: a key the snapshot recorded is authoritative (it is
    # what this run's weights were built against), but a key it never recorded
    # -- because it predates a config change that added a new required kwarg --
    # takes today's value instead of raising. `__init__` already deep-merges the
    # snapshot's raw blocks over the live files, so the live layering under a
    # snapshot already carries every key the snapshot doesn't override; this is
    # what makes that union authoritative rather than the snapshot's kwargs
    # alone. `class`/`size` are metadata, not constructor kwargs, and stay
    # popped exactly as the pre-overlay replay path popped them.
    replayed = dict(self._replayed(f"components.{slot}."))
    replayed.pop("size", None)
    cls_name = replayed.pop("class", cls_name)
    return cls_name, {**live, **replayed}

  def slots(self, policy_name: str) -> list[str]:
    if self.replaying:
      resolved = self.snapshot["resolved"]
      return sorted({k.split(".")[1] for k in resolved if k.startswith("components.")})
    return list(self.policy_cfg(policy_name).get("components", {}))

  # -- params / training ----------------------------------------------------

  def params_layers(self, policy_name: str, profile: str | None) -> list[tuple[str, dict]]:
    # The policy class's declared HParam defaults are the base layer, so a
    # param a model YAML never mentions still resolves -- with attribution.
    layers = [("decl params", _hparam_defaults(_policy_cls(policy_name, self)))]
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
    # Same overlay as resolve_slot: a param the snapshot recorded wins, a param
    # it never recorded (added to config/ after that run) falls back to today's
    # value rather than being missing from the dict entirely.
    return {**live, **self._replayed("params.")}

  def resolve_training(self, name: str, profile: str | None = None) -> dict:
    """The run keys `name` sets for itself, restricted to TRAINING_KEYS. Takes a
    *model* string rather than a policy class name, because its caller only ever
    has the `model:` a run entry named. Sits below the run config's own env/run
    levels: a model states what its algorithm wants, an explicit per-env or
    per-run statement still wins."""
    if self.replaying:
      return {k: v for k, v in self._replayed("training.").items() if k in TRAINING_KEYS}
    policy_name, alias_profile = self.resolve_model_name(name)
    profile = profile if profile is not None else alias_profile
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
    out.update({f"training.{k}": v for k, v in self.resolve_training(policy_name, profile).items()})
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
    the code or YAML: the fully resolved values under `resolved:` (the
    authoritative copy -- replay reads these and never re-runs the layering),
    plus the model blocks and the size-ladder rungs it resolved against, kept
    so a past run's provenance is still explainable."""
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
      size, kind = self.slot_size(policy_name, slot, profile), self.component_kind(cls_name)
      if size is not None:
        used_presets[f"{SIZES}.{kind}.{size}"] = self.ladder_kwargs(kind, size)

    return {
      "resolved":   self.resolved(name, profile),
      "models":     models,
      "presets":    _nest(used_presets),
    }


def _nest(flat: dict) -> dict:
  """`{"sizes.critic.large": {...}}` -> `{"sizes": {"critic": {"large": {...}}}}`,
  so a snapshot's presets load back into the same shape the config tree walks."""
  out: dict = {}
  for path, value in flat.items():
    node = out
    parts = path.split(".")
    for part in parts[:-1]:
      node = node.setdefault(part, {})
    node[parts[-1]] = value
  return out
