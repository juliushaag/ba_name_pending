# Configuration class restructuring plan

## Outcome

Replace the `hparam.Config` declarations, `ConfigDict`, `ConfigOwner`, and
`ConfigMixin` access path with typed immutable configuration classes. Every
policy, component, optimizer role, model variant/profile, tier/style preset,
and run-settings group owns a class whose fields contain its defaults.

YAML stays deliberately small: one YAML mapping describes one run. A loader
uses `yaml.safe_load`, recursively applies the mapping to a complete typed
default object, and strictly validates the result. There is no experiment
language, include graph, provenance database, root-overlay hierarchy, or
implicit multi-run envelope in the core configuration API. Grid/evaluation
callers may keep their own small file formats at their boundaries, while
training receives one `RunConfig`.

The current implementation is spread across:

- `src/barl_jax/hparam.py`: `Config`/`ConfigDict` declarations;
- `src/barl_jax/components.py`: `ConfigOwner` MRO discovery and static-value
  handling;
- `src/barl_jax/config/toolkit.py`: YAML includes, typed dotted access, and
  `ConfigMixin`;
- `src/barl_jax/config/tree.py`: model/slot/profile/tier precedence,
  snapshots, and validation;
- `src/barl_jax/config/run.py`: `RUN_CONFIGS`, run defaults, loading, and run
  validation;
- actors, critics, networks, optimizers, and `policy.py`: class-attribute
  declarations and flat dictionary consumers; and
- `scripts/train.py`, `scripts/evaluate.py`, `scripts/submit.py`, and
  `launch/grid.py`: training, evaluation, and grid file boundaries.

The ignored `src/barl_jax/build/lib` tree contains an earlier `_configs` /
`_hparams` split. Treat it as historical evidence only; do not reproduce a
second declaration framework.

## Target typed API

Use ordinary frozen dataclasses beside the code they configure and a few small
free functions for recursive construction, type checking, and JSON-safe
serialization. Do not add a permanent `ConfigBase`, `ConfigSource`,
`ConfigResolver`, or generic mapping framework. `RunConfig.from_dict` is the
single materializer used by YAML and by Python callers.

The public training API is:

```python
from barl_jax.config import from_yaml
from barl_jax.config.run import RunConfig

run = from_yaml("runs/sac_halfcheetah.yaml")  # -> RunConfig
run.validate()
print(run.to_dict())                         # fresh JSON-safe copy

base = RunConfig.defaults(model="SAC", env="HalfCheetah")
base = base.replace(num_seeds=3, batch_size=512)
base.validate()
```

`from_yaml(path) -> RunConfig` is a thin `safe_load` plus
`RunConfig.from_dict` wrapper. It accepts exactly one mapping and requires
`model` and `env` selectors. The minimal supported file is:

```yaml
model: SAC
env: HalfCheetah
profile: null
num_seeds: 3
batch_size: 512
model_overrides:
  params:
    policy_delay: 2
  components:
    actor:
      mlp:
        activation: silu
```

`model_overrides` is a typed nested section, not a dotted-path escape hatch.
The implementation may retain a short compatibility conversion for existing
saved sweep manifests at the script boundary, but new YAML and all runtime
objects use the nested shape above. There is no `defaults:` plus `runs:`
envelope for training configuration.

`RunConfig.defaults(model=..., env=..., profile=None)` is the one defaults query
for scripts and grid construction. It returns a complete fresh `RunConfig`,
including the selected typed model variant, all nested component/preset values,
model training settings, environment class defaults, and run settings. It must
resolve names and class metadata without reading YAML/files, calling
`get_env()`, importing a simulator factory, allocating JAX arrays/devices, or
creating an environment. Unknown model, profile, component, tier, style, or
environment names fail immediately with actionable errors.

`RunConfig.from_dict(mapping)` applies recursive overrides to a fresh defaults
object. It accepts plain Python mappings from grid code and test fixtures, so
scripts can construct and validate a run without a temporary YAML file.
`RunConfig.to_dict()` returns a deep, JSON-safe copy with no references to the
config object or mutable default values. `RunConfig.validate()` returns no
problems or raises/returns the existing key-specific validation diagnostics;
choose one convention and use it consistently across all callers.

## Typed classes and default ownership

Define `RunConfig` with the fields currently in `RUN_CONFIGS`, plus `model`,
`env`, and optional `profile`/typed `model_overrides`. Its class fields are the
sole built-in run defaults. Preserve the current cross-field rules for
checkpoint/resume, cadence positivity, vector divisibility, seed counts,
logger, learning-rate decay, and execution modes.

Define nested classes for policy parameters, training settings, component slots,
optimizer roles, MLP settings, time embeddings, schedules, noise schedules,
and environment metadata. Add typed classes beside the implementations for
`StochasticActor`, `ScoreContinuousActor`, `DDPMActor`, `DISActor`,
`LangevinActor`, `C51Critic`, `FloqCritic`, `AutotuneRegulator`, and all
registered schedule/time components. A component advertises its schema with
`config_type` and receives an already materialized instance:

```python
@dataclass(frozen=True)
class StochasticActorConfig(ActorConfig):
  mlp: MLPConfig = field(default_factory=StochasticActorMLPConfig)
  squash_cls: str = "TanhSquash"
  log_std_min: float = -20.0
  log_std_max: float = 2.0
  log_prob_clip: float = 0.99

class StochasticActor(ActorOptim, Actor):
  config_type = StochasticActorConfig

  def __init__(self, env_stats, config: StochasticActorConfig, *, rngs, **kwargs):
    super().__init__(env_stats, **kwargs)
    self.net = MLP(..., **config.mlp.to_mapping())
```

All built-in defaults belong to these typed classes. This includes the five
model variants (AMDP, DIME, NCLQL, SAC, SDAC), their profiles/aliases, capacity
tiers, time/style surfaces, environment metadata defaults, and nested schedule
values. A registry maps a name to a class only; it never carries a parallel
default dictionary. Replace actor `POLICY` dictionaries and `MLP_TIERS` /
`TIME_STYLES` value tables with typed variant/preset classes or class-based
registries whose values are owned by those classes.

Shared defaults currently inherited through `ConfigOwner` (`OptimizerConfig`,
`CriticLossConfig`, `Polyak`, policy bases, and actor/critic mixins) must become
explicit dataclass base classes or nested fields with the same derived-class-wins
behavior. Test inherited, overridden, and newly introduced fields before
removing MRO discovery. Mutable defaults use factories and are copied on every
materialization. Convert mappings to immutable runtime values only at the NNX
boundary; that storage concern must not leak into the config API.

## Recursive materialization and validation

Implement one recursive materializer over dataclass fields and type hints:

1. start from `RunConfig.defaults(...)`, which selects the model/profile and
   materializes all nested schema/preset defaults;
2. recursively merge only keys present in the input mapping;
3. construct nested dataclass instances and copy mutable collections; and
4. reject unknown keys, missing required selectors, wrong scalar/container
   types, bool/int confusion, invalid enum/selector values, and invalid
   cross-field relations with dotted key paths.

Selector values (`profile`, `tier`, style names such as `time_emb`, and explicit
component classes) are read first. The selected variant/preset classes are then
materialized before ordinary nested overrides are applied. This two-phase step
ensures a partial YAML override cannot erase hidden style/schedule defaults.

The canonical serializer must include every effective value, including hidden
nested schedule/noise kwargs, selected class names, profile identity, and tier/
style selectors. A flat projection may preserve current
`params.*`, `training.*`, and `components.<slot>.*` names for logging and
checkpoint metadata, but it is an output format. The new YAML input contract is
nested and strict; indefinite support for arbitrary legacy flat paths is not a
goal.

Environment facts are separate from defaults. Static benchmark metadata such as
return bounds and declared environment defaults belongs on typed environment
classes. Live spaces, dimensions, horizon, renderer capabilities, and other
simulator-derived values are computed during setup and may derive or validate a
`RunConfig`; `RunConfig.defaults` must not construct the environment or invoke
those runtime computations.

## Model variants, profiles, tiers, and styles

Replace mutable actor `POLICY` dictionaries with complete typed model-variant
classes. Each built-in variant owns policy implementation, component slots,
training defaults, policy parameters, and selector defaults. Profiles are typed
subclasses or typed companion classes with complete materialized defaults and
explicit field overrides. Aliases refer to a variant/profile class; they do not
store default mappings.

Capacity tiers and time/style surfaces become typed preset classes. Preset names
resolve through name-to-class lookup and are validated against the target
component schema. Tiers of one kind must expose comparable fields where grid
comparisons rely on shape. Styles must serialize all selected and hidden fields.

The useful public query surface remains small:

- `from_yaml(path) -> RunConfig`;
- `RunConfig.from_dict(mapping)`;
- `RunConfig.defaults(model, env, profile=None)`;
- `RunConfig.to_dict()` and `RunConfig.validate()`; and
- existing name availability queries where scripts already use them.

No generic `resolve_slot`, `resolve_params`, `ConfigMixin.get`,
`assert_all_consumed`, or provenance/explanation framework is needed for the
new design.

## Grid and evaluation callers

Grid construction queries `RunConfig.defaults(model, env, profile)` directly,
then applies each Python axis value with `dataclasses.replace` or a typed
nested override and calls `validate()` before writing/submitting anything. It
may continue to read a tiny grid YAML `{axes, exclude}` document with
`yaml.safe_load`; that file is a grid specification, not a training config and
never enters `from_yaml`. Each expanded point can be written as one minimal
single-run YAML file and then loaded by `from_yaml`, or passed directly as a
`RunConfig` to a local executor.

Evaluation keeps its own `EvalConfig`/entry parsing at `scripts/evaluate.py`'s
boundary because checkpoint paths, episode counts, and output are not training
fields. `scripts/submit.py` can batch those evaluation entries without routing
them through `RunConfig`. Training, evaluation, and grid documents therefore
remain three small, separate formats without a shared experiment envelope.

Update `scripts/train.py` to call `from_yaml` once per run file and pass the
returned object into trainer setup. Update `scripts/submit.py` and
`launch/grid.py` to construct/validate typed configs in Python, and update
trainer setup to consume `RunConfig` rather than rebuilding a `ConfigTree` from
raw dictionaries.

## Checkpoints and fingerprints

Keep versioned checkpoint architecture metadata as a separate concern.
`ModelSpec` may remain as the temporary or stable serialization boundary, but
its payload must be generated from the typed `ModelConfig`/`RunConfig` output,
not from `ConfigTree` resolution methods. A snapshot includes schema version,
canonical model/profile identity, selected classes, tier/style selectors, and
all materialized nested values. Loading a checkpoint uses saved resolved values
as authoritative; a migration function may fill only fields introduced by a
known schema version. Current class defaults must never silently replace saved
values.

Model fingerprints hash canonical resolved model values. Experiment fingerprints
add run settings while excluding per-instance identity keys. Equivalent typed
objects and their canonical flat projections hash identically. Changing a
hidden nested schedule value changes the versioned fingerprint.

## Concrete package/data changes

The source tree currently lacks tracked `src/barl_jax/defaults/`, while
`scripts/train.py` points at `CONFIG_DIR / "experiments" /
"online_learning.yaml"` and `src/barl_jax/pyproject.toml` declares
`defaults/**/*.yaml` package data. Restore concrete tracked entry files rather
than relying on ignored `build/lib` data.

Track these minimal files:

`src/barl_jax/defaults/experiments/online_learning.yaml`:

```yaml
model: SAC
env: HalfCheetah
```

`src/barl_jax/defaults/experiments/sweep_grid.yaml`:

```yaml
axes:
  algo: [SAC]
  env: [HalfCheetah]
  seed: [0]
exclude: []
```

They are entry examples and contain no algorithm/run/tier/style defaults.
Keep the `defaults/**/*.yaml` package-data declaration, add an installed
resource test for both files, and update `scripts/train.py`, `scripts/submit.py`,
`src/barl_jax/README.md`, and `README.md` help/examples. The old package
`run/defaults.yaml`, `run/envs.yaml`, model YAML, and include-root machinery are
not recreated as built-in default sources.

## Migration sequence

1. Add typed dataclasses and the minimal recursive materializer. Implement
   `RunConfig.defaults`, `RunConfig.from_dict`, `RunConfig.to_dict`,
   `RunConfig.validate`, and `from_yaml` with strict one-mapping semantics.
2. Add complete typed AMDP, DIME, NCLQL, SAC, and SDAC variants/profiles plus
   typed tier/style/time/schedule presets. Implement selector-first then
   materialization-second resolution and canonical serialization.
3. Convert actors, critics, optimizers, policy, environment metadata, and
   `Policy.build_component` to receive typed config instances. Preserve NNX
   private storage helpers only where runtime graph hashing requires them.
4. Update scripts/train, trainer setup, launch/grid, submit, and evaluation
   boundaries. Grid must query defaults and validate each point in Python;
   evaluation remains a separate parser; one run file is written per point.
5. Convert checkpoint metadata, model snapshots, validation, and fingerprints
   to typed serialization. Add the tracked package entry YAML and package
   resource tests.
6. During migration only, adapters may translate old raw manifests or
   `ConfigTree` calls for existing tests/checkpoints. Delete `ConfigTree`
   resolution methods, `ConfigMixin`, `ConfigOwner`, `RUN_CONFIGS`,
   `run_defaults`, generic `Config.load`, actor `POLICY` default dictionaries,
   and any short-lived `load_config` adapter before completion. Do not retain a
   second configuration API.

Do not delete YAML parsing itself: `from_yaml` and the small grid/evaluation
boundary readers still need safe parsing. Do not add include/deep-merge/root
overlay behavior to the new training loader.

## Acceptance and validation

- `RunConfig.defaults(model, env, profile)` returns a complete independent
  object without reading files, creating environments, or touching JAX devices.
- `RunConfig.from_dict` recursively overrides nested defaults, preserves
  untouched siblings, isolates mutable values, and rejects unknown keys and
  invalid types with dotted diagnostics.
- `from_yaml` is exactly one safe-loaded mapping with required model/env; an
  envelope containing `runs` or `defaults` is rejected for training.
- Selector-first resolution materializes complete tier/style/preset surfaces,
  including hidden schedule/noise values. Invalid selectors and profile/model
  combinations fail before setup.
- MRO-composed dataclass defaults preserve current base/derived semantics.
- All five built-in variants resolve to typed objects with effective values and
  canonical snapshots matching the current behavior, including aliases and
  profiles.
- Run validation retains checkpoint/resume, cadence, divisibility, logger,
  decay, execution-mode, and seed rules; grid validates every Python axis point
  before submission.
- `EvalConfig` and grid parsing reject fields belonging to another format.
- Canonical serialization/fingerprints include selected classes, profile/tier/
  style identity, and hidden nested schedule defaults; versioned checkpoint
  snapshots restore without current YAML files.
- Plugin policy/component/environment names resolve through registries that map
  names to classes and schemas, without a central default dictionary.
- Existing policy, critic, evaluation, grid, checkpoint, resume, and package
  resource tests pass through migration adapters and after old machinery is
  removed.

## Editing guardrail

This document is a plan-only artifact. Before any follow-up edit, inspect the
current file and working-tree status, preserve concurrent agent changes, and
apply incremental edits. Do not overwrite source/config implementation while
this plan is under review; implementation begins only after the parent agent
accepts this concrete contract.
