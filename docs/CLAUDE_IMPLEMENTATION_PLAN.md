# Implementation handoff for Claude Code

## Mandate and current state

Continue the QOL backlog in `docs/QOL.md`, while making the framework smaller,
more coherent, and easier to extend. The user explicitly prefers simplification
and net reduction in implementation code to accumulating helper layers.
The repository is in development: change internal APIs and update every caller
and test together. Do not maintain old APIs, checkpoint readers, config paths,
re-exports, forwarding methods, or migration shims for hypothetical users.

The attempted QOL implementation was reverted before this handoff. This document
is a plan, not a claim that the features below exist. The earlier modularity,
configuration, packaging, and checkpoint refactors remain in the working tree.
They are uncommitted; do not reset to HEAD or discard them as unrelated changes.
Inspect `git status` and the actual files before making changes.

The pre-QOL source snapshot measured 63 first-party Python files, 9,684 physical
lines, and 4,959 non-docstring AST statements. Its scope was
`src/barl_jax/**/*.py` excluding build directories, plus `scripts/*.py`; tests,
docs, defaults, and third-party code were excluded. Recompute this baseline if
work has continued. The revert restored all snapshot Python files byte-for-byte.

Previously recorded verification was 547 tests passed and 3 skipped, clean config
lint, a launcher dry run, and a built/installed wheel smoke test. These are
historical results, not a substitute for testing subsequent changes.

## Repository alignment

This is a research framework for online off-policy actor-critic learning using
JAX and Flax NNX. SAC, AMDP, DIME, SDAC, and NCLQL should share infrastructure
without hiding the mathematical distinctions between their algorithms.

Priorities, in order:

1. Preserve algorithm correctness, state ownership, and reproducible experiments.
2. Make invalid experiments fail before allocating expensive resources.
3. Make the common development loop work locally, on CPU, without network access.
4. Make adding an algorithm, component, environment, or executor a local change.
5. Reduce duplicated behavior, configuration, and operational paths.
6. Optimize measured bottlenecks without obscuring the implementation.

This is not a general workflow engine, service framework, or universal simulator
abstraction. Do not introduce a dependency-injection container, event bus, class
hierarchy for every utility, or a second configuration language. Use explicit
arguments, small functions, and existing ownership boundaries. Add a protocol
only when there are real interchangeable implementations or a useful test seam.

Research extensibility means adding a new objective without editing collection,
or a new environment without importing its simulator during config inspection.
It does not mean pretending every critic, sampler, or simulator has identical
semantics. Represent capabilities explicitly and reject unsupported combinations.

## Current ownership map

| Area | Owner | Boundary to preserve |
| --- | --- | --- |
| Declarations and config resolution | `hparam.py`, `config/` | Pure configuration; no active global experiment |
| Frozen model construction inputs | `config/model.py` | `ModelSpec` records resolved values and provenance |
| Component contracts and NNX utilities | `components.py`, `contracts.py` | Components do not depend on the policy runtime |
| Algorithm composition | `policies/`, `policy.py` | Compose components and objectives; no experiment I/O |
| Critic representation and objectives | `critics/`, `objectives.py` | Prediction/regression primitives separate from TD objectives |
| Optimizer ownership and cadence | `optimization.py` | One validated owner for each trainable parameter |
| Environment construction | `env.py`, `environments/` | Metadata is cheap; simulator construction is lazy |
| Live environment interaction | `collection.py` | Collector owns episode and pending action-chunk state |
| Replay storage and sampling | `rollout.py` | Independent sampling RNGs; injected transition sink |
| Evaluation | `evaluation.py` | Action-provider boundary, independent from training orchestration |
| Seed batching | `multiseed.py` | Independent seed state with a shared compiled structure |
| Policy persistence | `checkpoints.py` | Architecture snapshot and policy state |
| Training persistence | `training_checkpoint.py` | Optimizer, replay, RNGs, counters, progress |
| Run lifecycle | `trainer.py` | Resource acquisition, scheduling, coordination |
| Reporting | `logger.py`, `metrics.py` | Host-side sinks versus compiled metric collection |
| Experiment execution | `scripts/submit.py`, `grid.py`, `executors.py`, `hpo.py`, `tmux_pool.py` | Frozen run specs and executor-specific process management |

Read each owning implementation and its tests before designing replacements.
The map describes boundaries, not permission to split each module again.

## Invariants that must survive simplification

### Configuration, imports, and extension loading

- Keep package `__init__.py` files lightweight. Importing a leaf module still
  executes its package initializer. Config tooling must not initialize JAX or a
  simulator merely to list names or inspect defaults.
- Pass `ModelSpec` or `ConfigTree` explicitly. No process-global current model,
  active snapshot, or configuration mutation that affects another experiment.
- `HParam` declares searchable algorithm knobs; `Config` declares structural
  settings. A more-derived declaration wins even if it changes declaration kind.
- Defaults have one packaged home: `src/barl_jax/defaults`. Use package resources
  and the explicit config-root override. Do not restore a root `config/` alias.
- Saved model snapshots must be complete. Missing required fields fail; filling
  them from today's defaults silently changes the restored experiment.
- Preserve `module:Class` selection for external implementations. Classes and
  bound environment factories must be normally importable by spawned workers.
  Do not load files under invented package identities or modify `sys.modules`.
- Objective compatibility is an explicit capability check before construction.
  Do not restore critic loss forwarding methods or infer semantics from names.

### Collection and replay

- Policy actions use canonical bounds `[-1, 1]`. Action plans have shape
  `(batch, chunk_size, act_dim)`; critic inputs flatten the plan where required.
  Convert to actual environment bounds at the stepping boundary.
- Preserve sanitization/clipping accounting and terminal observations before
  reset. Truncation and true termination have different bootstrap semantics:
  only true termination removes the bootstrap term.
- Chunk reward is the discounted sum over the actual executed horizon; the
  bootstrap multiplier uses that horizon, including early episode endings.
- Collection calls must not reset episodes or discard pending chunks. Vector
  environments use explicit masked resets with autoreset disabled.
- Never mix replay partitions belonging to different seeds. Replay RNG state is
  owned by replay and independent of the global NumPy RNG.
- Use the supported spawn/forkserver process strategy around multithreaded JAX.
  Lazy simulator imports and picklable bound factories preserve worker settings.

### JAX, NNX, and optimization

- Keep pure compiled training separate from filesystem, logging, and lifecycle
  code. Avoid Python graph traversal or fresh function creation per env step.
- NNX graph and state are distinct. Static graph attributes must have stable,
  hashable semantics; do not hide arrays or mutable dictionaries in static data.
- Resolve component references from the active merged graph. Cached Python
  references can point at stale modules after split/merge transformations.
- Target parameters are excluded by `TRAINABLE`; target updates occur through
  the intended lifecycle, not through incidental optimizer ownership.
- A summed loss with one backward pass is valid only with correct gradient
  isolation. A frozen critic blocks parameter gradients while preserving action
  gradients; its batch-normalization state must remain frozen as intended.
- Loss keys and optimizer keys agree. Validate unowned and multiply-owned params.
- Delayed updates skip the entire optimizer transform, including Adam moments.
  Zeroing gradients is not equivalent. Keep shared gradient step unbatched where
  required so a vmapped conditional does not execute both update branches.
- State donation invalidates old arrays. Tests must copy/checkpoint before
  advancing rather than reading donated state afterward.
- Metric tiers remain trace-static. Pop collected intermediates once, aggregate
  on device, and cross to the host in a batch. Avoid per-leaf `.item()` calls.
- Keep inference's graph/state handling compatible with JIT caching and refresh
  captured state after training updates. Do not serve stale policy parameters.
- Scanning multiple unlogged updates amortizes dispatch, but changing scan/JIT
  boundaries can change floating-point association. Test tolerances and learning
  semantics; do not promise bitwise equality without evidence.
- Observation normalization currently updates from sampled training batches,
  including repeated replay samples. It does not count unique environment
  observations. Changing this is an algorithm change, not a cosmetic refactor.
- Preserve return-support bounds for distributional critics and reward scaling
  relationships. Expectation and pessimistic value are different operations.
- The current ManiSkill path is a CPU/NumPy bridge. Do not claim device-native
  simulation or introduce Torch into the JAX training core.

### Persistence and lifecycle

- Policy checkpoints currently accept version 2 only. Reject unsupported formats
  clearly. Restore the saved architecture before constructing policy state.
- `weights` starts fresh optimizer, replay, counters, and independent policy RNGs,
  while retaining saved normalizer and target weights.
- `resume` restores all seed states, optimizer/schedule, replay physical rows and
  write positions, sampling/collector RNGs, and progress. Training settings must
  match except the explicitly permitted output/logging/evaluation settings.
- Current resume restarts environment episodes and discards pending chunks. Do
  not describe this as exact trajectory continuation.
- Resume is saved after evaluation iterations and clean completion. A failed
  iteration retains the previous successful resume; final weights may be newer.
- Register resources with the run's `ExitStack` immediately after acquisition.
  Setup, warmup, validation, logger construction, and cleanup failures matter as
  much as exceptions in the main loop. Attempt all cleanup without masking the
  primary failure. Allocate render environments only when needed.
- Training requires an accelerator unless explicitly opted into CPU. Launcher
  boundaries must not accidentally carry test-only `JAX_PLATFORMS=cpu` into jobs.
- Preserve executor exit status, signal handling, and Slurm output copyback.
  Use fake executors in tests; tests must not submit jobs or contact W&B.

## Where to reduce code

The baseline hotspots include `trainer.py` (801 lines), `policy.py` (492),
`config/tree.py` (494), and several 200–340 line launcher/executor scripts.
Large size is a clue, not proof of poor design. Moving the same code into more
files does not reduce complexity or satisfy the user's request.

First inventory duplicated config resolution, result serialization, run metadata,
report reading, launch construction, and train/validate setup. Trace all callers.
Consolidate only code with the same semantics and state ownership. In particular:

- Use one resolved run specification through submit, HPO, execution, and reporting.
  Reuse existing manifest machinery rather than inventing a parallel database.
- Build one reporting path with interchangeable sinks. Share event formatting and
  scalar conversion, retaining existing useful terminal output.
- Centralize presentation JSON writing/reading and reuse it for results/status.
  Keep exact checkpoint state serialization separate.
- Share preflight validation across config loading, CLI execution, and direct
  `setup` callers. Keep cross-field checks at one boundary.
- Keep the numerical training kernel local. Extract host orchestration only when
  its independent use removes duplication or substantially clarifies ownership.
- Remove stale imports, dead branches, wrappers, and misleading comments after
  their replacements work. Update every caller in the same change.

Track physical production lines and non-docstring AST statements separately from
new tests and docs at every phase. The final production implementation should be
smaller than the baseline. Do not game this by deleting tests, removing useful
explanations, compressing statements, or moving implementation out of the measured
scope. If a feature cannot fit the reduction target responsibly, show its cost and
identify concrete consolidation before claiming completion. Correctness takes
precedence over an arbitrary numerical reduction.

## Implementation sequence

Each phase should leave a runnable repository with all callers updated. The order
below resolves dependencies; P3 is research work after the operational foundation.
Do not start by building a universal CLI or framework to host all future features.

### Phase 1 — reproducible verification and developer diagnostics

**Backlog:** CI, fast check, doctor, Ruff and gradual typing.

Provide one documented fast-check entry point reusing config lint and focused
boundary tests. Avoid a custom task-runner dependency if a small existing-command
wrapper suffices. Keep the full test suite separately discoverable.

Add CI for CPU tests, config lint, and a clean wheel installation. Root `uv sync`
includes CUDA and substantial simulator dependencies; design the CPU job around
the standalone library and explicit test dependencies rather than assuming the
root research environment is cheap or portable. Report optional simulator skips
separately and keep an optional integration job where appropriate.

The wheel check must run from outside the repo without source-path injection.
Check package/config imports, packaged defaults, registry resolution, and a tiny
policy construction. Inspect archives for missing packages and unwanted build
content. The nested setuptools package mapping currently uses `.`; reverting to
parent-relative discovery previously produced broken artifacts. Stale egg-info
manifests can also retain bad paths. Build in a clean isolated workspace.

A doctor command should report versions, selected JAX backend and devices,
accelerator visibility, optional simulator availability, submodule readiness, and
render probe results. Separate cheap inventory from active backend/render probes;
execute probes in subprocesses with timeouts so native crashes cannot kill the
report. Missing optional dependencies are informational unless requested. Return
a nonzero exit code for failed required probes and offer machine-readable output.
Do not import every optional simulator to check its installation.

Start Ruff with a deliberate small rule set. Type-check config and contracts
first, then actual boundaries. Do not mass-format algorithms or add broad ignores
to manufacture a passing baseline. Correct stale comments that still claim root
config directories or package-import registry side effects.

**Acceptance:** clean CPU checkout can run documented checks; installed-wheel
smoke works without the source tree; doctor survives missing and crashing optional
backends; fast check performs no network logging or real job submission.

### Phase 2 — configuration validation, logging, and result contracts

**Backlog:** preflight validation, offline/no-op logger, strict result JSON.

Implement validation after effective values are resolved and before JAX/env/logger
allocation. Validate direct `setup` calls as well as YAML/CLI users. Name the run,
key, value, and violated relation in errors. Reuse declarations for scalar rules
where practical and keep derived cross-field rules together.

Audit actual units before imposing restrictions: env steps versus vector calls,
per-seed versus total capacity, chunk plans versus stored transitions. Reject bool
where an integer is required, non-finite rates, invalid checkpoint modes, and
unsupported cadence values. Some counts legitimately allow zero. A batch can
exceed replay capacity when sampling with replacement; final partial rollouts
may be valid. Derive each constraint from a consumer and test both sides of it.
Do not copy hardcoded lists of numeric keys into several validators.

Inject a logger factory into the run lifecycle with a small contract covering the
operations actually used. Offer local terminal/JSON, W&B, and no-op sinks. Import
W&B only when selected. Preserve useful Rich presentation through an optional
renderer or equivalent compact implementation. A no-op logger should avoid
unnecessary metric conversion and media construction. Close partially created
sinks through the same resource ownership path.

Use strict presentation JSON: convert missing/non-finite objectives to `null` with
an explicit outcome/reason. Write a temporary file in the destination directory,
flush it, and replace atomically; document durability guarantees. Readers must
handle missing, incomplete, and malformed result files without declaring success.
Update HPO and status readers together. Never apply presentation sanitization to
exact computational state: infinity may be meaningful in metadata such as bounds.

**Acceptance:** invalid configs acquire no resources; a training smoke run works
without W&B installed; logging failure still closes envs; result readers handle
NaN/missing/partial data and never rank an unavailable objective as a valid score.

### Phase 3 — run identity, sweep status, and retry

**Backlog:** status/retry across local, tmux, and Slurm.

Start with `write_manifest`/`read_manifest` in `scripts/executors.py`, plus submit
and HPO readers. Define stable run IDs and attempt IDs. Frozen resolved configs,
model snapshots, and their hashes are the source for retry, not current defaults.
Retain prior attempts and outputs; a retry must not overwrite forensic evidence.

Use an explicit lifecycle: pending, running, completed, failed, with interrupted
or unknown when evidence is insufficient. Completion requires a successful
process outcome and required artifacts. Persist timestamps, executor/job identity,
exit status, and a concise error. Record startup failures before the trainer or
logger initializes; trainer-only status updates miss import failures.

Reconcile persisted state with executor evidence where available. A stale running
file is not proof a Slurm job is alive, nor proof it failed. Define how unknown
scheduler state is presented. Reuse shell/process management already in executors
and preserve Slurm traps, copyback, and array-row mapping.

Provide read-only status and a retry dry run that displays selected attempts and
frozen configs. Actual retry goes through existing executor interfaces. Concurrent
retry requests must not launch the same attempt twice; use a narrowly scoped
reservation/locking mechanism if concurrency is supported.

**Acceptance:** mocked executor lifecycle covers cold-start failure, interruption,
success, missing status, stale state, and retry; changing current defaults does not
change retry specs; dry runs neither launch nor modify submission artifacts.

### Phase 4 — checkpoint scheduling, retention, and inspection

**Backlog:** checkpoint cadence/retention/budget; config diff/checkpoint inspection.

Separate evaluation cadence from resume persistence. Specify cadence units and
whether crossing a threshold triggers one save; avoid modulo equality that misses
thresholds when a rollout advances several steps. Always keep clean-completion
behavior explicit. Reuse checkpoint owners rather than adding persistence to policy.

Write immutable checkpoint generations into staging, complete and validate them,
then publish the committed generation/pointer. Only after publication may retention
delete older committed generations. Protect policy best/final artifacts according
to an explicit separate policy. Define the single-writer assumption or implement
serialization; do not leave concurrency behavior accidental.

A disk budget must account for replay and temporary staging alongside the last
valid generation. Never delete the only recoverable checkpoint to make room for
its replacement. Specify what happens if a single checkpoint exceeds the budget
or saving runs out of space. Fault-inject failures before publication, during
publication, and during cleanup. Filesystem atomic rename alone does not solve
all directory replacement or crash durability problems.

Inspect metadata without constructing policies or envs. Show checkpoint kind,
format version, saved implementation, shapes/seed count, counters, config hash,
and resume limitations. Config diff compares effective values with provenance;
it must distinguish a changed declaration/default from a changed explicit override.
Handle malformed/incomplete snapshots with clear errors, not fallback defaults.

**Acceptance:** interrupted save retains a loadable generation; retention never
removes it; restored counters/RNGs match; cadence is independent of evaluation;
inspection needs no simulator. Explicitly test `weights` versus `resume` behavior.

### Phase 5 — deterministic evaluation, replay contracts, and CPU example

**Backlog:** evaluation seeds, transition validation, tiny end-to-end example.

Give evaluation explicit, independent RNG ownership. Derive seeds stably from run
seed, evaluation identity, environment slot, and episode identity; do not depend
on iteration order or Python's randomized `hash`. Record actual episode seeds and
control the policy action RNG as well as environment reset RNG. Specify whether
successive evaluations reuse episodes or sample new episodes. Evaluation/video
frequency must not consume training RNG state or change training trajectories.

Validate complete transition batches before touching replay rows, indices, or
size. Check leading lengths, observation/action shapes, dtype policy, reserved
field collisions, and supported extras. Validate once at the ingestion boundary,
not repeatedly in compiled training. Specify casting rather than silently making
lossy conversions. Include ring wraparound, over-capacity inserts, zero-length
batches, and rejection leaving the buffer entirely unchanged.

Build a tiny deterministic continuous-control environment with only core CPU
dependencies, selecting it through the existing external-class path where useful.
Use a real policy, collector, replay, training step, policy save/load, and training
resume. No W&B, render dependency, downloads, or accelerator. Keep this example
small enough to run under a minute on a documented reference machine; measure
cold compilation separately from warm execution before advertising a guarantee.

**Acceptance:** repeated evaluation with the same seed schedule reproduces episode
inputs/results; changing eval cadence leaves training RNG untouched; invalid
transition insertion is atomic; the example verifies restored optimizer/replay
state and acknowledges that ordinary resume restarts episodes.

### Phase 6 — scaffolding and performance evidence

**Backlog:** extension scaffold and benchmarks.

Generate a minimal algorithm or environment using today's public boundaries.
Do not copy entire existing algorithms. For algorithms include declarations,
component slots/objective requirements, model YAML where required, and a contract
test. Environment scaffolds need a lazy importable factory and valid metadata;
they do not need meaningless model YAML. Refuse to overwrite existing files.
Use the external class path to avoid requiring edits to central registries merely
to try a generated implementation. Smoke-test generated output in a temporary
project rather than snapshot-testing only template text.

Benchmark import latency in fresh processes, async worker RSS, transitions per
second, and compiled gradient throughput. Report CPU/GPU, library versions,
shapes, batch/chunk/seed counts, precision, and compilation/cache settings. Use
`block_until_ready` for JAX timing, isolate warmup, and compare equal workloads.
Tests disable some XLA optimization for speed; do not reuse those settings for
performance conclusions. Avoid noisy hard CI thresholds until variance is known.

Optimize based on profiles: host/device transfers, redundant graph operations,
logging conversion, replay copies, and compilation proliferation are candidates,
not established bottlenecks. Preserve learning behavior and report memory as well
as latency. Keep a small runnable benchmark harness, not a benchmarking framework.

**Acceptance:** generated code imports and passes its contract test; benchmarks
are repeatable with complete metadata; optimization claims include before/after
measurements on the same hardware and workload.

### Phase 7 — optional extension mechanisms and research replay

**Backlog:** plugin discovery, exact environment resume, sequence/prioritized replay.

These complete the backlog but warrant separate focused changes after earlier
phases. They are not prerequisites for the everyday development loop.

For plugins, define entry-point groups only for supported extension kinds. Cache
inventory discovery, lazily load requested implementations, and reject collisions
with built-ins or other distributions deterministically. Include distribution and
missing dependency in errors. Do not rescan installed metadata during every
component lookup or import all plugins just to list names. Preserve `module:Class`
as a direct explicit selection mechanism, not a legacy fallback.

For exact resume, add opt-in adapters only for environments with demonstrably
complete state capture. State may include simulator internals beyond qpos/qvel,
wrapper counters, reset RNGs, policy RNG, vector-slot observations, episode stats,
and pending chunks. A generic wrapper copying the base RNG is insufficient.
Unsupported environments retain ordinary resume semantics. Verify uninterrupted
versus save/restore continuation with the same actions, resets, transitions,
chunks, and future replay samples. Define platform/determinism limits precisely.

Sequence replay needs episode/seed identities, valid-start tracking through ring
wraparound, boundary masks, and an explicit padding/burn-in policy. Prioritized
replay needs probabilities, normalized importance weights, stable sample IDs, and
priority feedback after learning. A sampler alone does not implement prioritized
training: loss weighting and TD-error feedback must connect through trainer and
objectives. Guard stale priority updates when slots have been overwritten.

Keep ordinary uniform replay simple. Extend the sampling result contract only as
needed; do not force recurrent bookkeeping into every collector or algorithm.

**Acceptance:** plugin conflicts and optional import errors are reproducible;
supported exact-resume adapters pass continuation tests; sequence samples never
cross seed/episode boundaries; priority sampling distribution, weights, updates,
and overwritten-ID handling are tested with controlled RNGs.

## Additional requirement — policy-independent action executors

The user additionally requested Default, Best-of-N (BoN), Mean-of-N, and future
execution strategies that work independently of the policy implementation. Treat
this as part of the implementation scope alongside the original 18 backlog items.
Here “executor” means selecting actions at inference time, not launching jobs.
Use `ActionExecutor` (or an equally explicit name) and an `execution.py` module;
leave `scripts/executors.py` responsible for local/tmux/Slurm processes.

Implement this after Phase 2's config boundary and before Phase 5's evaluation
and example, so both use the same action-provider integration. Begin with Default
and its equivalence tests, then add the two candidate reducers. Do not postpone
this requirement behind plugin discovery or research replay.

### Semantics and minimal interface

Policy independence means the selection algorithm does not branch on SAC, AMDP,
DIME, SDAC, or NCLQL class names. It does not mean every policy supplies every
capability. Reuse the current `ActionProvider.inference_fn()` boundary for callers.
A configured provider exposes the same action dimensions and chunk size and binds
current policy state once per rollout/evaluation, refreshing after updates.

Keep the strategy itself a pure transformation over a candidate-generation
callable, an optional scoring callable, observations, and an explicit key. A small
immutable spec with a mode and candidate count plus shared functions is sufficient
initially. Do not require a subclass hierarchy, registry, or plugin mechanism for
three modes. Keep policy bindings separate from strategy logic.

| Mode | Required capability | Defined behavior |
| --- | --- | --- |
| Default | Current policy inference | One unchanged inference call using the supplied key |
| Best-of-N | Candidate generation and scalar action scoring | Generate N plans per observation, score each, select the highest-scoring complete plan |
| Mean-of-N | Candidate generation; continuous box actions | Generate N plans per observation and average the candidate axis elementwise |

Use this interpretation of Mean-of-N: the arithmetic mean of sampled actions,
not the analytical distribution mean, mean critic estimate, or model ensemble.
Use this interpretation of BoN: best under an explicitly supplied learned score,
not privileged simulator rollouts or true future return. Record those semantics
in CLI/config documentation so comparisons are interpretable.

The first implementation can adapt the existing pure inference step to generate
candidates with `vmap` over split keys. That supports actor-based policies and
NCLQL's direct sampler without forcing a fake actor into NCLQL. Avoid introducing
a second policy sampling API unless it actually replaces duplicated code.

BoN needs an explicit scoring capability. Audit critic `value` semantics and
NCLQL noise conditioning before writing the adapter. The policy binding handles
observation normalization once and correct action-plan flattening/conditioning;
the executor sees only one scalar score per observation and candidate. Freeze
normalizer and BatchNorm state and use the intended online/target critic and
ensemble reduction explicitly. Do not choose between expectation and pessimistic
value silently or infer a critic from arbitrary attribute names. A small binding
method per distinct policy family is preferable to type switches in the executor.
If a scorer is unavailable, reject BoN at preflight with an actionable message.
Default and Mean-of-N must still work without a critic.

### Shapes, random keys, and numerical behavior

- Use a documented candidate shape, for example `(N, batch, chunk, action_dim)`,
  and scoring shape `(N, batch)`. Return `(batch, chunk, action_dim)`. The seed,
  environment, candidate, chunk-time, and action axes must stay distinct.
- Select a whole plan in BoN; choosing different winners per timestep changes
  the method. Mean-of-N averages only the candidate axis, never seeds/envs/time.
- Default passes the caller key through unchanged. Special-case `N=1` for both
  reducers if promising equivalence with Default; splitting even one key otherwise
  changes samples. Use explicit split/fold-in keys for `N>1`, no global RNG state.
- Candidate keys should produce independent samples within each environment;
  include multiple environments and seeds in tests rather than only batch size 1.
  Candidate batching must not update normalization statistics or couple scores
  through training-mode BatchNorm.
- Restrict averaging to supported continuous box actions. Averaging can land
  between modes and produce poor behavior even while within bounds; expose this
  as a selectable research strategy, not an automatic improvement.
- Apply environment-bound conversion once at the existing stepping boundary.
  Define invalid-candidate handling before scoring/reduction so a NaN cannot win
  or contaminate a mean. Integrate existing sanitization counters. For BoN mask
  non-finite scores, break ties deterministically, and define the all-invalid
  outcome explicitly (fail clearly or use a recorded, configured fallback).
- Treat mode and N as static compilation choices. Keep state/observations/keys
  dynamic, use module-level compiled functions, and avoid splitting the NNX graph
  N times in Python or round-tripping scores through the host.
- Benchmark memory and latency as N grows. Add candidate microbatching only if
  measured needs justify it, preserving the same candidate keys and reductions.

### Integration and configuration

Use one immutable execution spec; expose distinct collection and evaluation
settings referencing that schema. Proposed conceptual fields are `mode` and
`num_candidates`, with scoring options only where needed. Choose final key names
in the existing declaration system and register them once. Default both contexts
to Default so this change preserves current behavior. Reject unknown modes,
non-integral/nonpositive N, and incompatible scoring/action-space combinations
before expensive resources are allocated.

Bind through both `Policy.inference_fn()` and `StackedPolicy.inference_fn()` using
one shared execution operation. The stacked path currently calls the policy's
inference step directly; changing only single-policy inference would silently
leave multi-seed collection unchanged. Collector and `run_episodes` should keep
consuming `ActionProvider` without strategy branches. Preserve the uniform random
warmup provider rather than requiring it to have a critic.

Do not route actor losses, Bellman target sampling, or internal diffusion/Langevin
steps through the executor. Collection-time selection changes the behavior
policy; evaluation-time selection changes reported results. Applying selection
to training targets is a separate algorithm change requiring its own derivation.

Persist the resolved execution specs in run metadata and comparison outputs.
Collection strategy changes are training-config changes for resume validation;
evaluation strategy changes can be permitted but must be recorded in evaluation
results. A weights-only load accepts an explicit execution spec without changing
the saved architecture. Execution strategy state should normally be only config:
caller-owned keys already supply reproducibility. If stateful strategies are later
added, define their checkpoint ownership before introducing hidden mutable state.

Keep diagnostics compact: candidate count, selected score where meaningful,
invalid count, and optional disagreement. Do not attach every candidate to every
transition. Existing inference extras need an explicit rule: Default preserves
them, BoN can gather candidate-associated extras with the winning index, and
Mean-of-N must omit or separately aggregate extras that do not describe the averaged
action. Do not accidentally report one sample's log probability for the mean or
selected action, whose induced distribution differs from the original policy.

### Acceptance tests and implementation order

1. Extract/bind the shared Default path and prove unchanged actions, extras, keys,
   and frozen inference state for single-policy and multi-seed callers.
2. Test reducers with a tiny fake sampler/scorer: known winners per batch row,
   exact arithmetic means, deterministic ties, invalid scores, unsupported actions,
   N=1 equivalence, and entire-plan selection with chunk size greater than one.
3. Integrate at least an actor policy and NCLQL's direct sampler. Verify capability
   errors for providers without scorers and test Default/Mean-of-N without critics.
4. Verify collection and evaluation independently select strategies; changing
   evaluation settings does not consume training RNG or mutate policy statistics.
5. Verify fresh weights after an update are used, configured specs survive resume,
   and compile counts do not grow on repeated calls at fixed shapes/specs.
6. Include the strategies in the CPU example/config tooling and benchmark N=1
   against the old Default path, plus several larger N values. Do not claim higher
   returns merely because BoN maximizes an estimated critic score.

The desired code reduction comes from sharing inference binding and execution
across single/stacked policy paths and avoiding per-algorithm BoN implementations.
Do not create three almost-identical wrappers around every policy class.

## Verification and completion contract

Use current repository commands as the starting point:

```bash
uv run pytest tests/ -q
JAX_PLATFORMS=cpu MUJOCO_GL=egl uv run python scripts/lint_config.py
uv run python scripts/submit.py --algos SAC --envs HalfCheetah --seeds 1 --dry-run
uv build src/barl_jax --out-dir /tmp/ba-wheel
```

Run relevant tests after each meaningful change and the full required suite at
integration points. Wheel verification additionally requires a fresh installation
outside the repo; building alone is insufficient. Do not accidentally test an old
wheel already in the output directory. Select optional integration tests according
to installed dependencies and report skips honestly.

Prefer contract and failure-path tests to tests that mirror implementation:
independent configs in one process, no simulator import during discovery, delayed
optimizer moments unchanged, terminal observation/discount correctness, replay
RNG restore, cleanup after partial setup, and checkpoint publication failure.
For deterministic numerical comparisons state tolerances and control RNGs.

Before declaring the backlog complete:

- The action-executor requirement and all 18 original backlog items have implemented acceptance evidence or are explicitly
  identified as remaining; do not silently relabel unimplemented work as done.
- Documentation and generated examples use only current owning-module APIs.
- No compatibility shims, duplicate defaults, unused wrappers, or abandoned draft
  files remain. Required third-party dependency adaptations are audited separately:
  remove by upgrading/replacing the dependency or explicitly dropping the affected
  support, not by silently breaking a still-advertised environment.
- Production line/statement deltas are reported against the agreed baseline,
  separately from tests and documentation, with concrete sources of reduction.
- Full tests, config lint, launcher dry run, and clean installed-wheel checks pass;
  performance claims have measurements rather than assumptions.
- Remaining numerical changes, platform limits, and checkpoint guarantees are
  documented where users make the relevant choice.

Keep README focused on how to use the result, CLAUDE.md focused on enduring rules,
and this plan/backlog updated with actual completion evidence. Do not copy these
implementation essays into source comments or product-facing CLI output.
