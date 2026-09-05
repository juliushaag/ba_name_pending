# barl_jax

A JAX/Flax-NNX research framework for online off-policy actor-critic learning.
SAC, AMDP, DIME, SDAC and NCLQL share configuration, collection, replay,
evaluation and execution infrastructure.

## Development

```bash
uv sync
uv run pytest tests/ -q
JAX_PLATFORMS=cpu uv run python scripts/lint_config.py
```

The repository environment includes the simulator families and CUDA dependencies
used for experiments. The library can also be installed independently from
`src/barl_jax`, with optional `logging`, `mujoco`, `dm-control`, `mani-skill` and
`humanoid-bench` extras. HumanoidBench experiments use the local submodule and
compatibility constraints declared in the root `pyproject.toml`.

```bash
uv build src/barl_jax
uv run python scripts/wheel_smoke.py   # builds + installs that wheel into a throwaway
                                        # venv and smoke-tests it outside the repo
```

`uv run ruff check .` lints the first-party tree (`third_party/`, `wandb/`, `sweeps/`
and `checkpoints/` are excluded); the rule set is deliberately small for now
(syntax errors, pyflakes, import sorting -- see `[tool.ruff]` in `pyproject.toml`).

`scripts/fastcheck.py` is the fast edit/check loop for config, registry and
launcher changes -- config lint plus the cheap boundary tests, without paying
for the full training-loop suite:

```bash
JAX_PLATFORMS=cpu uv run python scripts/fastcheck.py
```

`scripts/doctor.py` reports installed versions, optional simulator availability,
and submodule readiness; `--probes` additionally runs the JAX backend and EGL
render checks (each in its own subprocess, so a native crash is a reported
failure rather than a dead process), and `--json` gives machine-readable output:

```bash
uv run python scripts/doctor.py --probes
```

CI (`.github/workflows/ci.yml`) runs `ruff check`, `scripts/lint_config.py`, the
CPU test suite, `scripts/wheel_smoke.py`, and `examples/cpu_quickstart.py`, all
against the standalone `src/barl_jax` install rather than the root workspace's
CUDA/simulator dependencies.

`examples/cpu_quickstart.py` is a complete round trip on CPU only, no
accelerator, W&B account, simulator download, or render dependency: a tiny
environment (gymnasium's own `Pendulum-v1`, selected via the external
`module:Class` path rather than the central registry), a real policy,
collector, replay, training steps, a policy checkpoint save/load, and a
training resume that verifies the restored optimizer/policy state and replay
contents are bit-identical to the original run's. ~15-20s end to end on a
laptop CPU (first-compile cost is reported separately from warm steps in its
own output; recompute before repeating that number elsewhere).

```bash
BARL_ALLOW_CPU=1 uv run python -m examples.cpu_quickstart
```

## Running experiments

`scripts/submit.py` expands configurations before handing commands to the local,
tmux or Slurm executor. Seeds are vmapped within each run.

```bash
uv run python scripts/submit.py --algos SAC NCLQL --envs HalfCheetah --seeds 3 --dry-run
uv run python scripts/submit.py --algos SAC --envs HalfCheetah --executor local --validate
uv run python scripts/print_config.py SAC
uv run python scripts/online_learning.py --config sweeps/<submission>/<run>/config.yaml
```

The training CLI requires an accelerator. Set `BARL_ALLOW_CPU=1` for intentional
CPU smoke tests. Rendering uses EGL by default in the training CLI.

Every run config is preflight-checked (`barl_jax.config.run.validate_run`, plus
`barl_jax.config.tree.validate` for the model half) before an environment, a
policy, or a JAX device buffer is allocated -- an invalid `total_env_steps`,
`checkpoint_mode`, `lr_decay`, etc. fails immediately with the offending key,
value and violated rule, not partway through setup.

`logger:` selects the reporting sink -- `wandb` (default), `local` (the same
live terminal panel, with metrics/summary written as strict JSON under the
run directory instead of shipped to wandb), or `none` (step bookkeeping only,
for a training smoke test with no reporting backend at all). Only `wandb`
imports the `wandb` package. A run's `objective.json` (`trainer._write_objective`,
read by `scripts/hpo.py --report`) is strict JSON too: a seed with no
completed evaluation is `null` there, not a non-finite float, with
`objective_reason` saying why.

Defaults live in `src/barl_jax/defaults/` and ship inside the library wheel.
Set `BARL_CONFIG_DIR` before importing config tools, or pass an explicit root to
`ConfigTree`/`load_config`, to use another configuration directory.

### Action selection at inference time

`collection_mode`/`collection_candidates` and `eval_mode`/`eval_candidates`
choose, independently, how an action is selected at each inference call --
policy-independent (`barl_jax/execution.py` never branches on SAC/AMDP/DIME/
SDAC/NCLQL):

- `default` (the default; `num_candidates` must be 1) -- one unchanged
  inference call, bit-identical to not passing an execution spec at all.
- `mean` -- samples `num_candidates` action plans and averages them
  elementwise. Continuous (box) action spaces only; can land between modes
  of a multimodal policy, so treat it as a research strategy, not an
  automatic improvement.
- `bon` (Best-of-N) -- samples `num_candidates` plans and keeps the
  highest-scoring one under the policy's own critic (`Policy.score`,
  `Policy.supports_bon` -- SAC/AMDP/DIME/SDAC via `RLPolicy`; NCLQL has none
  and is rejected at preflight with an actionable message, same as any
  algorithm without a scorer).

Both are validated before any resource is allocated: an unknown mode or bad
candidate count in `barl_jax.config.run.validate_run`, and the
capability/action-space checks (`execution.validate_execution_spec`) as soon
as `trainer.setup()` has a constructed reference policy. `collection_mode`/
`collection_candidates` are part of the training config (changing them is a
behavior-policy change, so resume rejects a mismatch); `eval_mode`/
`eval_candidates` may change across a resume.

### Evaluation RNG

Each `evaluate()` call owns an `eval_seed` (`barl_jax.seeding.derive_seed(base_seed,
"eval", eval_count)`, logged at eval start) independent of every training RNG
stream (policy, collector, replay) -- evaluation never perturbs a training
trajectory, and running the same config from the same seed reproduces the same
eval episodes at the same iteration. `run_episodes()` derives both the
policy's action key and every wave's `env.reset(seed=...)` from that one
seed; a later wave (or a mid-wave slot refill) never replays an earlier one's
episode. `eval_count` is not restored on resume, consistent with resume's
existing "environment episodes restart" semantics.

## Extending the framework

- `components.py` defines actors, critics, configuration declarations and NNX
  state utilities. Low-level components do not import the policy runtime.
- `policies/` composes components and defines algorithm behavior. `objectives.py`
  holds TD and noise-conditioned TD objectives over critic prediction/regression
  primitives; `SLOT_OBJECTIVES` validates compatibility before construction.
- `optimization.py` assembles parameter ownership, update cadence and schedules.
- `collection.py` owns environment/chunk state and emits transitions into an
  injected sink. `rollout.py` owns replay storage and independent sampling RNGs.
- `evaluation.py` evaluates action providers; `multiseed.py` stacks independent
  policies and optimizers. `trainer.py` coordinates their lifecycle.
- `checkpoints.py` restores policy architectures from saved metadata;
  `training_checkpoint.py` handles optimizer/replay/counter restoration.

Importing `barl_jax` or its config toolkit does not initialize JAX or simulators.
Registry lookups load built-ins on demand. Environment-family imports expose
metadata; simulator dependencies load when `get_env()` runs. External policy,
component and environment classes can be selected with `module:Class` paths.
Async environments pickle their bound factory, preserving instance settings;
the factory and its class must be importable in the worker process.

Three ways to name a Policy/Environment/component implementation, tried in
order: a built-in name; an installed plugin -- an entry point under
`barl_jax.policies`/`barl_jax.environments`/`barl_jax.components` in another
distribution's `pyproject.toml` (`[project.entry-points."barl_jax.policies"]`),
discovered once per process and loaded only when actually selected; or
`module:Class`, unaffected by what plugins are installed. A name shared by a
built-in and a plugin, or by two plugins, is a deterministic error at lookup
time, not a silent precedence rule.

`scripts/scaffold.py env <ClassName> --gym-id <id>` generates a new
`Environment` subclass (selected via the external `module:Class` path, no
registry edit) and smoke-tests it in a temp copy. `scripts/scaffold.py
algorithm <Name> --actor <ActorClass> --critic <CriticClass>` generates a
`config/models/<name>.yaml`-shaped file combining existing actor/critic
classes on the shared `MaxEntRLPolicy` template (how SAC/AMDP/DIME/SDAC are
each already just a model YAML entry) and smoke-tests it against the Policy
contract (`train_step` losses match `create_optimizer()`'s keys and are
finite; `inference_step`'s shape is right) in a disposable copy of the
packaged defaults. Both refuse to overwrite an existing file.

`scripts/benchmark.py` measures fresh-process import latency, async
vector-env worker RSS, collection throughput, and compiled gradient
throughput, reporting device/library versions and every shape/count needed
to judge whether a number is comparable to another run. No hard thresholds --
compare two of its own `--json` outputs on the same machine.

For independent in-process experiments, pass configuration explicitly:

```python
from barl_jax.config.model import ModelSpec
from barl_jax.config.tree import ConfigTree
from barl_jax.policy import Policy

config = ConfigTree(overrides={"params.policy_delay": 2})
spec = ModelSpec.resolve("SAC", config)
policy = Policy.make("SAC", env, env_stats=env_stats, spec=spec)
```

`ModelSpec` freezes resolved parameters, component choices and provenance.
Policies retain their own resolved specification; later config changes cannot affect
their components. Import contracts, checkpoint functions and inference utilities from
their owning modules. Older checkpoint formats are rejected explicitly.

Observation normalization now updates running moments from each training batch,
before the objective runs. Inference reads those moments without updating them.
This activates the previously ineffective `normalize_obs: true` setting and
changes new training results. Set `model.params.normalize_obs=false` through the
usual override mechanism to retain unnormalized inputs.

## Checkpoints

Policy checkpoints (`latest`, `best`, `final`) contain policy state and a
versioned architecture snapshot. They are suitable for inference or initializing
a new run. With `checkpoint_mode: weights` (the default), optimizer/replay and
training counters start fresh; saved normalization and target weights are kept.
Each requested seed receives an independent policy RNG.

Training writes `checkpoints/resume/` on its own cadence (`resume_every` env
steps, independent of `eval_every` -- the replay dump it carries, not the
eval, is what makes a save expensive) and again on normal completion. Each
save is an immutable, numbered generation (`resume/gen-000003/`, ...)
published via `resume/pointer.json` only once fully written; `resume_keep`
(default 2) is how many generations are then kept -- older ones are deleted
only after the new one is committed, so an interrupted or out-of-disk save
never leaves the previous generation as collateral damage. To continue an
interrupted run, use its original configuration with:

```yaml
checkpoint: /absolute/path/to/checkpoints/resume
checkpoint_mode: resume
```

Resume restores all seeds, optimizer moments/schedules, replay contents and RNGs,
normalization state, counters and best scores. The original training parameters,
seed count and replay capacity must match. Output paths, logging/evaluation
cadences, `logger:`, and `resume_every`/`resume_keep` may change. Saved model
snapshots take precedence over current defaults.

Environment episodes restart on resume. Partial action chunks are discarded;
completed replay transitions remain intact. This is training-state continuation,
not bit-identical simulator trajectory continuation. A failed iteration retains
the last completed resume generation; `final` policy weights may be newer.
Full replay checkpoints can be large, so account for their storage and I/O cost --
`save_training` refuses to start a new generation when free disk looks
insufficient to hold it alongside the previous one.

`scripts/checkpoint_info.py <path>` inspects a policy checkpoint or a resume
root's metadata -- kind, format version, saved implementation, array/shape
counts, counters, config hash, resume limitations -- without constructing a
policy or an environment. `scripts/config_diff.py <model> --against <path>`
diffs a model's currently-resolved config against a frozen snapshot (from
either kind of checkpoint, or a bare snapshot JSON file), labeling each
changed key as a declaration/default change, an explicit override, or a
tier/profile choice.

Use `with setup(config, num_seeds) as run:` when managing a run directly. The
public `train` and `validate` functions close owned resources on success or
failure, including failures during initialization and warmup.

See [the prioritized QOL backlog](docs/QOL.md) for follow-up improvements.
