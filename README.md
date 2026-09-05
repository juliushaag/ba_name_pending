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
CPU test suite, and `scripts/wheel_smoke.py`, all against the standalone
`src/barl_jax` install rather than the root workspace's CUDA/simulator
dependencies.

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

Training writes `checkpoints/resume/` after evaluation iterations and on normal
completion. To continue an interrupted run, use its original configuration with:

```yaml
checkpoint: /absolute/path/to/checkpoints/resume
checkpoint_mode: resume
```

Resume restores all seeds, optimizer moments/schedules, replay contents and RNGs,
normalization state, counters and best scores. The original training parameters,
seed count and replay capacity must match. Output paths and logging/evaluation
cadences may change. Saved model snapshots take precedence over current defaults.

Environment episodes restart on resume. Partial action chunks are discarded;
completed replay transitions remain intact. This is training-state continuation,
not bit-identical simulator trajectory continuation. A failed iteration retains
the last completed resume checkpoint; `final` policy weights may be newer.
Full replay checkpoints can be large, so account for their storage and I/O cost.

Use `with setup(config, num_seeds) as run:` when managing a run directly. The
public `train` and `validate` functions close owned resources on success or
failure, including failures during initialization and warmup.

See [the prioritized QOL backlog](docs/QOL.md) for follow-up improvements.
