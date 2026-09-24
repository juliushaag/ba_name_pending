# barl_jax

A JAX/Flax-NNX research framework for online off-policy actor-critic learning.
SAC, AMDP, DIME, SDAC and NCLQL share configuration, collection, replay,
evaluation and execution infrastructure.

## Development

```bash
uv sync
uv run pytest tests/ -q
```

The repository environment includes the simulator families and CUDA dependencies
used for experiments. The library can also be installed independently from
`src/barl_jax`, with optional `logging`, `mujoco`, `dm-control`, `mani-skill` and
`humanoid-bench` extras. HumanoidBench experiments use the local submodule and
compatibility constraints declared in the root `pyproject.toml`.

```bash
uv build src/barl_jax
```

`uv run ruff check .` lints the first-party tree (`third_party/`, `wandb/`, `sweeps/`
and `checkpoints/` are excluded); the rule set is deliberately small for now
(syntax errors, pyflakes, import sorting -- see `[tool.ruff]` in `pyproject.toml`).

The command-line tools are limited to four scripts:

- `scripts/submit.py`: build a grid (or replay a saved config) and hand each run off
  to an executor -- local, tmux, or Slurm. It never trains or evaluates anything
  itself.
- `scripts/train.py`: run one experiment config's entries -- the actual training
  loop, invoked by submit.py's executors, and directly runnable to replay one
  saved run.
- `scripts/evaluate.py`: evaluate a checkpoint, run directory, or sweep under any
  of the action-selection modes below -- same role as train.py for evaluation,
  and likewise invoked by submit.py's executors for a batched `--from-sweep`/
  `--eval-config` submission.
- `scripts/debug.py`: report package/backend diagnostics and resolved model settings;
  `--json` produces output suitable for sharing with an LLM.

```bash
uv run python scripts/debug.py --model SAC --json
uv run python scripts/debug.py --probes
uv run python scripts/evaluate.py sweeps/<submission> --mode default mean --json
```

CI runs Ruff, the CPU test suite, an installed-wheel import check, and the CPU
example against the standalone library. Slurm/tmux executor support lives in
`barl_jax.launch`; it is an implementation detail of `submit.py`, which shells
out to train.py/evaluate.py exactly as a human running one of them directly
would.

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
tmux or Slurm executor. Seeds are vmapped within each run. Grid settings use `run.<key>` axes; the
obsolete grid `base:` field is rejected. `--config` replays a saved experiment
in the foreground and accepts only `--seeds` as an override.

```bash
uv run python scripts/submit.py --algos SAC NCLQL --envs HalfCheetah --seeds 3 --dry-run
uv run python scripts/submit.py --algos SAC --envs HalfCheetah --executor local --validate
uv run python scripts/debug.py --model SAC --json
uv run python scripts/submit.py --config sweeps/<submission>/<run>/config.yaml
```

### Flow-matching critic (floq)

The optional `FloqCritic` implements the scalar flow-matching critic from [arXiv:2509.06863](https://arxiv.org/abs/2509.06863). Select it for SAC, AMDP, DIME, or SDAC with a model override; the existing actor, replay, and policy interfaces remain unchanged. The default uses 8 Euler flow steps, 8 random target-noise samples, a 51-edge HL-Gauss (50 histogram features) interpolant encoding with `sigma=16.0`, and a 64-dimensional Fourier time encoding. See [docs/floq.md](docs/floq.md) for equations, adaptation details, defaults, and validation requirements.

```bash
uv run python scripts/submit.py --algos SAC --envs HalfCheetah --seeds 3 --set model.components.critic.class=FloqCritic
uv run python scripts/submit.py --algos AMDP --envs HalfCheetah --seeds 3 --set model.components.critic.class=FloqCritic --set model.components.critic.flow_steps=4
uv run python scripts/debug.py --model SAC --json --set components.critic.class=FloqCritic
```

The flow critic differentiates through its Euler chain during SAC and AMDP actor updates, so actor cost grows with `flow_steps`. Compare 4, 8, and 16 at matched seeds and report wall-clock cost; larger flow budgets can have diminishing returns.

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
imports the `wandb` package. A run's `objective.json` (`trainer._write_objective`) is strict JSON too: a seed with no
completed evaluation is `null` there, not a non-finite float, with
`objective_reason` saying why.

Algorithm defaults live in typed dataclasses beside their implementations
(`StochasticActorConfig`, `C51CriticConfig`, ... with `config_type`, the
single source of truth for that component -- `config/schemas.py` stays
generic model/components scaffolding and never redeclares them); policy YAML
files are no longer required. MLP settings are grouped under one nested `mlp`
dataclass on actors and critics, read back as
`self.config.mlp.activation`/`self.config.mlp.norm`/... There is no capacity-tier
selector; set width/depth directly. For example, change an activation with
`--set model.components.actor.mlp.activation=silu`, or widen a critic with
`--set model.components.critic.latent_dim=1024`.

### Action selection at inference time

`RunConfig.collection_executor`/`eval_executor` each select, independently,
an `ActionExecutor` (`barl_jax/executors/`) that sits between a policy's
candidate generation and the env, at every inference call -- policy-
independent (no executor branches on SAC/AMDP/DIME/SDAC/NCLQL). Each is a
registered component (`class:` in YAML, `get_executor()` at runtime) with
its own typed config declaring only the fields it uses:

- `DirectExecutor` (the default) -- one unchanged inference call, bit-
  identical to not passing an executor at all; its config has no
  `num_candidates` field at all, so it cannot be pointed at one.
- `MeanExecutor` (`num_candidates`) -- samples `num_candidates` action plans
  and averages them elementwise. Continuous (box) action spaces only; can
  land between modes of a multimodal policy, so treat it as a research
  strategy, not an automatic improvement.
- `MedianExecutor` (`num_candidates`) -- like `MeanExecutor`, but the
  elementwise median: unmoved by a single wild candidate the way an average
  would be. Continuous action spaces only.
- `AxisShuffleExecutor` (`num_candidates`) -- samples `num_candidates` plans
  and, independently for every `(batch, chunk, act_dim)` entry, takes the
  value from a randomly chosen candidate -- a per-axis recombination of all
  N, not any one candidate's action. Needs no critic; works for discrete
  action spaces too (it only ever copies a real candidate's value, never
  blends one).
- `BestOfNExecutor` (`num_candidates`) -- samples `num_candidates` plans and
  keeps the highest-scoring one under the policy's own critic (`Policy.score`,
  `Policy.supports_bon` -- SAC/AMDP/DIME/SDAC via `RLPolicy`; NCLQL has none
  and is rejected at preflight with an actionable message, same as any
  algorithm without a scorer).
- `TopKExecutor` (`num_candidates`, `top_k`) -- BoN generalized: averages the
  `top_k` highest-scoring candidates instead of hard-selecting one, softening
  BoN's failure mode of exploiting a miscalibrated critic's noise on a single
  winner. `top_k=1` behaves like `BestOfNExecutor`; `top_k=num_candidates`
  behaves like `MeanExecutor`. Needs a scorer; blends elementwise
  (continuous-only) whenever `top_k > 1`.
- `PerturbExecutor` (`num_candidates`, `sigma`) -- one base action plus
  `num_candidates - 1` local Gaussian perturbations of it (std `sigma`,
  clipped back to a valid action), scored and BoN-selected. Since the base
  action is always in the pool, the winner can never score worse than not
  perturbing at all -- a local search around the policy's own proposal,
  handy when independent resampling (`BestOfNExecutor`) barely moves the
  action (e.g. a near-deterministic policy late in training). Needs a scorer.

All are validated before any resource is allocated: an unknown `class` or a
bad `num_candidates`/`top_k`/`sigma` value in `barl_jax.config.run.validate_run`
(each executor config's own `__post_init__`/unknown-field rejection), and the
capability/action-space checks (`ActionExecutor.validate`) as soon as
`trainer.setup()` has a constructed reference policy. `collection_executor` is
part of the training config (changing it is a behavior-policy change, so
resume rejects a mismatch); `eval_executor` may change across a resume.

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
- `policies/` composes components and defines algorithm behavior. Critic losses
  live in `critics/`: `QCritic` owns TD, the target net, ensemble reduction and
  noise-conditioned TD; each value head (`DirectCritic`, `HLGaussCritic`,
  `C51Critic`) supplies only its representation-specific target and regression.
  `SLOT_OBJECTIVES` validates compatibility before construction.
- `optimization.py` assembles parameter ownership, update cadence and schedules.
- `collection.py` owns environment/chunk state and emits transitions into an
  injected sink. `replay.py` owns replay storage and independent sampling RNGs.
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

For independent in-process experiments, pass configuration explicitly:

```python
from barl_jax.config.run import RunConfig
from barl_jax.policy import Policy

run = RunConfig.from_dict({
  "model": "SAC", "env": "HalfCheetah",
  "model_overrides": {"params": {"policy_delay": 2}},
})
policy = Policy.make("SAC", env, run.model_config, env_stats=env_stats)
```

`RunConfig.model_config` is the complete typed model (policy params and component
configs). Policies retain it as `policy.config`; it is frozen, so later config
changes cannot affect their components. `chunk_size` is a policy param
(`model.params.chunk_size`), not a run key. Import contracts, checkpoint functions and inference utilities from
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

Use `with setup(config, num_seeds) as run:` when managing a run directly. The
public `train` and `validate` functions close owned resources on success or
failure, including failures during initialization and warmup.
