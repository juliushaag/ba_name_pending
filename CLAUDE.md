# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`barl` is a research framework for online (off and on-Policy)
deep-RL experiments across MuJoCo Gym and DeepMind Control Suite continuous-control benchmarks.
It implements a range of actor-critic algorithms (SAC, PPO, IQL, DQN, FlowQL/FlowBC, AMDP, DIME)
behind one shared `Policy`/`Environment` interface so any algorithm can be pointed at any
registered environment through a single training script and YAML config.

`main.py` at the repo root is a scratch/throwaway script (loading a Minari dataset), not part of
the framework's entry points.

## Code style

Comments are almost never necessary — well-named identifiers and the surrounding code should
speak for themselves. Only add one when something is genuinely non-obvious (a hidden invariant, a
subtle numerical/shape gotcha, a workaround for a specific upstream bug), and even then, one
sentence max. Never write multi-line comment blocks or docstrings explaining what code does.

## Commands

Dependency management is via `uv` (see `pyproject.toml` / `uv.lock`); the `barl` package itself is
installed editable from `src/barl`.

```bash
uv sync                                     # install/sync the environment
```

Train one or more algorithm/env combinations in series (defined by a YAML config, see
`config/online_learning.yaml`):

```bash
uv run python scripts/online_learning.py                              # uses config/online_learning.yaml
uv run python scripts/online_learning.py --config path/to/other.yaml
```

Launch a grid of `{algo x env x seed}` runs as truly parallel OS processes (each in its own
subprocess with its own CUDA context), logs under `sweeps/<timestamp>/`:

```bash
uv run python scripts/run_sweep.py --algos SACPolicy PPOPolicy --envs HalfCheetah CheetahRun --seeds 3
uv run python scripts/run_sweep.py --algos SACPolicy --envs HalfCheetah --max_parallel 4
```

There is no unit test suite. The closest thing to "run the tests" is `--validate`: a cheap smoke
test that runs one rollout chunk + one `train_step` + one optimizer step per `{algo, env}`
combination (no eval/checkpointing/wandb logging), to catch shape/dtype/algo-env mismatches
before committing to a full run:

```bash
uv run python scripts/run_sweep.py --algos SACPolicy PPOPolicy --envs HalfCheetah CheetahRun --validate
```

`scripts/run_sweep.py --script` also accepts `offline_learning`/`offline_to_online`, but those
scripts and their configs (`config/offline_learning.yaml`, `config/offline_to_online.yaml`) are
currently deleted on this branch — only `online_learning` is live.

No linter/formatter/type-checker is configured in `pyproject.toml`; there's no CI to match.

## Architecture

### Registries: `Policy` and `Environment`

Both algorithms and environments are plain string-keyed registries (`src/barl/policy.py`,
`src/barl/env.py`), populated by decorators at import time:

```python
@register_policy
class SACPolicy(Policy): ...

@register_environment
class HalfCheetah(MuJoCoGymEnvironment): ...
```

`Policy.make(name, env)` / `Environment.make(name)` construct by string, and `src/barl/__init__.py`
is what actually triggers every registration by importing `barl.policies` / `barl.environments.*`
wildcard-style — anything not imported there won't show up in `Policy.available()` /
`Environment.available()`. A new algorithm or environment must be added to that module (or its
own `__init__.py`, for policies) to be reachable from a YAML config.

`Environment` wraps a single benchmark family: `get_env()` builds one `gym.Env` (used for eval and
video rendering), `get_vector_env(n)` builds an `N`-way vectorized env for rollout collection
(`SyncVectorEnv` by default; `use_async_vector_env = True` opts into subprocess-per-env
`AsyncVectorEnv`, worth it once MuJoCo's C-level physics step dominates over IPC overhead —
`autoreset_mode=DISABLED` is mandatory since `ReplayBuffer.rollout` resets finished sub-envs
itself). `get_dataset()` is only implemented by offline-dataset-backed environments; MuJoCo Gym
and DM Control are online-only and raise `NotImplementedError` there.

### `Policy` / `Actor` / `Critic` split

Every algorithm is one `Policy` subclass (`src/barl/policy.py`) composing an `Actor`
(`src/barl/networks/actor.py`) and, where applicable, a `Critic` (`src/barl/networks/critic.py`).
The load-bearing convention throughout the codebase:

- **Actor/Critic = pure architecture** (plus whatever HParams are intrinsic to that architecture's
  own process — e.g. a diffusion actor's noise schedule, a critic's ensemble pessimism). No
  algorithm logic lives here.
- **Policy = the whole algorithm**: Bellman targets, advantage weighting, clipped surrogates,
  entropy/trust-region duals, adjoint matching — everything algorithm-specific lives in that
  Policy's `train_step`, calling into whatever its actor/critic expose.

This is why the same `StochasticActor` backs SAC, PPO and IQL (three very different updates over
the same Gaussian-or-Categorical architecture), and the same `QCriticBase` subclasses back SAC,
FlowQL and AMDP. When extending an existing actor/critic, ask whether the new behavior is
architectural (belongs there) or algorithmic (belongs on the owning Policy) before adding it.

Concrete actors in `networks/actor.py`:
- `DeterministicActor` — flat MLP regression / BC point estimate (`DirectPolicy`).
- `StochasticActor` — Categorical (discrete) or tanh-squashed diagonal-Normal (continuous); shared
  by PPO, IQL, SAC.
- `ScoreContinuousActor` — continuous-time SDE (AMDP-style) score-matching actor (`AMDP`).
- `ScoreDiscreteActor` — fixed-step DDPM-style discrete-time diffusion actor (`DIMEPolicy`).

Score-based actors deliberately return a **raw, unsquashed** terminal state from `sample()`/
`generate_action()` — squashing it into a canonical action is an algorithm decision the owning
Policy makes on top (e.g. AMDP's `ErfSquash`), not something the actor architecture bakes in.

Concrete critics in `networks/critic.py`, all subclassing `QCriticBase` (shared Polyak-averaged
target net, `rho`/`polyak_tau`/`gamma` HParams, mean-minus-`rho`*std pessimistic lower bound):
- `MeanStdEnsembleCritic` — `n_models` independent MLP heads reduced by mean - rho*std; spread =
  epistemic uncertainty. Default for every continuous-action off-policy algorithm here.
- `MeanEnsembleCritic` / `MinEnsembleCritic` — same `n_models` MLP-head ensemble, reduced by plain
  mean or TD3-style min instead; both ignore `rho` (std is forced to 0 in `_mean_std`).
- `QRDistributionalCritic` — single MLP head predicting quantile atoms, QR-DQN pinball loss
  (Dabney et al., 2017); spread = aleatoric uncertainty.
- `C51Critic` — single MLP head predicting logits over a fixed categorical support
  (`v_min`..`v_max`, `num_atoms`), standard C51 (Bellemare, Dabney & Munos, 2017): categorical
  Bellman projection (`categorical_projection`) + cross-entropy loss.

`_mean_std()`'s reduction feeds both `value()`/`target_value()` (always the pessimistic bound) and,
via `pessimistic_score`, `score()`/the action-gradient used for actor updates — AMDP wants the
pessimistic gradient, FlowQL/QAM want the plain mean.

### Action scaling convention

Every continuous actor is trained to output a canonical **[-1, 1] per dimension** action,
independent of the env's real bounds (which aren't always ±1, e.g. Humanoid is ±0.4).
`Policy.scale_action()`/`unscale_action()` do the one-place affine remap to/from real env bounds.
Only two call sites ever need it: `ReplayBuffer.rollout()` (immediately before `env.step()`) and
`Policy.evaluate()`. Everything a Policy stores (trajectories) or trains against is **already**
canonical — a Policy re-invoking its own actor mid-`train_step` (e.g. for a Bellman bootstrap
action) can feed the result straight to its critic with no further conversion.

### HParam machinery

`HParam`/`HParamMixin` (`src/barl/policy.py`) is the shared hyperparameter declaration system
mixed into `Policy`, `Actor` and `Critic`. Declare an HParam as a class attribute:

```python
rho = HParam(2.0, "Pessimism weight on Q std", range=(0.0, 5.0))
```

`apply_hparams(kwargs)` resolves+validates+`setattr`s every HParam this class declares and is
called in every `__init__`; each class's full unfiltered `kwargs` is forwarded to every component
it constructs (Policy → Actor, Policy → Critic), and each recipient's `resolve_hparams` silently
ignores names it doesn't declare — so a Policy never needs to know which of its sub-components
owns a given hyperparameter name. `Policy.hparams` merges the Policy's own HParams with its
actor's/critic's into one flat dict for checkpointing and run-config logging.
`describe_hparams()`/`sample_hparams()` are introspection/random-search helpers off the same
descriptors.

### Rollout / replay / dataset pipeline

`ReplayBuffer.rollout()` (`src/barl/rollout.py`) drives either a plain `gym.Env` or a `VectorEnv`
for `n_steps`, calling `policy.inference_step()` once per step over the whole batch and
accumulating one `Trajectory` chunk per env slot. Chunks longer than `traj_len` are cut and a new
chunk continues the same episode; `save_trajectory()` evicts random whole chunks (never splits
one) to stay under `max_transitions` — this is why on-policy algorithms need `buffer.buffer = []`
after each rollout+train pass rather than relying on eviction (see below).

`SingleStepDataset` (`src/barl/dataset.py`) flattens a list of `Trajectory` chunks into a flat,
shuffleable single-step `torch.utils.data.Dataset` (`obs`/`next_obs`/`act`/`rew`/`term`, plus
whatever's in each trajectory's `extras`). `compute_gae()` fills each `Trajectory.extras` with
GAE(λ) advantages/returns **in place, before** flattening — it needs sequential in-order
rewards/values, which a shuffled dataset no longer has.

`Policy.on_policy = True` (set by `PPOPolicy`) is how `scripts/online_learning.py` decides to run
`compute_gae()` before building the dataset each iteration, and to clear
`buffer.buffer = []` after training instead of letting old-policy trajectories linger the way
every off-policy critic here relies on.

### Training loop (`scripts/online_learning.py`)

Outer loop, per iteration: `buffer.rollout()` collects `rollout_steps` env steps →
(on-policy only) `compute_gae()` → build a `SingleStepDataset`/`DataLoader` → run
`epochs_per_rollout` passes of `train_epoch()` over it → (on-policy only) clear the buffer.
`train_epoch()` calls `model.train_step(batch)` which returns `(losses, stats)`; every loss key
must match a key in `model.create_optimizer()` — each is zero-grad'd, backward'd
(`retain_graph=True`, since e.g. an actor loss reads the critic's forward pass), grad-clipped, and
stepped independently, so no optimizer step invalidates another loss's graph mid-loop. Periodic
eval (`evaluate_and_log`) checkpoints `latest.pt` every eval and `best.pt` on a new best avg reward,
under `checkpoints/<model>/<env>/<timestamp>/`.

`Logger` (`src/barl/logger.py`) wraps `wandb` (source of truth for history/plots) with a live
`rich` terminal panel driven off the same `log()`/`increment()` calls — nothing needs logging
twice. `wandb.define_metric(step_metric=...)` alignment means step counters (`env_steps`,
`grad_steps`) must be logged in the *same* `wandb.log()` call as the data they index, not a
separate `increment()` call before/after.

### Config resolution (`src/barl/config.py`)

A YAML config has a `defaults:` dict and a `runs:` list; each run entry is merged onto `defaults`
(run-specific keys win) and `load_config()` returns the resolved list — `scripts/online_learning.py`
iterates it and runs each `{model, env, ...}` combination in series in one process.
`scripts/run_sweep.py` instead fans a `{algo x env x seed}` grid out across **separate processes**
(own interpreter, own CUDA context each), writing one throwaway per-run YAML (same `defaults` +
single-entry `runs`) per subprocess so it can reuse the exact same script/config-loading path.

### Registered algorithms (`src/barl/policies/`)

`DirectPolicy` (BC), `FlowBCPolicy`/`FlowQLPolicy` (flow-matching BC / flow-matching
+ Q), `AMDP` (direct-score regression against a critic's action-gradient, entropy +
trust-region duals — see the extensive docstring in `amdp.py`), `IQLPolicy`, `PPOPolicy`,
`SACPolicy`, `DIMEPolicy`. All discrete-action support was removed with `QPolicy` (DQN) --
every remaining Policy is continuous-action only. `on_policy = True` only on `PPOPolicy` --
everything else is off-policy and relies on the replay buffer retaining older trajectories
across iterations.

### Registered environments (`src/barl/environments/`)

`mujoco_gym.py` wraps gymnasium's built-in MuJoCo `-v5` envs (HalfCheetah, Hopper, Walker2d, Ant,
Humanoid, HumanoidStandup, InvertedPendulum, InvertedDoublePendulum, Reacher, Swimmer, Pusher).
`dm_control.py` wraps DeepMind Control Suite domain/task pairs via `shimmy`, flattening each dict
observation into a single flat `Box` (every `Policy`/`MLP` here assumes that shape). Both families
are online-only (`get_dataset()` raises) and set `use_async_vector_env = True`.

`MUJOCO_GL=egl` is set in `scripts/online_learning.py` **before** the first (transitive) `import
mujoco` — mujoco picks its GL backend at import time, so this must happen before any
`barl`/`dm_control` import or headless `env.render()` calls hard-crash looking for an X11 display.

### EMA vs. Shadow (`src/barl/networks/ema.py`)

Two different "frozen reference copy" wrappers, not interchangeable: `EMA` exponentially blends
toward the source every `step()` call (used for Q target nets — call once per train step) and is
kept `vmap`-safe (no `.item()`/Python-scalar extraction in its update, no `.data` in-place ops).
`Shadow` hard-copies the source once per epoch via `on_epoch_start()` (used for trust-region
references, e.g. AMDP/DIME regularizing their live diffusion net against last epoch's snapshot) —
named after the `Policy` lifecycle hook that's meant to drive it.
