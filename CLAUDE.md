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

Anything in `config/models.yaml` is sweepable as a `model.<dotted path>` axis and anything in the
run config as a `run.<key>` axis, either inline or via a `--grid` spec file (see
`config/sweep_grid.yaml`); `--dry_run` prints the expansion instead of launching it:

```bash
uv run python scripts/run_sweep.py --algos AMDP --envs HalfCheetah --seeds 3 \
    --set model.components.critic.rho=0.0 --set run.utd_ratio=0.5
uv run python scripts/run_sweep.py --grid config/sweep_grid.yaml --dry_run
```

To see what a `model:` string actually resolves to, with every value attributed to the layer that
set it:

```bash
uv run python scripts/print_config.py AMDP    # no args: every model in config/models.yaml
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

- **Actor/Critic = pure architecture** (plus whatever tunables are intrinsic to that architecture's
  own process — e.g. a diffusion actor's noise schedule, a critic's ensemble pessimism — supplied
  as required kwargs from `config/models.yaml`, no Python-level defaults). No algorithm logic
  lives here.
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
target net, `rho`/`polyak_tau`/`gamma` -- config/models.yaml-supplied, see below --
mean-minus-`rho`*std pessimistic lower bound):
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

### Tunables all live in YAML -- there is no Python-level HParam system

Every tunable value -- both architecture kwargs (`hidden_dim`, `layers`, ...) and algorithm
tunables that used to be a separate `HParam`-declared axis (`rho`, `gamma`, `tr_lambda_init`, PPO's
`clip_eps`, ...) -- is a required, no-default keyword argument supplied from YAML, across exactly
two files: `config/components.yaml` (baseline kwargs per component *class*) under
`config/models.yaml` (per *policy* slots/params/profiles). There is no Python-level fallback
anywhere in this chain; a value listed in neither file is a `KeyError`/`TypeError` at construction
time.

- **Component-level tunables** (things an `Actor`/`Critic` itself needs, e.g. `QCriticBase`'s
  `rho`/`polyak_tau`/`gamma`, `ScoreContinuousActor`'s `sigma_min`/`sigma_max`/`tau_min`/`tau_max`)
  resolve through `Policy.build_component()`/`resolve_component()` → `model_config.resolve_slot()`,
  which layers, lowest precedence first: `config/components.yaml`'s entry for the slot's `class:` <
  the slot dict in `config/models.yaml` < the active profile's override for that slot < the
  Policy's own explicit kwargs. `config/components.yaml` is the *only* defaults layer in the
  codebase; a policy slot names its `class:` and then lists only its deltas, so swapping a slot's
  class (`MeanStdEnsembleCritic` → `QRDistributionalCritic`) is a one-word edit as long as the new
  class has an entry there. Classes without one must have every kwarg spelled out in the slot.
  `uv run python scripts/print_config.py <model>` prints a model's fully resolved config with every
  value annotated by the layer it came from (and what it shadows), so no default is unattributable.
- **Per-run overrides** (`model_config.set_overrides()`) sit above both YAML files and any profile,
  below only the Policy's own explicit kwargs: a flat dotted-path map,
  `{"components.critic.rho": 1.0, "params.entropy_scale": -2.0}`, coming from a run config's
  `model_overrides:` key (which `scripts/run_sweep.py`'s `model.*` grid axes write). This is the
  layer a sweep varies, so **sweeping a tunable never means editing a config file in place**.
  `components.<slot>.class` is overridable too and re-bases the slot on the new class's
  `config/components.yaml` entry, which is what makes an architecture axis
  (`MeanStdEnsembleCritic` vs `QRDistributionalCritic`) sweepable. The active overrides are
  captured into every checkpoint (`Policy.save`) and reapplied by `Policy.load`, since they are
  part of the architecture.

`model_config.validate(name)` is run by `scripts/online_learning.py`'s `main()` before anything is
constructed, and raises rather than starting a doomed run: overrides that land on no slot/param,
kwargs no `__init__` in the resolved class's MRO accepts (every component has a `**kwargs`
catch-all, so a typo'd kwarg is otherwise silently swallowed and a mistyped sweep axis silently
becomes a no-op), and values whose YAML type contradicts the annotation (a bare `3e-4` or a stray
`true,` parses as a `str`). `scripts/print_config.py` reports the same problems non-fatally.
- **Policy-level tunables** (things a Policy reads directly off `self` in `train_step`, not part of
  any actor/critic slot -- AMDP's `tr_lambda_init`/`tr_epsilon`, PPO's `gamma`/`lam`/`clip_eps`/
  `entropy_coef`, IQL's `tau`/`beta`, FlowQL's `alpha`) live in a `params:` block alongside that
  policy's `components:` block in `config/models.yaml`, resolved via `Policy.resolve_params()` →
  `model_config.resolve_params()` (same `models.yaml` < profile-override < `_raw_kwargs` <
  explicit-kwargs precedence as `resolve_component()`, minus the `components.yaml` layer -- a
  `params:` block has no `class:`), called once early in the Policy's `__init__` and assigned by
  required dict key.

Because those two YAML files are the *only* place tunables live, a checkpoint's reconstruction
(`Policy.save`/`Policy.load`) only needs to persist which `profile` was used, not a `hparams` dict
-- both architecture and former-HParams come back from YAML at load time, so keeping those files
(or a run's own frozen snapshot of them, see below) stable is what makes a checkpoint reproducible.

`model_config.snapshot_for(name)` returns the minimal config sub-dict needed to reproduce resolving
`name` later: the `aliases:` entry if `name` is one, the resolved policy's full `models.yaml` block,
and -- under the reserved `component_defaults:` key -- the `components.yaml` entry for every class
that policy's slots name, so replay never depends on either file's current contents.
`scripts/run_sweep.py` embeds this into every per-run YAML it writes under a `models:`
key (via a small dependency-free local mirror of `snapshot_for()`, so the lightweight
subprocess-launcher script never has to import `barl` itself); `config.py`'s `load_config()` threads
that key through to `cfg["models"]`, and `scripts/online_learning.py`'s `main()` applies it via
`model_config.set_models_override()` before constructing the Policy -- so a saved sweep run stays
exactly recreatable even if `config/models.yaml` changes afterward. The same resolved-config dict
(embedded snapshot, or freshly computed via `snapshot_for()` for a non-sweep run) is logged to
wandb under `"resolved_model"` in every run's `config=` dict, alongside `resolved/*` — every
*effective* value flattened to `resolved/components.critic.rho`-style keys via
`model_config.resolved_flat()`, which is what makes runs in a sweep groupable/comparable by
hyperparameter rather than by which layer happened to define it.

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

A YAML config has a `defaults:` dict, an optional `env_defaults:` map (keyed by env name) and a
`runs:` list; each run entry is merged onto `env_defaults[<its env>]` onto `defaults` (later wins)
and `load_config()` returns the resolved list — `scripts/online_learning.py` iterates it and runs
each `{model, env, ...}` combination in series in one process. `model_overrides:` is the one key
merged key-wise across those three levels rather than replaced wholesale, since a per-env and a
per-run override normally mean to stack.

`scripts/run_sweep.py` instead fans a grid out across **separate processes** (own interpreter, own
CUDA context each), writing one throwaway per-run YAML (same `defaults`/`env_defaults` +
single-entry `runs`) per subprocess so it can reuse the exact same script/config-loading path. The
grid is a cross product over named axes — `algo`/`env`/`seed`, plus any number of
`model.<dotted path>` axes (written into that run's `model_overrides:`) and `run.<key>` axes
(written into its run entry) — declared either on the CLI or in a `--grid` spec file
(`config/sweep_grid.yaml` is a documented example), with an `exclude:` list for pruning
combinations out of the full product. `--dry_run` prints the expansion without launching anything.
Because the axes are just overrides, **no config file is ever edited to run a sweep**.

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
