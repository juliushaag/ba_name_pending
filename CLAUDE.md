# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`barl_jax` is a research framework for online off-policy deep-RL experiments across MuJoCo Gym
and DeepMind Control Suite continuous-control benchmarks. It implements a range of actor-critic
algorithms (`SAC`, `AMDP`, `DIME`, `SDAC`, `NCLQL`) behind one shared `Policy`/`Environment`
interface so any algorithm can be pointed at any registered environment through a single training
script and YAML config.

**JAX only.** The framework is jax/flax-nnx/optax; there is no torch dependency, and the torch
implementation this was ported from (`src/barl`) is gone. Comments that say "the torch version
did X" are kept where they explain *why* the current code looks the way it does, not because that
code is still around.

> **Stale below this point.** Everything from "### Registries" down still describes the torch
> implementation — its file paths (`src/barl/...`), its `DataLoader`/`retain_graph` training loop,
> and policies that are not registered here (PPO, IQL, DQN, FlowQL/FlowBC, DirectPolicy). Read it
> as background on the design, not as a description of this code.

## Code style

Comments are almost never necessary — well-named identifiers and the surrounding code should
speak for themselves. Only add one when something is genuinely non-obvious (a hidden invariant, a
subtle numerical/shape gotcha, a workaround for a specific upstream bug), and even then, one
sentence max. Never write multi-line comment blocks or docstrings explaining what code does.

## Commands

Dependency management is via `uv` (see `pyproject.toml` / `uv.lock`); the `barl_jax` package
itself is installed editable from `src/barl_jax`.

```bash
uv sync                                     # install/sync the environment
```

Train one or more algorithm/env combinations in series (defined by a YAML config, see
`config/online_learning.yaml`):

```bash
uv run python scripts/online_learning.py                              # uses config/online_learning.yaml
uv run python scripts/online_learning.py --config path/to/other.yaml
```

Launch a grid of `{algo x env}` runs as separate OS processes (each with its own CUDA context),
one sequential worker per GPU, one directory per run under `sweeps/<timestamp>/`:

```bash
uv run python scripts/run_sweep.py --algos SAC NCLQL --envs HalfCheetah CheetahRun --seeds 3
```

Every GPU `nvidia-smi` reports gets one worker, which pins its subprocess to that GPU with
`CUDA_VISIBLE_DEVICES` and only takes the next combo once its own finishes — so a run never shares
a GPU. `--seeds N` is *not* a process axis: those N seeds are vmapped inside a single run.
`--sweep_dir` relocates the output tree (the Slurm jobs point it at node-local scratch).

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
uv run python scripts/run_sweep.py --algos SAC NCLQL --envs HalfCheetah CheetahRun --validate
```

Submit an `{algo x env}` matrix to Slurm as one job array (see the Slurm section below);
`--debug` puts it on the dev queue with a 30-minute cap, one task at a time:

```bash
uv run python scripts/submit.py --algos NCLQL SDAC --envs WalkerRun --seeds 5
uv run python scripts/submit.py --algos NCLQL --envs WalkerRun --debug --dry-run
```

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

`get_env_stats(env)` builds the run's `EnvStats` (`src/barl/env_stats.py`) — a plain dict of
env-derived facts handed to every component, so adding a key costs nothing. Static entries come
from the spaces and spec (`obs_dim`, `act_low`/`act_high`, `max_episode_steps`, ...) merged with
the class attribute `Environment.stats`, itself merged down the MRO so a family base can declare a
default and one subclass override a single key. That is where `return_min`/`return_max` live —
`C51Critic`'s fixed categorical support defaults from them, since no gym space reports a return
range. A `live` sub-dict is refreshed from each rollout's stats and, because every component holds
a reference to the same object, one update reaches all of them. `EnvStats` is *not* part of the
models.yaml layering: it is a function of the environment, not a tunable, so a component that wants
one of its values tunable takes a normal kwarg that falls back to the `EnvStats` entry.

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
the same squashed-Gaussian architecture), and the same `QCriticBase` subclasses back SAC,
FlowQL and AMDP. When extending an existing actor/critic, ask whether the new behavior is
architectural (belongs there) or algorithmic (belongs on the owning Policy) before adding it.

Everything `build_component()` constructs inherits the `Component` mixin (`policy.py`), which
supplies four things: the run's `EnvStats` (**first positional argument** of every component —
`self.obs_dim`/`self.action_dim` come from it, so dims are never passed separately), the slot name,
a slot-namespaced `metrics` sink, and the `before_step`/`after_step`/`on_epoch_start`/
`on_epoch_end` lifecycle. The Policy registers each slot in `self.components` and fans all four
hooks out over it, which is why a critic's Polyak update lives in `QCriticBase.before_step()`
rather than being hand-driven from each `train_step`.

Concrete actors in `networks/actor.py`:
- `DeterministicActor` — flat MLP regression / BC point estimate (`DirectPolicy`).
- `StochasticActor` — tanh-squashed diagonal-Normal; shared by PPO, IQL, SAC.
- `ScoreContinuousActor` — continuous-time SDE (AMDP-style) score-matching actor (`AMDP`).
- `DiscreteDiffusionActorBase` — abstract fixed-step DDPM-style skeleton; concrete subclasses live
  next to their algorithm (`OUDiffusionActor` in `policies/dime.py`, `VPDiffusionActor` in
  `policies/sdac.py`).

Score-based actors deliberately return a **raw, unsquashed** terminal state from `sample()`/
`generate_action()` — squashing it into a canonical action is an algorithm decision the owning
Policy makes on top (e.g. AMDP's `ErfSquash`), not something the actor architecture bakes in.

Concrete critics in `networks/critic.py`, all subclassing `QCriticBase` (shared Polyak-averaged
target net: `polyak_tau`/`polyak_tau_init`/`target_warmup_steps`/`gamma`, all
`config/components.yaml`-supplied, see below) via one of two head-building bases — `MLPCriticBase`
(one MLP over `[s, a]`, `output_dim` wide) or `EnsembleCriticBase` (`n_models` vmapped MLP heads).
Both own the head's shape kwargs (`hidden_dim`/`layers`/`activation`/`layernorm`/`residual`), so a
concrete critic declares only what its own representation needs. Only a critic that actually
reduces with a spread takes a coefficient for it, and the two spreads are named apart because they
are different quantities:
- `MeanStdEnsembleCritic` — `n_models` independent MLP heads reduced by mean - `rho`*std; spread =
  epistemic uncertainty. Default for every continuous-action off-policy algorithm here.
- `MeanEnsembleCritic` / `MinEnsembleCritic` — same `n_models` MLP-head ensemble, reduced by plain
  mean or TD3-style min instead; neither takes a coefficient (a min has no scale to tune).
- `QRDistributionalCritic` — single MLP head predicting quantile atoms, QR-DQN pinball loss
  (Dabney et al., 2017); spread = aleatoric, so its coefficient is `risk_aversion` (a
  mean-variance risk measure over the return), not `rho`.
- `C51Critic` — single MLP head predicting logits over a fixed categorical support
  (`v_min`..`v_max`, `num_atoms`), standard C51 (Bellemare, Dabney & Munos, 2017): categorical
  Bellman projection (`categorical_projection`) + cross-entropy loss; `risk_aversion` likewise.

A critic exposes two pricing accessors, both meaningful for every representation:
`expectation(s, a, target=False)` is E[Z(s,a)] and `value(s, a, target=False)` is the price a
Policy improves against — the same thing unless the class reduces with a spread. A Policy takes
`torch.func.grad` of whichever it means: `value()` for a pessimistic improvement direction (AMDP),
`expectation()` where the gradient field *is* the policy and a bound would change the sampled
distribution (NC-LQL's Langevin chain).

Regression is three primitives, with the **Policy** owning the target and the critic owning only
its representation (scalar, quantile atoms, categorical probabilities):

```python
with torch.no_grad():
  boot   = critic.bootstrap(next_obs, next_act) + entropy_bonus   # target net, own representation
  target = critic.bellman_target(reward, term, boot)              # transported one step
critic.loss(obs, action, target, losses)                          # MSE / pinball / cross-entropy
```

`bootstrap()` returns a `Bootstrap` — prediction plus a per-sample value shift — so
`+ entropy_bonus` means the same thing for a categorical critic, where a value shift moves the
*support* rather than the probabilities (C51 overrides `bellman_target` with its projection and
reads the shift there). `terminal_target(value)` lifts an arbitrary scalar into the same target
representation, which is how NC-LQL's smoothing loss regresses its noisy levels onto the clean
value through the ordinary `loss()`. The un-reduced `raw(s, a, target=False)` stays an
implementation detail used only by a critic's own `loss()`, never by a Policy.

### Action scaling convention

Every continuous actor is trained to output a canonical **[-1, 1] per dimension** action,
independent of the env's real bounds (which aren't always ±1, e.g. Humanoid is ±0.4).
`Policy.scale_action()`/`unscale_action()` do the one-place affine remap to/from real env bounds.
Only two call sites ever need it: `ReplayBuffer.rollout()` (immediately before `env.step()`) and
`Policy.evaluate()`. Everything a Policy stores (trajectories) or trains against is **already**
canonical — a Policy re-invoking its own actor mid-`train_step` (e.g. for a Bellman bootstrap
action) can feed the result straight to its critic with no further conversion.

### Tunables all live in YAML -- there is no Python-level HParam system

Every tunable value -- architecture kwargs (`hidden_dim`, `layers`, `activation`, `layernorm`,
`residual`), optimizer kwargs (`lr`, `weight_decay`, `adam_beta1`, `adam_beta2`, `grad_clip`), and
algorithm tunables that used to be a separate `HParam`-declared axis (`rho`, `gamma`,
`tr_lambda_init`, PPO's `clip_eps`, ...) -- is a required, no-default **keyword-only** argument
supplied from YAML, across exactly two files: `config/components.yaml` (baseline kwargs per
component *class*) under `config/models.yaml` (per *policy* slots/params/profiles). There is no
Python-level fallback anywhere in this chain; a value listed in neither file is a
`KeyError`/`TypeError` at construction time, and `model_config.validate()` reports it before a run
starts (keyword-only-and-defaultless is exactly the marker that check keys off).
`config/components.yaml`'s `_`-prefixed top-level keys are YAML anchors the class entries
`<<:`-merge, not components; nothing resolves against them.

- **Component-level tunables** (everything an `Actor`/`Critic` itself needs — its architecture,
  its process's own constants, its target net's update law, and its optimizer's `lr`/
  `weight_decay`/`adam_beta1`/`adam_beta2`/`grad_clip` via the `Optimized` mixin in
  `networks/optim.py`, which every component owning an optimizer mixes in *before* `Component`)
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
`train_epoch()` calls `model.train_step(batch)` which returns `losses`; every loss key
must match a key in `model.create_optimizer()` — each is zero-grad'd, backward'd
(`retain_graph=True`, since e.g. an actor loss reads the critic's forward pass), grad-clipped to
`Policy.grad_clip(key, cfg["grad_clip"])` (the owning component's own `grad_clip` from
`config/components.yaml`, falling back to the run config for keys no component owns), and
stepped independently, so no optimizer step invalidates another loss's graph mid-loop. Periodic
eval (`evaluate_and_log`) checkpoints `latest` every eval and `best` on a new best avg reward,
under the run's `run_dir/checkpoints/` inside a sweep, else `checkpoints/<model>/<env>/<timestamp>/`
(with a `seed<i>/` level under either when a run vmaps more than one seed).

### Metrics (`src/barl/metrics.py`)

`train_step` returns *only* losses — every diagnostic goes to a `Metrics` sink instead, and the
training loop drains it once per step. Keys are `<owner>/<quantity>/<reduction>`, owner-first, so
wandb's sidebar reads as one section per component (`critic/`, `actor/`, `entropy/`) plus one per
algorithm (`amdp/`, `ppo/`, `iql/`).

`Policy.build_component()` hands each slot a `child(slot)` view as `self.metrics`, so a component
writes `self.metrics.dist("q", pred)` and it lands at `critic/q/*` **without naming itself** — the
same mechanism that lets `AutotuneRegulator` key its optimizer off `self.slot` rather than a `name`
kwarg. A Policy's own `self.metrics` is pre-namespaced to `algo_name()`. This is why a critic's
`loss()` no longer takes a `log` dict.

Families (`scalar`, `dist`, `vec`, `ensemble`, `ema_gap`, plus `Policy.record_optimizer`/
`record_batch`) each emit **one primary key and put their expansion on the debug tier** —
`log_every` paces the headline, `debug_every` the min/max/std/quantiles and gradient health.
`Policy.before_step(tiers)` arms them; an unarmed family early-returns, so a step that won't be
logged never pays for its reductions. Values stay 0-dim tensors until `drain()`, which crosses to
the CPU once per device per step rather than one `.item()` per key.

Two reductions that look alike are deliberately distinct: `critic/ensemble/disagreement` is spread
*across ensemble heads* (epistemic), while a distributional critic's `critic/q/std` is spread of
the return distribution itself (aleatoric).

`Logger` (`src/barl/logger.py`) wraps `wandb` (source of truth for history/plots) with a live
`rich` terminal panel driven off the same `log()`/`increment()` calls — nothing needs logging
twice. `log_quiet()` is the wandb-only path used for the debug tier, whose key count would swamp
the panel. `wandb.define_metric(step_metric=...)` alignment means step counters (`env_steps`,
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

Combos go onto one shared queue (algo-major) and are drained by one worker thread per `--gpus`
entry; a worker sets `CUDA_VISIBLE_DEVICES` to its own GPU, blocks on the subprocess, then takes
the next combo — so parallelism is exactly "one run per GPU at a time", never two on one device.
Terminal output from concurrent workers is `gpu<id>|<tag>`-prefixed and emitted under a lock, and
each run still gets its own directory, `sweeps/<timestamp>/<tag>/`, holding the `config.yaml` it
ran with, its `run.log`, and its `checkpoints/` (the run entry's `run_dir` key is what points the
trainer there — a run launched outside a sweep has no `run_dir` and keeps the global
`checkpoints/<model>/<env>/<timestamp>/` tree). `sweeps/<timestamp>/sweep.yaml` indexes them:
the axes, the resolved command, and one entry per run with its tag, directory and — rewritten once
the grid finishes — its exit code. `--sweep_dir` relocates that whole tree, which is how the Slurm
jobs put it on node-local scratch and copy it back (see below).

### Slurm (`scripts/*.slurm`, `scripts/submit.py`)

`scripts/submit.py` submits an `{algo x env}` matrix as one job array: it writes
`slurm_runs/<array job id>.csv`, one row per array task, and every task of that submission shares
the array job id as its **run id** — the manifest's filename and `run_id` column, the `--sweep_tag`
every run carries into wandb, and the name of its `~/results/<run id>/` directory. The id only
exists once `sbatch` has accepted the job, so `submit.py` submits `--hold`, writes the manifest,
then releases: held tasks cannot start before the file they read exists.

All three job scripts (`run_rl.slurm`, `run_sweep.sh`, `debug.sh`) `source scripts/slurm_common.sh`,
which establishes `RUN_ID`, a node-local `SWEEP_DIR` handed to `run_sweep.py --sweep_dir`, and a
`RESULT_DIR` under `~/results/`, plus an EXIT trap that copies the sweep tree back on *every* exit
path — that is what makes `#SBATCH --signal=B:USR1@300` load-bearing: a walltime kill still saves
whatever finished. `run_rl.slurm` is the array task (one manifest row each);
`run_sweep.sh`/`debug.sh` are single jobs with the grid written into them.

`submit.py --debug` is the array equivalent of `debug.sh`: same dev queue and 30-minute cap, but
over a whole matrix, one task at a time. It overrides `run_rl.slurm`'s `#SBATCH` partition/time/
job-name from sbatch's command line, so the job script stays the description of a real run.

### Registered algorithms (`src/barl/policies/`)

`DirectPolicy` (BC), `FlowBCPolicy`/`FlowQLPolicy` (flow-matching BC / flow-matching
+ Q), `AMDP` (direct-score regression against a critic's action-gradient, entropy +
trust-region duals — see the extensive docstring in `amdp.py`), `IQLPolicy`, `PPOPolicy`,
`SACPolicy`, `DIMEPolicy`, `SDAC`, `NCLQLPolicy`. All discrete-action support was removed with `QPolicy` (DQN) --
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
