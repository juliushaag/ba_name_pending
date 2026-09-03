# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`barl_jax` is a research framework for online off-policy deep-RL experiments across MuJoCo Gym,
DeepMind Control Suite and ManiSkill continuous-control benchmarks. It implements a range of actor-critic
algorithms (`SAC`, `AMDP`, `DIME`, `SDAC`, `NCLQL`) behind one shared `Policy`/`Environment`
interface so any algorithm can be pointed at any registered environment through a single training
script and YAML config.

**JAX only.** The framework is jax/flax-nnx/optax, and the torch implementation this was ported
from (`src/barl`) is gone. Comments that say "the torch version did X" are kept where they explain
*why* the current code looks the way it does, not because that code is still around.

Torch *is* installed, but only as a transitive dependency of `mani_skill` (via sapien) -- no
`barl_jax` module imports it, and nothing in a training step touches it. The ManiSkill family is
wired on the CPU backend precisely so it hands back numpy and that stays true; ManiSkill's
GPU-parallel simulator would put torch tensors in the rollout loop and is deliberately not used
(see `environments/mani_skill.py`).

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

**`scripts/submit.py` is the entry point for running anything** -- one run or a thousand, on a
workstation or on the cluster. It expands the sweep, writes each run's self-contained config, and
hands the list to an executor:

```bash
uv run python scripts/submit.py --algos SAC NCLQL --envs HalfCheetah CheetahRun --seeds 3
uv run python scripts/submit.py --algos NCLQL --envs WalkerRun --debug        # 30-min smoke test
uv run python scripts/submit.py --algos SAC --envs HalfCheetah --executor local --validate
```

`--executor auto` (the default) picks **slurm** where `sbatch` is on PATH, **tmux** where only tmux
is, and **local** otherwise, so the same command works everywhere. `local` runs the sweep in this
terminal -- one worker per GPU, each pinning its subprocess with `CUDA_VISIBLE_DEVICES` and taking
the next run only once its own finishes, so a run never shares a GPU -- and is the only one that
blocks. `--max-parallel` is the runs in flight (Slurm's `--array %N`); setting it above the GPU
count is how you deliberately oversubscribe a device. `--seeds N` is *not* a process axis: those N
seeds are vmapped inside a single run, and the default comes from `config/run/defaults.yaml`'s
`num_seeds` -- the one place the seed count is declared.

The grid is a cross product over named axes: `algo`/`env`/`seed`, plus **any** `model.<dotted path>`
axis (anything under `config/models/` or `config/components/`, `.class` and `.size` included) and
any `run.<key>` axis. Declare them inline or in a `--grid` spec file
(`config/experiments/sweep_grid.yaml` is a documented example, with an `exclude:` list for pruning);
`--dry-run` prints the expansion instead of running it. Because the axes are just overrides, **no
config file is ever edited to run a sweep**, and because the grid is expanded *before* submission,
a model axis reaches the cluster exactly as it reaches this machine:

```bash
uv run python scripts/submit.py --algos AMDP --envs HalfCheetah --seeds 3 \
    --set model.components.critic.size=large --set run.utd_ratio=0.5
uv run python scripts/submit.py --grid config/experiments/sweep_grid.yaml --dry-run
```

Every submission leaves one `sweeps/<timestamp>/` folder, the same shape whichever backend produced
it: `experiments.csv` (the manifest, one row per run), `sweep.yaml` (what the grid was, one entry
per run), and one directory per run holding its `config.yaml`, its log and its `checkpoints/`.

`scripts/online_learning.py` is what an executor invokes per task -- it takes one config and runs
it. It is not the entry point, but it stays the way to re-run a single finished run from the config
it left behind:

```bash
uv run python scripts/online_learning.py --config sweeps/<ts>/<tag>/config.yaml
```


To see what a `model:` string actually resolves to, with every value attributed to the layer that
set it:

```bash
uv run python scripts/print_config.py AMDP    # no args: every model in config/models/
uv run python scripts/print_config.py AMDP --json                          # the flat resolved form
uv run python scripts/print_config.py SAC --set components.critic.size=large   # preview an override
```

And to check the tree itself -- no Python defaults, no unreferenced presets, no slot missing its
`size:`, no run key read but undeclared:

```bash
uv run python scripts/lint_config.py
```

There is a pytest suite (`uv run pytest tests/ -q`, ~400 tests in about ten seconds; every
registered component is built from its real `config/components/` entry, and `tests/test_config_tree.py`
holds the tree's guarantees). The complement to it is `--validate`: a cheap smoke test that runs one
rollout chunk + one `train_step` + one optimizer step per `{algo, env}` combination (no
eval/checkpointing/wandb logging), to catch shape/dtype/algo-env mismatches before committing to a
full run. It collapses every axis but algo/env to its first value and streams each run's output:

```bash
uv run python scripts/submit.py --algos SAC NCLQL --envs HalfCheetah CheetahRun \
    --executor local --validate
```

Launch an `{algo x env}` matrix through the one interface that works in both places — a Slurm job
array on the cluster, one tmux window per run on a workstation (see Execution backends below):

```bash
uv run python scripts/submit.py --algos NCLQL SDAC --envs WalkerRun --seeds 5
uv run python scripts/submit.py --algos NCLQL --envs WalkerRun --executor tmux --dry-run
uv run python scripts/submit.py --algos NCLQL --envs WalkerRun --debug     # 30-min smoke test
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
config/models/ layering: it is a function of the environment, not a tunable, so a component that wants
one of its values tunable takes a normal kwarg that falls back to the `EnvStats` entry.

### `Policy` / `Actor` / `Critic` split

Every algorithm is one `Policy` subclass (`src/barl/policy.py`) composing an `Actor`
(`src/barl/networks/actor.py`) and, where applicable, a `Critic` (`src/barl/networks/critic.py`).
The load-bearing convention throughout the codebase:

- **Actor/Critic = pure architecture** (plus whatever tunables are intrinsic to that architecture's
  own process — e.g. a diffusion actor's noise schedule, a critic's ensemble pessimism — supplied
  as required kwargs from `config/`, no Python-level defaults). No algorithm logic
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
a slot-namespaced `metrics` sink, and the `before_step`/`after_step` lifecycle. The Policy
registers each slot in `self.components` and fans both hooks out over it, which is why a critic's
Polyak update lives in `QCriticBase.after_step()`
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
`config/components/critics.yaml`-supplied, see below) via one of two head-building bases — `MLPCriticBase`
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

### Tunables and structure are declared next to the code -- YAML is override-only

Defaults live on the classes that use them, as class attributes, in two kinds (`src/barl_jax/hparam.py`):

- **`HParam(default, *, range=, choices=, descr=)`** -- a *searchable algorithmic knob*: the default
  the code ships, the range/choices a hyperparameter sweep varies, and a description. These are
  the only class attributes a search (scripts/hpo.py) may target.
- **`Config(default)`** -- a *structural default*, type inferred: how a net is put together, the
  squash/ladder-class selection, an optimizer role. Not searchable -- you vary structure with
  `--set` or a grid axis, not per-trial.

**Which net / which time embedding is a configuration value, not an inheritance
choice.** The named styles (MLP surfaces, time/level-embedding coders) live in
code registries (`barl_jax.hparam.STYLES`: `net.gelu`, `time.fourier`, ...),
registered by `networks/mlp.py` and `networks/flow.py`. A component *selects*
one with a style-tagged `Config`:

    class DDPMActor(Actor, ActorOptim):
        net_style = Config("gelu",   style="net")    # net.gelu's whole surface
        time_emb  = Config("sin_20", style="time")   # time.sin_20's whole surface

so switching `--set model.components.actor.net_style=silu` (or a grid axis)
changes the whole net, and the resolved config records the *choice* -- a style's
surface keys are attributed as `style: net.gelu`, never hidden as class
defaults. A component can still pin one key on top of a style (a per-slot
`activation:` beats the style's), and a style's surface sits below the component
own declarations and the slot's YAML.

What is still composed by **class inheritance** is *algorithm geometry* -- an
optimizer role (`CriticOptim`/`ActorOptim`/`DualOptim` in networks/optim.py), a
critic's Bellman geometry (`CriticLossConfig`/`CriticDistConfig`/`Polyak` in
networks/critic.py): those are decisions about which algorithm code a component
implements, not about a selectable default. A concrete component states nothing
beyond its geometry bases, its style selectors, and the HParams only it owns:

`Component`/`Policy` collect `HParam`s and `Config`s down the MRO into `_hparams`/`_configs` (own /
more-derived declarations win, so a mixin listed earlier in a class's bases takes precedence) and
resolve each to an instance attribute at construction: the YAML layer wins if present, the class
default otherwise. Because the default lives in the declaration, **the resolved config a run logs
can never be silent about it** -- every value is attributed to the layer that set it (`decl configs:`,
`decl hparams:`, `models: <model>.components.<slot>`, `size: <kind>.<name>`, `override: ...`).

```
config/
  models/       sac.yaml  dime.yaml  ...  aliases.yaml        per *policy* overrides (slots/params/training)
  experiments/  online_learning.yaml  sweep_grid.yaml         run lists and sweep specs
  search/       demo.yaml                                     HPO search spaces (scripts/hpo.py)
```

`config/components/`, `config/presets/`, `config/run/defaults.yaml` and `config/run/envs.yaml` are
gone: component defaults are class HParams/Configs (inherited from style mixins), run-key defaults
are `barl_jax.run_config.RUN_HPARAMS`, and per-env facts/overrides are `Environment.stats` /
`Environment.defaults` class attributes merged down the MRO.

`size: large` resolves one rung of the component kind's *capacity ladder* (code-declared in
`barl_jax.hparam`), where `<kind>` is the component's declared `kind:`. **The size tables own their
keys exclusively**: no component class declares `latent_dim`/`layers`/`n_models`, every slot of a
sized kind must name a `size:`, and all sizes of
a kind must declare the same keys -- all three checked by the lint. That is what makes `large`
mean one net for every critic class, so `--set model.components.critic.size=large` is a capacity
comparison rather than an incidental-architecture one. Ladders are width at fixed depth; a net
needing a different depth states `layers:` inline and `print_config.py` reports it as shadowing.

Precedence for a component kwarg, lowest first:

| # | layer |
|---|---|
| 1 | the class's inherited `Config` structural defaults (composed down the MRO) |
| 2 | the class's inherited `HParam` algorithmic defaults |
| 3 | the slot's `size:` (a capacity-ladder rung) |
| 4 | the slot's own keys (`config/models/<policy>.yaml`) |
| 5 | the active profile's block for that slot |
| 6 | the run's dotted overrides (`model_overrides:`, i.e. a `model.*` sweep axis) |
| 7 | the Policy's leftover `**kwargs` |
| 8 | the explicit kwargs at its own `build_component()` call site |

1-6 live in `config_tree.py`; 7-8 in `policy.py`. `params:` and `training:` use the same chain
minus the structural rungs, with the policy class's own HParams as their lowest rung.
`components.<slot>.class` and `.size` are overridable too, which is what makes an architecture axis
and a capacity axis sweepable without editing a file.

```bash
uv run python scripts/print_config.py AMDP          # resolved, every value attributed to its layer
uv run python scripts/print_config.py AMDP --json   # the flat form a run snapshots and logs
uv run python scripts/lint_config.py                # the whole-tree gate
```

**`src/barl_jax/config_tree.py` is the only resolver.** It imports only `yaml`/`os`/`pathlib` and
`barl_jax.configs` at import time; the component/policy class registry is pulled lazily when a class's
declared defaults (or `kind`) are needed. Because defaults live on the classes, the launcher
(`scripts/grid.py`, `scripts/hpo.py`, `scripts/submit.py`'s check) now imports `barl_jax` directly
under `JAX_PLATFORMS=cpu` -- that is the price of code-coupled defaults, and `test_launcher_devices.py`
still holds the run-reaches-GPU guarantee. `model_config.py` is a thin registry-aware facade over it
(class lookup, validation) plus a process-wide default tree.

**A run's snapshot is fully resolved.** `ConfigTree.snapshot_for()` returns the complete flat
resolved config under `resolved:` -- every declared HParam/Config with its effective value -- and a
tree constructed with that snapshot *serves those values directly* rather than re-running the
layering -- so a saved sweep run and a checkpoint replay exactly, whatever the code or YAML says
later. `Policy.save` writes the same snapshot into `meta.json`, which is what closes the old gap
where a checkpoint silently reconstructed a different architecture. `Policy.load` restores it and
the run's overrides, then puts both back, so loading a checkpoint no longer redefines the config
for the rest of the process.

**What is not resolvable from the declarations alone is reported by the component.** A critic's
support follows the env (`v_min`/`v_max` from `EnvStats`), `HLGaussCritic`'s `sigma` is derived from
it and `num_bins`, and AMDP's `erf_k` from its actor's noise schedule. Each such component overrides
`Component.derived()`, the Policy collects them, and the trainer logs them under `model/derived.*`.

**Run keys** are declared in `barl_jax.run_config.RUN_HPARAMS` and nowhere else. A resolved run is
complete by construction (every run key present), `trainer.py` indexes `cfg[...]` with no
`cfg.get(key, default)` anywhere, and the lint rejects a YAML key that names no code declaration.
Six of them (`config_tree.TRAINING_KEYS`) may also be set by a model's own `training:` block, for
knobs that belong to the *algorithm* rather than the experiment; per-env facts and per-env run
overrides are `Environment.stats` / `Environment.defaults` class attributes (`return_min`/`return_max`
included), merged down the MRO in `get_env_stats()`/`merged_defaults()`.

**What every run logs to wandb** (`trainer.wandb_config`): `run/*` every run key, `model/*` every
resolved component kwarg plus its `class` and `size`, `model/derived.*` the values above,
`env/*` the full `EnvStats` including the array bounds, `meta/git_sha` and `meta/config_hash` (a
digest of everything but the seed, so "same config, different seed" is one group), and `snapshot`
for exact replay.

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
`config/components/`, falling back to the run config for keys no component owns), and
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
writes `self.metrics.dist("critic", pred)` and it lands at `critic/q/*` **without naming itself** — the
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

### Run-config resolution (`src/barl_jax/config.py`)

An experiment file (`config/experiments/*.yaml`) carries a `defaults:` dict and a `runs:` list.
`load_config()` merges each run entry over, lowest first: `config/run/defaults.yaml` < this file's
`defaults:` < the model's own `training:` block < `config/run/envs.yaml`'s `defaults:` for that
env < the run entry — and returns the resolved list, which `scripts/online_learning.py` iterates,
running each `{model, env, ...}` combination in series in one process. A resolved run carrying a
key `config/run/defaults.yaml` does not declare is an error, not a silent no-op.
`model_overrides:` is the one key merged key-wise across the levels rather than replaced
wholesale, since a per-env and a per-run override normally mean to stack.

### Launching: one CLI, three executors

`scripts/submit.py` is the only entry point, and it is only a CLI. The pipeline is three stages
with one direction of travel:

```
submit.py          expand the grid → write every run's config → hand the list over
  grid.py          axes / combos / exclude / naming / materialization  (pure, executes nothing)
  executors.py     LocalExecutor | SlurmExecutor | TmuxExecutor
       └─ per task: online_learning.py --config sweeps/<ts>/<tag>/config.yaml
```

**The grid is fully expanded before anything is submitted.** That is the load-bearing decision:
after `grid.materialize()`, a run *is* a path to a self-contained config, so an executor's whole job
is "run these N commands, at most M at a time". None of the three — nor `run_rl.slurm`, nor
`tmux_pool.py`, nor the manifest format — knows what an algorithm, an environment or an axis is,
which is why an arbitrary `model.*` axis reaches the cluster without any of them learning anything.
The manifest (`experiments.csv`) is four columns, `index,run_id,tag,config`, and only the last is
load-bearing.

Each run's `config.yaml` carries a fully resolved `snapshot:` taken *under that run's own
overrides*, so a run replays exactly regardless of what `config/` says later (see the snapshot
section above).

| | local | slurm | tmux |
|---|---|---|---|
| a task is | a subprocess | an array task (`run_rl.slurm`) | a window (`tmux_pool.py` opens it) |
| the queue is | one worker thread per GPU | `--array=0-N%M` | the manager's GPU pool |
| default parallelism | one per detected GPU | 4 | one per detected GPU |
| scratch is | none — writes in place | node-local `$TMPDIR` | none — writes in place |
| results get there by | already there | `cp` in the exit trap | already there |
| blocks until done | yes | no | no |
| cancel with | ctrl-C | `scancel <id>` | `tmux kill-session -t barl-<name>` |
| status from | the terminal, then `sweep.yaml` | `sacct` | `sweeps/<name>/tasks.yaml`, and the live table |
| the run id is | the sweep's timestamp | the array job id | the sweep's timestamp |

Every submission lands in **`sweeps/<timestamp>/`**, the same shape whichever backend produced it:
`experiments.csv`, `sweep.yaml` (the grid, one entry per run, with exit codes once a blocking
executor finishes), and one directory per run holding its `config.yaml`, its log and its
`checkpoints/`. `--sweep-dir` relocates the tree.

**Slurm** (`SlurmExecutor`). One array task per *run* — per grid point, not per `{algo, env}` — so a
grid that varies a model axis submits that many more tasks, each one GPU's work, queued
independently. Every task of one submission shares the array job id as its **run id**: the
manifest's `run_id` column, the wandb sweep tag, and the folder they all write into. The id only
exists once `sbatch` has accepted the job, so `submit.py` submits `--hold`, writes the manifest,
then releases — held tasks cannot start before the file they read exists. Before any of that it
resolves and validates every algorithm's model config in-process, pinned to `JAX_PLATFORMS=cpu`
since a login node has no GPU to spend (`--skip-check` opts out), because a config error would
otherwise fail every task in the array identically.

`run_rl.slurm` is the only Slurm job script: one array task per manifest row, self-contained, and it
must be submitted through `submit.py` — `SWEEP_NAME` comes from that script's `--export`. It reads
its row, runs that config with `--run-dir` pointed at node-local scratch, and copies out on *every*
exit path, which is what makes `#SBATCH --signal=B:USR1@300` load-bearing: a walltime kill still
saves whatever finished. The run's directory under the sweep folder already exists — it is where
`submit.py` put the config — so the copy-back merges into it. The job's `term.out`/`term.err` are
**moved** (not copied) in beside it: a rename keeps slurmstepd's descriptor live, so the relocated
file still collects the footer and any cancellation notice a copy would miss. `SWEEPS_ROOT`
overrides where `sweeps/` itself lives.

**tmux** (`TmuxExecutor`). It writes the manifest, then opens one detached session `barl-<timestamp>`:
`scripts/tmux_pool.py` in window 0 owning the queue and a `rich` status table, and one window per
run that it opens as a slot frees. Each window runs the *same command* a Slurm array task runs.
Unlike a Slurm task there is no node-local scratch to copy back — it is the same filesystem, and
each run writes only into its own directory, so it writes there directly. Details that are
load-bearing rather than incidental: `PATH` is passed explicitly to the session and to every window
(a window otherwise inherits the tmux *server's* environment, which need not have this venv's `uv`)
— the tmux answer to `--export=ALL`; the exit code comes from `${PIPESTATUS[0]}` because it is piped
through `tee`; `remain-on-exit` is set on every window and successes are killed explicitly, which is
what leaves a failure on screen; `tmux_pool.py` traps **SIGHUP** as well as SIGINT/SIGTERM, since
`kill-session` hangs the pane up rather than terminating it; and `submit.py` checks the session is
still alive a second after opening it, because `tmux new-session` succeeds even when the manager it
was handed dies on the spot.

**local** (`LocalExecutor`). The foreground option, and the only blocking one: one worker thread per
GPU, each pinning its subprocess with `CUDA_VISIBLE_DEVICES` and blocking on it before taking the
next run. It is never chosen by `auto` — running a sweep in your terminal is something you ask for.
`--validate` streams each run's output through it, label-prefixed.

**Where jax runs is decided by the task, never by whoever launched it.** A run that
lands on the CPU is silent -- it trains, logs and checkpoints normally, ~160x slower per
gradient step -- so nothing downstream can tell it from a slow algorithm, and one sweep
spent 10 GPU-hours reaching 2% of a run before anyone could. Four things keep that from
recurring, and a change to any of them should keep the others in mind: `check_config()`
runs in a **child process** (its `JAX_PLATFORMS=cpu` cannot reach a run, since sbatch's
`--export=ALL`, tmux's `-e` and `LocalExecutor`'s `env=` all hand a run the launcher's
environment); `executors.child_env()` strips `DEVICE_VARS` from what it passes;
`run_rl.slurm` and `tmux_pool.py` `unset` the same list at their end, which is the only
side that can, since neither `--export=ALL` nor a tmux server's environment is the
launcher's to filter; and `online_learning.require_accelerator()` refuses to start a run
with no GPU/TPU backend whatever the cause -- a leaked variable, a venv without the CUDA
plugin, a bad node -- unless `BARL_ALLOW_CPU=1`. `tests/test_launcher_devices.py` holds
all four. It is also why `grid.py` loads `config_tree.py` **by path** rather than as
`barl_jax.config_tree`: the package's `__init__` imports every policy and environment,
so importing it is what put jax in the launcher and made a device variable look
necessary there at all.

`submit.py --debug` runs one task at a time with a 30-minute cap (`timeout` under tmux and local,
`--time` under slurm, plus the dev queue there). It overrides `run_rl.slurm`'s `#SBATCH`
partition/time/job-name from sbatch's command line, so the job script stays the description of a
real run.


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

### EMA vs. Shadow (`src/barl/networks/ema.py`) -- **stale: no such module exists**

Two different "frozen reference copy" wrappers, not interchangeable: `EMA` exponentially blends
toward the source every `step()` call (used for Q target nets — call once per train step) and is
kept `vmap`-safe (no `.item()`/Python-scalar extraction in its update, no `.data` in-place ops).
`Shadow` hard-copies the source once per epoch via `on_epoch_start()` (used for trust-region
references, e.g. AMDP/DIME regularizing their live diffusion net against last epoch's snapshot) —
named after the `Policy` lifecycle hook that's meant to drive it.
