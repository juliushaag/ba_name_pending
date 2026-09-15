# floq flow-matching critics

This document is the implementation contract for the `FloqCritic` component. It
translates Agrawalla et al., *floq: Training Critics via Flow-Matching for
Scaling Compute in Value-Based RL* (arXiv:2509.06863v2) into barl's critic
interface. The paper and the authors' reference implementation are the
authoritative sources:

* [paper (HTML, v2)](https://arxiv.org/html/2509.06863v2), especially sections
  4.1--4.3 and Appendix B;
* [paper abstract/version record](https://arxiv.org/abs/2509.06863);
* [CMU-AIRe/floq](https://github.com/CMU-AIRe/floq), in particular
  [`agents/floq.py`](https://raw.githubusercontent.com/CMU-AIRe/floq/main/agents/floq.py).

The goal in this repository is an additional critic selectable by SAC, AMDP,
DIME, and SDAC through the existing `critic` slot. The actor and policy remain
the algorithm's existing implementations. This is a deliberate adaptation of
the paper's FQL experiment: the paper trains a flow policy, a flow critic, and a
separate distilled critic together, while barl already supplies the policy and
the actor loss. The sections below mark which pieces are paper-faithful and
which are barl-specific.

## Mathematical contract

For a state-action pair $x=(s,a)$, let $z_0\sim\operatorname{Unif}[l,u]$.
The flow critic owns a scalar, time-conditioned velocity field

\[
v_\theta(t,z\mid s,a)\in\mathbb R,
\qquad t\in[0,1].
\]

With $K$ Euler steps, the scalar flow sample is

\[
z_{i+1}=z_i+K^{-1}v_\theta(i/K,z_i\mid s,a),
\quad i=0,\ldots,K-1,
\qquad Q_\theta(s,a;z_0)=z_K.
\]

The recursive input $z_i$ is the key distinction from a parallel ensemble:
the same velocity parameters are reused, but every call sees the previous
call's output. `K` therefore scales inference compute and the critic's
effective depth without adding parameters.

For a transition $(s,a,r,s',d,h)$, barl's replay contract supplies the
already accumulated $h$-step return `chunk_return`, a true termination flag
`terminated`, and the executed horizon `horizon`. Define

\[
\delta_h=(1-d)\gamma^h,
\qquad
y=\texttt{chunk\_return}+\delta_h
  \left(\bar Q(s',a')+b_\mathrm{ent}\right).
\]

Here $a'\sim\pi(\cdot\mid s')$, $b_\mathrm{ent}$ is the existing SAC-style
entropy bonus (zero for algorithms without one), and

\[
\bar Q(s',a')=\frac1m\sum_{j=1}^{m}
 Q_{\bar\theta}(s',a';z'_{0,j}),
\qquad z'_{0,j}\sim\operatorname{Unif}[l,u].
\]

The target flow is a stop-gradient Polyak copy of the velocity field. The paper
uses a fresh set of $m=8$ initial-noise samples for the target. The implementation
uses a deterministic midpoint grid for the side-effect-free `value()`/`raw()`
methods, while the training target uses fresh keys for its configured
`noise_samples`. Do not let target-flow gradients reach the live critic.

For the current transition, draw $z_0\sim\operatorname{Unif}[l,u]$ and
$t\sim\operatorname{Unif}[0,1]$, form the straight-line interpolant

\[
z_t=(1-t)z_0+t y,
\qquad u^\star=y-z_0,
\]

and regress the velocity field:

\[
\mathcal L_\mathrm{flow}(\theta)=
\mathbb E\left[
  \left(v_\theta(t,z_t\mid s,a)-(y-z_0)\right)^2
\right].
\]

This is the paper's Equation 4.2. It is not the ordinary scalar TD MSE on
`value(s, a)`: the interpolant and its velocity target are the learning
objective. Sampling $t=0$ only is a useful diagnostic, but is the paper's
ablated variant and should not be the default.

## Representation and numerical details

The flow field receives the state and action features plus two encodings:

1. Encode the scalar $z_t$ with the existing HL-Gauss construction: evaluate
   the Gaussian CDF at `num_bins` support edges spanning the configured
   expected return interval, then normalize the bin masses with a positive denominator so out-of-range interpolants remain finite. The encoder clamps only its z input to the support; it does not clip the integrated Q endpoint or Bellman target y. The paper adds
   Gaussian smoothing with `sigma=16.0`; this wide value is intentional and
   activates roughly 80% of bins at initialization. Passing raw `z_t` or a
   normalized scalar is useful only as an ablation because the paper reports
   flow collapse or materially worse results for those encodings.
2. Encode $t$ with a Fourier basis. The paper uses dimension 64. The current
   implementation uses the repository time-embedding utility with normalized
   times in `[0, 1]`; a scalar time input is an ablation.

The velocity head is scalar per ensemble member. The flow chain and losses run in
at least float32: repeated Euler
additions and the HL-Gauss CDF are sensitive to reduced precision. The existing
MLP tier and `ConfigDict` conventions still control hidden width/depth and
activation; the paper's reference uses GELU and a 4-layer, width-512 MLP.

The implementation derives `q_min` and `q_max` from `env_stats["return_min"]` and `env_stats["return_max"]` unless explicitly configured. The initial flow interval is scaled about zero by setting `noise_min = noise_coverage * q_min` and `noise_max = noise_coverage * q_max`; with the default `noise_coverage=0.1`, its width is 0.1 times the Q-range. This scaling about zero is the barl implementation choice. If either environment return bound is missing, the implementation falls back to q_min = -1 and q_max = 1; configure bounds to match expected discounted returns, including any entropy bonus, before training. The paper describes choosing an upper-anchored interval with `u = Q_max` and tuning `l >= Q_min`. Validate finite bounds and `q_min < q_max` before constructing the flow. A too-narrow interval makes the field nearly constant in `z_t`; a very wide interval produces excessive curvature and Euler error.

## Required barl component shape

Implement `FloqCritic` as a registered `QCriticBase` subclass. Keep the public
contract used by `RLPolicy`, `SAC`, `AMDP`, and action-selection utilities:

* `raw(s, a)` returns `(n_models, batch, 1)` flow means (or an equivalent raw
  shape consumed consistently by `reduce_value`), and is deterministic and
  side-effect free;
* `value(s, a)` returns `(batch, 1)` after the existing ensemble reduction;
* `expectation(s, a)` is the mean across ensemble heads;
* `td_loss(obs, action, next_obs, next_action, chunk_return, terminated,
  horizon, entropy_bonus=0.0, key=...)` returns `{"critic": scalar_loss}`;
* `target_modules()`/`after_step()` expose and Polyak-update every target-flow
  parameter; and
* `derived()` reports at least the resolved flow interval, `K`, `m`, number of
  bins, and HL-Gauss sigma for checkpoint/debug output.

`td_loss` calls the shared `n_step_discount`/Bellman arithmetic so chunk
returns, termination, horizon, and entropy semantics remain identical to every
other scalar critic, then uses that scalar target as $y$ in the flow-matching
loss. Do not call `QCriticBase.regression_loss` on a scalar and claim
that is floq training; the representation-specific loss belongs in the floq
critic.

Use `jax.lax.scan` or a statically unrolled `for` for the Euler chain. `K` is a
static configuration value, not a data-dependent loop length. For a vectorized
batch, represent flow samples as `(n_models, batch, samples, 1)` internally;
reduce the sample axis only when producing a scalar expectation. Keep action and
observation broadcasting explicit so `chunk_size > 1` still sees one flattened
plan action, as required by `RLPolicy.train_step` and `AMDP.loss`.

For deterministic public inference, the implementation uses a computed midpoint grid

\[
z_{0,j}=l+(j+\tfrac12)(u-l)/m
\]

The midpoint grid is the fixed value quadrature used by `value_samples`; training target noise remains random and key-owned. This fixed-grid choice is a barl adaptation; the reference implementation samples uniforms for both current and target flow returns.

## Target reduction and pessimism

The author's code creates `flow_num_ensembles=2`, computes each head's flow
returns, and aggregates target returns with either mean or min. Barl already
separates the two choices as `rho` (the value used by actors) and
`pessimistic_target` (the Bellman bootstrap). Preserve that distinction:

* `raw` retains the head axis;
* `reduce_value(raw)` applies mean minus `rho * std` (for two heads, `rho=1`
  is the usual min approximation already used by `EnsembleCritic`);
* `bootstrap_from_raw` uses the mean unless `pessimistic_target` is true, in
  which case it uses the lower-head reduction; and
* do not take a min of initial-noise samples within a head. Noise samples are
  Monte Carlo estimates of one head's scalar flow output, not independent
  pessimistic critics.

The paper's default is mean aggregation and no clipped double Q except for
selected hard environments. The barl defaults are explicit: `rho=0` gives mean actor pricing and
`pessimistic_target=false` gives mean Bellman bootstrapping, preserving the
existing SAC and AMDP optimism behavior.

## SAC, AMDP, DIME, and SDAC adaptation

`MaxEntRLPolicy.train_step` already does the common sequence: sample a
stop-gradient next action, call critic `td_loss`, and gate the actor/entropy
updates by policy delay. With a correctly implemented `FloqCritic`, SAC needs no
special actor code. Its Bellman target is the scalar target $y$ above, with
the existing `alpha * entropy` bonus. SAC's actor receives gradients through
`critic.value`; this differentiates through the K-step Euler chain and costs K
velocity evaluations per actor score.

AMDP similarly works through its existing `nnx.grad(... critic.value ...)` path.
The paper avoids this potentially expensive action derivative by training a
separate distilled critic

\[
Q_\psi^\mathrm{distill}(s,a)\approx
\frac1m\sum_jQ_\theta^\mathrm{flow}(s,a;z_{0,j}),
\quad
\mathcal L_\mathrm{distill}=\left(Q_\psi^\mathrm{distill}-\operatorname{sg}(\bar Q_\theta^\mathrm{flow})\right)^2.
\]

The reference paper uses a distilled critic for cheaper policy extraction. This barl implementation deliberately differentiates the integrated flow directly through its K-step chain, so actor scoring pays the K-fold velocity evaluation cost. No distilled critic is included; the flow target remains stop-gradient and is the sole Bellman training signal.

DIME and SDAC also consume `critic.value` through their existing actor APIs.
Their entropy and exploration objectives remain unchanged. Keep critic flow
noise independent of actor SDE/diffusion keys by using the explicit `key` passed
to `td_loss`; never consume the actor's private `rngs` from the critic.

## Paper-faithful pieces versus this integration

| Area | Paper / author code | barl implementation contract |
| --- | --- | --- |
| Critic | Velocity field over scalar return; Euler flow; dense flow-matching target | Same, registered as one `QCriticBase` component |
| Initial noise | Uniform interval, sampled per call | Uniform during training; fixed midpoint grid for deterministic `raw/value` |
| Target | Target velocity field; mean over 8 noises | Same target module and `noise_samples=8`; reuse shared n-step and entropy semantics |
| Ensemble | Two flow heads; paper usually mean/min by experiment | Existing `n_models`, `rho`, and `pessimistic_target` semantics |
| Stabilization | HL-Gauss return embedding, sigma 16; 64-D Fourier time embedding | Required defaults; reuse local embedding utilities |
| Policy | FQL behavior-cloning flow plus one-step policy; distilled Q used for extraction | Existing SAC/AMDP/DIME/SDAC actor; direct integrated-flow gradients through the K-step chain; no distilled critic is included |
| Replay | One-step `(r, mask)` in reference FQL | Chunked `chunk_return`, true `terminated`, and `horizon` are mandatory |
| Optimizer | One Adam flow update plus distill update in reference | One `critic` optimizer and target lifecycle through `QCriticBase`; extra optimizer only if distillation is explicit |
| Return bounds | `r_min/r_max`, reward offset, interval derived in reference code | `env_stats` return bounds, explicit validation, and checkpointed resolved config |

The reference implementation also has an optional residual velocity field and
separate policy encoders. The current generic barl critic exposes the shared MLP
configuration, with `residual=false` and no separate policy encoder by default.

## Configuration defaults and rationale

Every setting is a `Config`/`ConfigDict` field on the critic, so config
snapshots and `--set model.components.critic.<key>=...` work. The implemented
defaults are:

| Key | Default | Rationale |
| --- | ---: | --- |
| `flow_steps` | `8` | Paper default; 4 is a useful hard-task/CPU setting |
| `noise_samples` | `8` | Paper target average |
| `value_samples` | `8` | Fixed midpoint samples averaged by deterministic `raw` and `value` |
| `noise_coverage` | `0.1` | Reference scaling around zero; use `0.25` for a wider initial interval |
| `q_min` | `null` → env_stats.return_min (fallback -1) | Lower expected return bound; include discounted entropy effects in explicit bounds |
| `q_max` | `null` → env_stats.return_max (fallback +1) | Upper expected return bound; include discounted entropy effects in explicit bounds |
| `num_bins` | `51` | Author code; enough categorical coverage for scalar conditioning |
| `sigma` | `16.0` | HL-Gauss smoothing in bin-width units; essential stabilization |
| `time_emb` | `fourier` (64-D) | Author code and paper default |
| MLP activation | `gelu` | Author code |
| MLP hidden shape | four 512 layers | Author code; use the repository tier for controlled scaling |
| `polyak_tau` | `0.005` | Paper/reference target smoothing |
| `rho` | `0.0` | Paper's mean aggregation; set 1 for pessimistic SAC-style pricing |
| `pessimistic_target` | `False` | Paper default mean; turn on explicitly for clipped-double behavior |
| `train_at_zero_only` | `False` | Full dense supervision is the method; true is an ablation |

## Acceptance tests

Add tests alongside the existing critic contract tests. Test mathematical laws rather than only whether a JAX update runs:

1. Build the registered critic with the tiny test tier and assert finite, correctly shaped `raw`, `value`, `expectation`, and `td_loss` outputs.
2. Replace the velocity field with a constant c. Assert K-step Euler output is exactly `z_0+c`, independent of K, and that repeated `value(s,a)` calls are identical.
3. Set a hand-built velocity field and compare its K-step output with a Python Euler reference for K in {1, 2, 4}.
4. With a fixed target flow and fixed next noise, assert the target equals `chunk_return + (1-terminated)*gamma**horizon*(mean_target + entropy_bonus)`; test a multi-step horizon and a terminated row separately.
5. Assert the action gradient of `critic.value` is finite, and that `frozen(critic)` contributes no parameter gradients.
6. Assert target parameters change under `after_step()` while live flow parameters do not; test target delay if configured.
7. Verify `rho` changes actor pricing but not a non-pessimistic Bellman target, and `pessimistic_target` changes only the Bellman target.
8. For `chunk_size > 1`, verify flattened plan actions are consumed as one state-action input and output remains `(batch, 1)`.
9. Resolve a SAC and AMDP model with a `FloqCritic` override and verify all keys appear in the resolved snapshot.

Recommended checks after implementation:

```bash
uv run pytest tests/test_critics.py tests/test_critic_stats.py tests/test_floq.py -q
uv run pytest tests/test_amdp.py tests/test_actors.py -q
uv run ruff check .
BARL_ALLOW_CPU=1 uv run python scripts/submit.py --algos SAC --envs HalfCheetah --set model.components.critic.class=FloqCritic --set model.components.critic.flow_steps=2 --set run.total_env_steps=64 --set run.random_steps=8 --validate --executor local
```

The final command checks configuration and registration only; it is not evidence of a training run or learned performance. For an actual comparison, hold algorithm, seed, replay, actor, batch, and optimizer settings fixed; compare `flow_steps` in {1, 4, 8} and report wall-clock cost. The paper observes diminishing returns and occasional degradation at 16 steps, so larger K is not automatically better.

## User-facing examples

Select the critic for any existing continuous-control algorithm with a model override. The override is resolved into the run snapshot and remains checkpoint-compatible:

```bash
# SAC + floq, default settings
uv run python scripts/submit.py --algos SAC --envs HalfCheetah --seeds 3 --set model.components.critic.class=FloqCritic

# AMDP + floq, cheaper four-step critic
uv run python scripts/submit.py --algos AMDP --envs HalfCheetah --seeds 3 --set model.components.critic.class=FloqCritic --set model.components.critic.flow_steps=4

# Explicit pessimistic target and wider centered interval
uv run python scripts/submit.py --algos SAC --envs Ant --seeds 3 --set model.components.critic.class=FloqCritic --set model.components.critic.pessimistic_target=true --set model.components.critic.rho=1.0 --set model.components.critic.noise_coverage=0.25
```

## Sources

1. Bhavya Agrawalla, Michal Nauman, Khush Agrawal, and Aviral Kumar,
   “floq: Training Critics via Flow-Matching for Scaling Compute in Value-Based
   RL,” arXiv:2509.06863v2, 23 October 2025,
   [HTML paper](https://arxiv.org/html/2509.06863v2),
   [abstract/version record](https://arxiv.org/abs/2509.06863).
2. CMU-AIRe, “floq” reference implementation,
   [repository](https://github.com/CMU-AIRe/floq),
   [`agents/floq.py`](https://raw.githubusercontent.com/CMU-AIRe/floq/main/agents/floq.py).
