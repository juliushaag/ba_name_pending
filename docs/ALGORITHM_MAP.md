# Algorithm map

Equation-to-code inventory required by `docs/ALGORITHMIC_CLARITY.md`'s phase 1
("Audit the other learning updates") and phase 3 ("provide a compact
equation-to-code map next to the owning objective or in a linked algorithm
reference"). One section per registered algorithm. `file:line` references are
against the commit this document was written against; re-check them before
trusting a reference after further edits.

Shared machinery every algorithm below reuses, not repeated per section:

- **Gradient isolation**: `frozen(module)` (`components.py:66-85`) =
  `jax.lax.stop_gradient(module)` + `.eval()` (also freezes BatchNorm/BatchRenorm
  running-stat updates) — used at every actor-loss-through-critic call site.
  `TRAINABLE = nnx.All(nnx.Param, nnx.Not(is_target))` (`components.py:37`)
  structurally excludes every `*_target`-suffixed param from all training
  gradients; this is not a convention, it is what `nnx.value_and_grad`'s
  `argnums=nnx.DiffState(0, TRAINABLE)` actually differentiates against
  (`trainer.py`'s `train_step`).
- **Target-network update**: `QCriticBase.after_step` (`critics/base.py:64-82`),
  a Polyak blend at `polyak_tau`, gated at `target_delay` — independent of
  `policy_delay` (the actor/dual's own cadence).
- **Optimizer-key ownership**: `OptimizerMixin.optimizer_labels()`
  (`optimization.py:33-56`) partitions params by component slot by default;
  `SDAC`'s `DDPMActor.param_labels` (`actors/sdac.py:356-359`) is the one
  override, routing `explore_raw` onto its own `"explore"` key.
- **Best-of-N / Mean-of-N at inference** (`execution.py`): isolated from every
  training path below — `execution` is imported by neither `policies/rl.py`,
  `policies/lql.py`, nor any file under `critics/` (verified:
  `rg -l execution src/barl_jax/policies/ src/barl_jax/critics/` → no hits).
  `Policy.score`/`RLPolicy.score` (`policies/rl.py`) is the only bridge, called
  from `Policy.inference_fn(spec=...)`/`StackedPolicy`, never from `train_step`.

---

## SAC

| | |
|---|---|
| Critic objective | `QCriticBase.td_loss` (`critics/base.py:122-131`) — scalar/ensemble TD |
| Critic representation | `C51Critic` by default (`config/models/sac.yaml`) |
| Actor objective | `StochasticActor.loss` (`actors/sac.py:109-117`): `-(q + stop_gradient(alpha)*entropy).mean()` |
| Actor sampling | Reparameterized tanh-Normal, `generate_action` (`actors/sac.py:83-98`) |
| Entropy | True per-sample log-density: `entropy = -log_prob.sum(...)` including the squash Jacobian and `log_act_scale` correction (`actors/sac.py:91-97`). `critic` param accepted, explicitly unused (docstring: "SAC's actor needs none to sample"). |
| Bootstrap source | `next_action, entropy = stop_gradient(actor.generate_action(next_obs, critic=self.critic, key=...))` (`policies/rl.py`'s `RLPolicy.train_step`) — no BoN, `critic=` is a no-op for this actor |
| Entropy in target | `MaxEntRLPolicy.entropy_in_target = True` (default) — `target_entropy_bonus = alpha * entropy` added to the bootstrap before discounting |
| Dual | `AutotuneRegulator` (`networks/regulator.py`), `target = chunk_size*action_dim*entropy_scale`, `entropy_scale = -1.0` (canonical SAC target entropy) |
| Cadence | critic every step; actor + `entropy_alpha` gated at `policy_delay` (`RLPolicy.optimizer_cadence`, `MaxEntRLPolicy.optimizer_cadence`) |
| Reference | Haarnoja et al. 2018 (SAC); action-chunk n-step backup follows Q-chunking (arXiv 2507.07969 §4.1) |
| Deviation | `log_std` is a state-independent parameter, not a network head (`actors/sac.py:26-27`, "barl's deliberate deviation from canonical SAC, kept as-is") |

## AMDP

| | |
|---|---|
| Critic objective | `QCriticBase.td_loss` — scalar/ensemble TD, same as SAC |
| Actor objective | `ScoreContinuousActor.loss` (`actors/amdp.py:167-195`) — **not** entropy-regularized policy improvement; direct score-matching regression against the critic's own action-gradient: `target = stop_gradient(sigma_tau * (grad_a Q(s,a) / alpha))`, `loss = 0.5*((u - target)^2).sum(-1)/sigma_tau` |
| Actor sampling | Controlled SDE integration (`euler_step`, `actors/amdp.py:145-165`), accumulates a path log-weight `log_w` |
| Entropy | `generate_action` returns `log_w` (path log-weight, `(B,1)`) as its second value **but `loss()` never uses it** — the actor's own objective is score-matching, not `-(q+alpha*H)`. `log_w` still feeds `target_shift`/`dual_loss` at the `RLPolicy.train_step` call site (it is a valid entropy quantity for the Bellman shift/dual, just not what this actor's own loss optimizes against). `critic` param accepted, unused in sampling. |
| Bootstrap source | Same `RLPolicy.train_step` bootstrap line as SAC; no BoN |
| Entropy in target | `entropy_in_target = True` (default, not overridden by AMDP) |
| Cadence | Same pattern as SAC |
| Reference | AMDP's own controlled-SDE actor-critic formulation; Q-chunking for the backup |
| Deviation | fp32 throughout (not bf16) — "AMDP went non-finite in bf16 on both DMC envs" (`actors/amdp.py:33-36`), a documented numerical-stability exception, not a modeling choice |

## DIME

| | |
|---|---|
| Critic objective | `QCriticBase.td_loss` — scalar/ensemble TD, same as SAC |
| Actor objective | `DISActor.loss` (`actors/dime.py:155-160`): `-(q + stop_gradient(alpha)*entropy).mean()`, same form as SAC |
| Actor sampling | Time-reversed diffusion sampler (DIS) — forward integration from a **fixed** Gaussian prior under a learned control (`actors/dime.py:129-153`), not a denoiser |
| Entropy | Accumulated path log-ratio between the controlled-forward and uncontrolled-backward kernel, `(B,1)` — "the entropy falls out of the integration rather than being read off a closed-form density" (`actors/dime.py:33-35`). The exact per-step update is documented in the class docstring (`actors/dime.py:37-49`). `critic` unused in sampling. |
| Bootstrap source | Same `RLPolicy.train_step` bootstrap line; no BoN |
| Entropy in target | `entropy_in_target = True` (default) |
| Cadence | Same pattern as SAC |
| Reference | DIME's own time-reversed diffusion sampler (`dime_am/diffusion/od/dis.py`, per docstring); Q-chunking for the backup |
| Deviation | fp32 (log-ratio accumulates); `adam_b1 = 0.5` override on this actor's own optimizer (not the usual 0.9) — "dime_am trains the sampler's nets with the critic's b1" (`actors/dime.py:59-61`) |

## SDAC

| | |
|---|---|
| Critic objective | `QCriticBase.td_loss` — scalar/ensemble TD |
| Actor objective | `DDPMActor.loss` (`actors/sdac.py:283-350`) — soft-DSM regression, softmax-weighted over `num_candidates` freshly-decoded candidates' critic value, **plus** a second loss `explore_loss` training `explore_raw` (its own optimizer key) |
| Actor sampling | Fixed-step DDPM ancestral sampler (`sample_raw`, `actors/sdac.py:218-228`) |
| Entropy | `noise_entropy()` (`actors/sdac.py:264-268`) — a **scalar** analytic entropy of the additive exploration-noise Gaussian only, "the entropy of a parameter, not of a draw" (unlike SAC/AMDP/DIME's per-sample `(B,1)` entropy) |
| **Bootstrap source — best-of-N, not a plain draw** | `generate_action` (`actors/sdac.py:275-281`) calls `self.best_of_n(obs, self.num_bon, critic=critic, key=...)` — an **argmax over `num_bon` reverse-chain draws scored by the *online* critic** (`best_of_n`/`pick_best_from`, `actors/sdac.py:243-253`). `RLPolicy.train_step`'s bootstrap line calls this same `generate_action` for `next_action`, so **SDAC's Bellman-target bootstrap already contains a real best-of-N maximization** — a different mechanism from, and unrelated to, `execution.py`'s eval-time `ExecutionSpec(mode="bon")` (see "Shared machinery" above: `execution` is never imported by any training path). This is SDAC's own intentional design (argmax-over-samples as an implicit max, not a leak of the newer inference-time BoN executor) — already flagged in-code at `actors/sdac.py:68-72` and `policies/rl.py`'s `MaxEntRLPolicy` docstring. **No code change recommended.** |
| Entropy in target | `entropy_in_target = False` for SDAC (set in its model YAML) — "SDAC's Bellman target carries no entropy bonus (its `entropy` is a scalar over the exploration parameter, not the per-sample quantity the target wants), so it sets this false and the dual trains purely off the actor's DSM loss" (`policies/rl.py`'s `MaxEntRLPolicy` docstring) |
| Cadence | critic every step; `actor`/`explore`/`entropy_alpha` at `policy_delay` |
| Reference | SDAC's own soft-DSM formulation; Q-chunking for the backup |
| Deviation | Two independent best-of-N draws per `train_step` — one inside `generate_action` (bootstrap), one inside `loss` (softmax-weighted DSM target) — documented as deliberately *not* fused (`actors/sdac.py:291-296`, points to a separate fusion measurement) |

## NCLQL

| | |
|---|---|
| Critic objective | `NoiseConditionedCritic.noise_td_loss` (`critics/nclql.py:36-61`) — TD at the clean noise level **plus** a smoothing/consistency regression at randomly-sampled noisier levels, `losses["critic"] = td_loss + smoothing_coef*smoothing_loss` |
| Critic representation | `NCLQLCritic` (scalar ensemble), `NCLQLC51Critic`, or `NCLQLHLGaussCritic` (`critics/nclql.py:73-93`) — interchangeable by `class:` since nothing in `NCLQL`/`NoiseConditionedCritic` assumes a scalar head |
| Actor | **None** — no `Actor` slot at all (`SLOTS = ("critic", "sigmas")`). Sampling is annealed Langevin dynamics on the critic's own action-gradient (`NCLQL.sample`, `policies/lql.py:102-139`), using `critic.expectation()` (not `.value()`) specifically to keep the pessimism term out of the sampler's gradient field (`policies/lql.py:88-100`'s `critic_score` docstring). |
| Entropy | **None** — no entropy estimate, no dual; `NCLQL.train_step` calls `noise_td_loss(...)` with `entropy_bonus` at its default (no target-entropy term at all) — "grad_a log pi_soft(a\|s) = grad_a Q(s,a) means the target score is already available from policy evaluation" (`policies/lql.py:22-27`) |
| Bootstrap source | `next_action, score_norm = self.sample(next_obs, keys[0])`, `stop_gradient`'d (`policies/lql.py:156-157`) — the Langevin chain's terminal draw, not any `Actor.generate_action` |
| Noise conditioning | `NoiseConditionedCritic.augment(act, level, target=False)` (`critics/nclql.py:63-68`) concatenates a learned time/level embedding onto the action before `q_function` — `Q(s, [a, embed(level)])`. The critic is queried at the clean level (`clean_level = sigmas.shape[0]-1`) for the TD target and at randomly-sampled noisier levels for the smoothing loss. |
| Cadence | critic (only optimizer key) every step |
| Reward scale | `reward = batch["chunk_return"] * self.reward_scale` (`reward_scale = 5.0`), a scalar rescale applied at the policy, not the objective |
| Reference | Ki et al. 2026, Noise-Conditioned Langevin Q-Learning (https://github.com/ku-dmlab/NCLQL/tree/main) |
| Deviation | None noted beyond the reward rescale (a run-config choice, not a math deviation from the paper) |

---

## Counters and reproducibility (cross-algorithm)

- `env_steps`: primitive actions executed, summed across environments and seeds
  (`trainer.py`, `log.advance({"env_steps": cfg["rollout_steps"]})` per `collect()`
  call). `random_steps` prefill is **not** included in `env_steps`/`grad_steps` —
  no `log.advance` wraps the prefill call in `trainer.py`; see the code comment
  added at that call site.
- `grad_steps`: optimizer updates. Cumulative-budget accounting
  (`grad_steps_this_iter = max(0, round(env_steps*utd_ratio) - grad_steps)`,
  guarded on `run.buffer.min_transitions() > 0`) makes fractional `utd_ratio`
  exact over many iterations and correctly waits when a chunk plan spans more
  than one `collect()` call — covered end-to-end by
  `tests/test_modularity.py::test_chunked_training_counts_primitive_steps`
  (parametrized `utd_ratio ∈ {0.5, 2.0}`, `chunk_size=4 > rollout_steps=2`).
- Resume restores `env_steps`/`grad_steps` via `run.progress` (`training_checkpoint.py`),
  plus per-seed replay contents/RNGs and the collector's own RNG. It does **not**
  restore `Collector.state` (env/obs/chunk_state/pending) — every resume restarts
  environment episodes and discards any pending (not-yet-emitted) action chunk.
  This is a documented, current limitation, not a bug.
- **Gamma/discount consistency**: `Collector.collect(..., discount=...)`'s three
  call sites in `trainer.py` (rollout, warmup, `_validate`) always pass
  `discount=<policy>.discount`, which is always `self.critic.gamma` (`RLPolicy.discount`,
  `NCLQL.discount`) — there is no code path where a live collection call
  accumulates a chunk return at a `gamma` different from the training critic's
  own. Across a **resume**, `training_checkpoint.validate_resume` already
  rejects any `model_overrides`/`snapshot` difference from the saved run
  ("Resume must use the saved {key}") — since `gamma` is a critic HParam
  resolved through that snapshot, a resumed run cannot silently train an
  already-accumulated replay buffer under a changed gamma. No new runtime
  check was added: the existing resume-config equality check already covers
  the directive's "changing gamma while reusing aggregated returns must fail"
  requirement, and an additional assertion at `Collector.collect`'s call
  sites would have nothing to check against but itself (both are always
  sourced from the same `<policy>.discount`) — see CLAUDE.md's "don't add
  error handling for scenarios that can't happen."

## What was audited and found already clear (no code change)

- SAC/AMDP/DIME actor `loss()`/`generate_action()` bodies (`actors/sac.py`,
  `actors/amdp.py`, `actors/dime.py`, read in full for this map) — entropy,
  score-matching targets, and SDE integration steps are already named variables
  with equations in their own docstrings.
- SDAC's best-of-N-in-bootstrap (documented above) — real, intentional, already
  flagged in-code.
- NCLQL's `augment`/smoothing-loss composition (`critics/nclql.py`) — already one
  readable method.
- `AutotuneRegulator`'s sign/coefficient — already exactly tested
  (`tests/test_primitives.py::test_regulator_dual_gradient_scales_with_alpha`).
- Gradient isolation (`frozen()`/`TRAINABLE`/`is_target`) — structurally sound;
  needs a *proving test* (added, see `tests/test_policies.py`), not new code.
- `execution.py`'s Best-of-N/Mean-of-N executors — already isolated from every
  training path (verified above).
- `learning_objective`/`SLOT_OBJECTIVES` tags (`policy.py`, `critics/base.py:34`,
  `critics/nclql.py:18`) — a config-consistency assertion
  (`tests/test_policies.py::test_critics_declare_supported_objectives`), not the
  directive's forbidden objective-dispatch hierarchy. Left as-is.

## What was renamed for clarity (this overhaul's actual code changes)

None of the following are numerical corrections — `collection.py`'s discounted
chunk-return accumulation was already correct before this overhaul. These are
renames and a required-argument tightening, made to satisfy "explicit at the
objective, not reconstructed from collector side effects":

- `QCriticBase.bellman_target`/`categorical_projection`/`td_loss`/
  `NoiseConditionedCritic.noise_td_loss`: `reward`→`chunk_return`,
  `term`→`terminated`, `shift`→`entropy_bonus`, `horizon=None`→required `horizon`
  (no default; K=1 must be passed explicitly).
- `RLPolicy.target_shift`/`MaxEntRLPolicy.target_shift`→`target_entropy_bonus`.
- `collection.py`'s `pending["rew"]`→`pending["chunk_return"]` and the emitted
  replay field `rew`→`chunk_return`. `acc["rew_*"]` (the raw per-primitive-step
  reward diagnostic feeding `avg_reward`/`reward_std` logging) is untouched —
  a distinct quantity from the accumulated chunk return.
- Shared discount arithmetic, previously duplicated between `bellman_target` and
  `categorical_projection`, factored into `n_step_discount(gamma, horizon,
  terminated, dtype)` (`critics/base.py`) — a plain function both call, not a
  hook or dispatch layer.
- `training_checkpoint.py`'s `VERSION` bumped `1`→`2` (a saved replay `.npz`'s
  field names changed); old training checkpoints are rejected with a clear
  error, not silently reinterpreted.
