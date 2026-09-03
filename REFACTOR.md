# Implementing the policy-template refactor

Step-by-step instructions for collapsing `SAC`/`AMDP`/`DIME`/`SDAC` into two configurable policy
classes and moving SDAC's best-of-N and exploration onto its actor. The reasoning behind each
decision is in the plan; this file is the execution order, the exact edits, and what to run after
each one.

**Ground rules.** Every stage ends green and is committable on its own. Do not start a stage before
the previous one's verification passes — several of them are only debuggable in isolation. Follow
CLAUDE.md's conventions throughout: no Python-level defaults on components, every tunable a required
keyword-only argument from `config/`, comments only where something is genuinely non-obvious.

---

## Step 0 — Prerequisite: finish the in-flight critic rename

The working tree renames `QCriticBase.td_loss` → `loss` and
`NoiseConditionedCritic.noise_conditioned_loss` → `loss` in the source but not in the tests. Stage 2
does the *same* rename on actors, so this has to land first or the two collide.

**0a. Update the stale call sites.**

| file | change |
|---|---|
| `tests/test_critics.py:84-91` | `test_td_loss_is_differentiable` → `test_loss_is_differentiable`; `critic.td_loss(` → `critic.loss(`; the two docstring/comment mentions |
| `tests/test_critic_stats.py:272` | `c.noise_conditioned_loss(` → `c.loss(` |
| `tests/test_critic_stats.py:293` | `c.td_loss(` → `c.loss(` |

That takes the suite from **11 failed** to **3 failed**.

**0b. Resolve the signature collision the rename created.** The remaining 3 failures are not stale
call sites. `NoiseConditionedCritic.loss` (`lql.py:69`) now *overrides* `QCriticBase.loss`
(`critic.py:181`) with an incompatible signature — it inserts `sigmas, key` before `losses`:

```python
QCriticBase.loss           (self, obs, action, next_obs, next_action, reward, term, losses, shift=0.0)
NoiseConditionedCritic.loss(self, obs, action, next_obs, next_action, reward, term, sigmas, key, losses, shift=0.0)
```

Before the rename these were distinct names and there was no override. This matters beyond the
tests: `RLPolicy.train_step` will call `critic.loss(..., losses, shift=...)` and the build gate is
`QCriticBase`, which `NCLQLCritic` satisfies — so a config pairing an NCLQL critic with `RLPolicy`
would pass validation and fail at runtime.

Two options. **Decide this before proceeding**, because Stage 2 names the actor entry point `loss`
by analogy with the critic:

- *Keep both named `loss`* (what the working tree does). Then split the contract test by family:
  in `tests/test_critics.py`, add `PLAIN = [n for n in CRITICS if not issubclass(get_component(n),
  NoiseConditionedCritic)]` and `NOISE_CONDITIONED = [...]` (the idiom already exists at
  `tests/test_critic_stats.py:24-29`), parameterise `test_loss_is_differentiable` over `PLAIN`, and
  add a sibling `test_noise_conditioned_loss_is_differentiable` over `NOISE_CONDITIONED` asserting
  the same three things through the wider signature. Use the **raw** `act`, not `critic_action(c,
  act)` — `augment()` is what widens the action to the inner critic's input, and doing both widens
  it twice (`dot_general` shape error, 40 vs 24).
- *Give the noise-conditioned variant its own name back.* Cleaner typing — it genuinely is a
  different contract — at the cost of partially reverting the rename.

Either way, **verify**: `uv run pytest tests/ -q` → `533 passed, 3 skipped`. That is the baseline
every later stage is measured against; anything else is new breakage.

---

## Step 1 — Characterize before moving anything

Land `tests/test_policies.py` against *current* code, parameterised over
`ConfigTree().available_models()`:

- each model builds via `Policy.make(name, FakeEnv(), env_stats=..., seed=0)`
- 3-4 `train_step`s produce finite losses
- `set(losses) == set(policy.create_optimizer())`
- `optimizer_labels()` covers every `TRAINABLE` leaf
- the **sorted set of metric keys** emitted under `recording(PRIMARY)`, asserted against a stored
  literal

Reuse the fixtures in `tests/conftest.py` and the `FakeEnv`/`make_env_stats` helpers from
`tests/test_policy_delay.py:29-40`.

The metric-key assertion is the one that earns its place: it turns every wandb series rename later in
this refactor into a diff you approve rather than a surprise. It also gives SDAC its first
behavioural coverage — today it is never constructed and never stepped by any test.

**Verify:** suite still `533 passed` plus the new tests.

---

## Step 2 — Split the model name from the class name

`type(self).__name__` currently does three jobs at once: the `config/models/*.yaml` lookup key, the
metric namespace, and checkpoint identity. One class serving three algorithms breaks all three, so
this has to come first.

**`src/barl_jax/config_tree.py`** — add next to `resolve_model_name` (`:199`):

```python
def policy_class_name(self, policy_name: str) -> str:
    return self.policy_cfg(policy_name).get("policy", policy_name)
```

Defaulting to the model name makes identity the rule and leaves `NCLQL` untouched. Keep it
YAML-only: `scripts/grid.py` loads this module *by path* to avoid importing `barl_jax`
(`grid.py:33-41`), so the mapping must not need the Python registry. A `policy:` key in a model block
is structurally safe — model blocks are read by explicit key, and `_META_KEYS` (`:80`) governs
component entries only.

**Do not** add `policy_class` to `resolved()`. That map is the wandb config and the snapshot payload;
a new key there makes every new snapshot differ from every old one in a field that also differs
between a live and a replayed tree. The raw `models:` block already carries `policy:` into the
snapshot.

**`src/barl_jax/policy.py`**:

- `Policy._class_for(name)` (new, shared by `make` and `load`): if `policy_cfg(name)` is non-empty,
  return `_classes[policy_class_name(name)]`; else if `name in _classes`, return it; else raise
  listing both registries. Model-name-first is what makes old sidecars work — `"policy_class":
  "SDAC"` takes branch one and resolves to `MaxEntRLPolicy`.
- `make` (`:272-281`): resolve the alias, look up via `_class_for`, pass `model_name=name` in rather
  than discarding it. Store the *resolved* name so `sac_big` → `SAC`. Assert it is non-None.
- `__init__` (`:291`): add `model_name: str = None`; set `self.model_name = model_name or
  type(self).__name__`. A plain `str` is static in nnx — it lands in the GraphDef (so SAC and AMDP
  get distinct jit cache entries, which is correct) and never enters the state tree that
  `multiseed.stack()` compares.
- Replace `type(self).__name__` with `self.model_name` at `:302` (`snapshot_for`), `:383`
  (`resolve_slot`), `:392` (`resolve_params`), and in the `build_component` assertion text
  (`:404-408`) and `_fallback_label`'s error — those are config-facing and should name the model.
- `algo_name()` (`:426-430`): `@classmethod` → instance method over `self.model_name`. All three
  callers are instances (`:357`, `:451`, `trainer.py:420`), so `sac/*`, `amdp/*`, `dime/*`, `sdac/*`
  survive.
- `save` (`:587`): write `{"model": self.model_name, "policy_class": type(self).__name__, ...}`.
  Keep the class name — it is the only record of which template produced the weights.
- `load` (`:601`) takes a model name and resolves via `_class_for`; `load_any` (`:629`) reads
  `meta.get("model") or meta["policy_class"]`.
- `Policy.available()` (`:283`) is dead code — no caller in `src/`, `scripts/` or `tests/`. Delete it
  or repoint at `available_models()`. Do not leave it returning `["RLPolicy", "MaxEntRLPolicy",
  "NCLQL"]`, which reads as an answer and is not one.

**`config/models/*.yaml`** — add `policy: <SameName>` to all five blocks as an identity mapping, so
the mechanism is exercised before anything depends on it. Document it in `aliases.yaml`'s header,
where the model-name-vs-policy distinction already lives.

**Verify:** suite; `Policy.load_any` on a real `checkpoints/sdac/.../latest`;
`uv run python scripts/print_config.py` for each algo; `uv run python scripts/lint_config.py`;
`uv run python scripts/submit.py --algos SAC AMDP DIME SDAC NCLQL --envs HalfCheetah --executor local --validate`.

---

## Step 3 — Make replayed snapshots overlay instead of replace

**Do not skip this.** 826 artifacts under `checkpoints/` and `sweeps/` carry an embedded `snapshot:`,
and `resolve_params`/`resolve_slot` serve a snapshot's `resolved:` map **wholesale** when replaying
(`config_tree.py:312-317`, `:345-347`). Later stages add *required* params (`policy_delay`,
`normalize_obs`, `entropy_in_target`) and *required* actor kwargs (`num_bon`, `softmax_temp`,
`explore_*`, `q_score_max_*`). Replaying an old snapshot would then hand `MaxEntRLPolicy` a params
dict with no `policy_delay` (KeyError) and `DDPMActor.__init__` a kwargs dict with no `num_bon`
(TypeError) — at construction, on the compute node, after the GPU is allocated.

The machinery to fix it is already in place: `ConfigTree.__init__` (`:157-159`) deep-merges the
snapshot's raw `presets`/`components`/`models` blocks **over** the live files, and `slot_class`
(`:228`) and `slot_size` (`:276`) already short-circuit to the snapshot. So the live layering under a
snapshot already yields the union. Turn replay into an overlay:

```python
def resolve_params(self, policy_name, profile):
    live = merge(self.params_layers(policy_name, profile))
    return {**live, **self._replayed("params.")} if self.replaying else live
```

and the same shape in `resolve_slot`, keeping `class`/`size` popped from the replayed kwargs.

The guarantee this weakens is narrow and statable, and belongs in the docstring: **a key the snapshot
recorded is authoritative; a key it never recorded takes today's value.**
`tests/test_config_tree.py:127-143` (edit `sizes.yaml`, assert the replay is unchanged) still passes,
because `latent_dim` *is* in the snapshot — which is the proof that the overlay is safe for recorded
keys. A stray old key (`params.num_bon` after step 5) overlays harmlessly: params are read by
explicit key and are not signature-checked.

**Verify:** copy a real pre-refactor snapshot into `tests/fixtures/` and assert
`ConfigTree(snapshot=<it>).resolve_slot("SDAC", "actor", None)` succeeds — now, and again after step 5.

---

## Step 4 — One actor contract

`Actor` (`policy.py:217-229`) is deliberately contract-free today, which is what let four actors
diverge into four calling conventions. Give it:

```python
class Actor(Component, nnx.Module, ABC):
  @abstractmethod
  def generate_action(self, obs, *, critic: Critic = None, key=None) -> tuple:
      """(action, entropy): canonical [-1,1] action, one row per obs row, and the
      differential entropy of that draw, shaped (B, 1) or scalar."""

  @abstractmethod
  def loss(self, obs, critic: Critic, alpha, *, key=None):
      """This actor's scalar objective at `obs`, priced against a frozen critic."""

  def extra_losses(self, obs, critic: Critic, alpha, *, key=None) -> dict:
      return {}

  def sample(self, obs, *, critic: Critic = None, key=None):
      return self.generate_action(obs, critic=critic, key=key)[0]
```

`loss` **returns and never mutates** — this is forced, not stylistic: `gated_loss` (`:118-132`) runs
it under `nnx.cond`, and both branches must hand back one matching pytree.

`extra_losses` is how SDAC expresses its second objective without a second convention. Document two
invariants on it: the keys must be a subset of this actor's `create_optimizer()` keys, and **extra
losses are never gated** — they run outside `gated_loss`, because `optimizer_cadence` does not list
them.

Convert one actor at a time, policies unchanged:

| actor | change |
|---|---|
| `StochasticActor` (`sac.py:94`) | `loss` stops mutating `losses` and stops returning entropy; returns the scalar. **Delete its `metrics.vec("action", ...)` and `metrics.dist("entropy", ...)` calls** (`:105-106`) — the template records both, and the `Recorder`'s `setattr` makes a double-write last-wins. SAC's `train_step` loses a line. |
| `DISActor` (`dime.py:148`) | `actor_loss` → `loss`; `generate` → `generate_action`; drop the `sample` shim |
| `ScoreContinuousActor` (`amdp.py:159`) | `actor_loss` → `loss`; move `max_norm`/`percentile` from the call site to `__init__` as required kwargs named `q_score_max_norm`/`q_score_max_percentile`, with the values moving from `config/models/amdp.yaml`'s `params:` to the `ScoreContinuousActor:` entry in `config/components/actors.yaml` |
| `DDPMActor` (`sdac.py:197`) | `loss` returns the scalar instead of `None`; `sample` → `sample_raw` |

**Verify after each actor:** suite plus step 1's characterization unchanged. This stage should change
nothing numerically.

---

## Step 5 — Move SDAC's best-of-N and exploration onto `DDPMActor`

Keep `class SDAC(Policy)` for this stage. It is a pure code move, and it is the only stage that is
**provably bit-identical** — which is worth having as a checkpoint before the template lands.

Migrating from `sdac.py`: `explore_raw` :274, `explore_std` :278, the `after_step` clamp :283,
`noise_entropy` :288, `_tile` :292, `score` :296, `pick_best_from` :299, `pick_best` :304,
`best_of_n` :307, `explore` :311, `act` :315. The `Component.after_step` hook already exists and the
Policy already fans it out (`policy.py:486-489`).

Resulting shape on the actor:

```python
def sample_raw(self, obs, key)                       # reverse chain, unsquashed x0
def best_of_n(self, obs, n, *, critic, key)          # tile, chain, score, argmax -> raw x0
def explore(self, action, key)
def generate_action(self, obs, *, critic=None, key=None) -> (action, noise_entropy)
def loss(self, obs, critic, alpha, *, key=None) -> scalar
def extra_losses(self, obs, critic, alpha, *, key=None) -> {"explore": scalar}
def param_labels(self, subpath) -> str
def after_step(self)
```

With `critic=None`, `best_of_n` degenerates to `n=1` — a single chain draw needs no scoring. That is
what keeps the duck-typed sampler contract tests in `tests/test_actors.py` working.

`__init__` gains eight required keyword-only kwargs: `num_bon`, `num_candidates`, `softmax_temp`,
`explore_init`, `explore_min`, `explore_max`, `explore_raw_limit`, `explore_lr`. Move them from
`config/models/sdac.yaml`'s `params:` into the `DDPMActor:` entry in
`config/components/actors.yaml` — **with their comments**, the `num_bon` cost note and the
`num_candidates` ESS note both carry real measurement history.

```python
def create_optimizer(self):
    return {"actor": adam(self.lr, self.adam_b1, self.grad_clip, self.grad_clip_mode),
            "explore": optax.adam(self.explore_lr)}

def param_labels(self, subpath):
    return "explore" if str(subpath[0]) == "explore_raw" else "actor"
```

Make `param_labels` **total**, not `None`-returning: the `None` fallback goes to
`default_key.setdefault(slot, key)` (`policy.py:523-525`), whose value depends on
`create_optimizer()`'s dict insertion order — a correctness property hiding in a literal's key order.

Keep `optax.adam(self.explore_lr)` verbatim. Switching to `networks/optim.adam` would add a grad clip
and an `adam_b1` SDAC has never trained with; that is a tuning change and belongs in its own commit
(it would also fix a hidden `b1=0.9` Python default in a repo whose whole point is that there are
none).

`SDAC.policy_params` (`:251`) goes away and the mechanism loses its only user. **Keep the
mechanism** — `policy_params` and `_fallback_label` (`:540-548`) are one unit, and deleting the dict
deletes the loud error that turns an unowned parameter into a failure rather than a silently
untrained one. Note in the docstring that nothing currently uses it.

**Keep the fused pass in this stage.** `SDAC.train_step` continues to call
`self.actor.sample_raw(joint_obs, key)` and `self.actor.pick_best_from(...)` directly over
`concat([tile(next_obs, N), tile(obs, N)])`, and the 4-way key split is untouched. That is what makes
this stage bit-identical.

**The metric renames land here.** Once the actor owns two optimizer keys, `metrics_key`'s `shared`
branch (`policy.py:446-454`) flips:

```
actor/loss              ->  actor/loss/actor          (same for grad_norm, grad_max, grad_frac_*)
sdac/loss/explore       ->  actor/loss/explore
sdac/explore_std        ->  actor/explore_std
sdac/noise_entropy      ->  actor/noise_entropy
```

Existing SDAC dashboard panels go **empty, not red**. `trainer.py:420`'s register loop covers both
namespaces, so nothing breaks operationally. Approve the diff that step 1's metric-key test shows you.

**Verify:** the A/B harness below, expecting *identical* state leaves.

---

## Step 6 — Land the template, one algorithm at a time

Create `src/barl_jax/policies/rl.py`.

`RLPolicy(Policy)` — slots `critic` (gated `QCriticBase`) and `actor` (gated `Actor`); params
`policy_delay`, `normalize_obs`; `optimizer_cadence() -> {"actor": self.policy_delay}`. One
`train_step` with three one-line seams that the base answers trivially:

```python
def alpha(self):                        return jnp.asarray(1.0)
def target_shift(self, alpha, entropy): return 0.0
def dual_loss(self, entropy, losses):   pass

def train_step(self, batch):
    obs, next_obs = self._obs(batch["obs"]), self._obs(batch["next_obs"])
    losses, alpha = {}, self.alpha()
    keys = jax.random.split(self.rngs.noise(), 3)

    next_action, entropy = jax.lax.stop_gradient(
        self.actor.generate_action(next_obs, critic=self.critic, key=keys[0]))

    self.critic.loss(obs, batch["act"], next_obs, next_action, batch["rew"], batch["term"],
                     losses, shift=self.target_shift(alpha, entropy))

    losses["actor"] = gated_loss(self, self.policy_delay,
                                 lambda a, c: a.loss(obs, c, alpha, key=keys[1]),
                                 self.actor, self.critic)
    losses.update(self.actor.extra_losses(obs, self.critic, alpha, key=keys[2]))
    self.dual_loss(entropy, losses)

    self.actor.metrics.vec("action", next_action, limit=1.0)
    self.actor.metrics.dist("entropy", entropy)
    return losses
```

`MaxEntRLPolicy(RLPolicy)` overrides `__init__` (builds the `entropy` slot at `target = action_dim *
entropy_scale`; params `entropy_scale`, `entropy_in_target`), `optimizer_cadence` (adds
`"entropy_alpha": self.policy_delay`), and the three seams — `self.entropy.value()`,
`alpha * entropy if self.entropy_in_target else 0.0`, `self.entropy.loss(entropy, losses)`.

Add at construction: `assert set(self.optimizer_cadence()) <= set(self.create_optimizer())`.
`build_optimizer` does `cadence.get(key, 1)`, so an unmatched cadence key is dropped without a word —
and `"entropy_alpha"` derives from the *regulator's slot name*, so a model naming that slot anything
but `entropy` would silently ungate the dual.

Config, one algorithm per commit — SAC, then DIME, then AMDP, then **SDAC last** (it is where the
computation genuinely changes):

| | policy_delay | normalize_obs | entropy_scale | entropy_in_target |
|---|---|---|---|---|
| SAC | 1 | true | -1.0 | true |
| AMDP | 3 | false | -1.0 | true |
| DIME | 3 | true | -4.0 | true |
| SDAC | 1 | true | -0.9 | **false** |

Switch each model's `policy:` to `MaxEntRLPolicy` and delete its class. `policy_delay: 1` makes
`update_due` return python `True`, so `gated_loss` degenerates to a direct call and SAC/SDAC's path
is structurally unchanged.

Honour `normalize_obs` (`self._obs = self.normalize_obs if flag else (lambda x: x)`) even though it
is numerically the identity for everyone today — `obs_mean`/`obs_std` (`policy.py:343-344`) are
created and **never written anywhere in the repo**, so AMDP's documented exemption has no effect
*yet*. Deleting the flag would silently change AMDP the day someone wires the running stats up. File
a follow-up to either wire them up or delete both.

**Verify per algorithm:** the A/B harness. Expect AMDP and DIME bit-identical; SAC to differ only in
`entropy.log_alpha` and its downstream; SDAC to differ by construction.

---

## Step 7 — Move the actors, delete the old modules

```
src/barl_jax/networks/actors/__init__.py
                            stochastic.py  StochasticActor                    (from sac.py)
                            score.py       ScoreContinuousActor, clip_score   (from amdp.py)
                            dis.py         DISActor, _inv_softplus            (from dime.py)
                            ddpm.py        DDPMActor, alpha_bar_from_betas    (from sdac.py)
```

`networks/critic.py` already holds critics *with their losses*; once an actor is "pure architecture
plus its own objective" it is the same kind of object. `stochastic.py` not `gaussian.py` — that name
is taken by the log-prob helpers.

- Delete `policies/{sac,amdp,dime,sdac}.py`.
- `policies/__init__.py` → import `rl` and `lql` only.
- `barl_jax/__init__.py` → add `import barl_jax.networks.actors` next to the `networks.critic`
  import. **This import becomes genuinely load-bearing** — nothing else pulls the actor modules in
  any more, so extend the existing "unused-import cleanup would silently break that" comment to say
  so.
- `tests/test_actors.py:14` → `from barl_jax.networks.actors.ddpm import DDPMActor`.

**Then measure the fused pass.** With BoN on the actor, `generate_action(next_obs)` and `loss(obs)`
each do their own — identical FLOPs, but ~130 extra kernel launches for the 20-step unrolled chain
plus one extra critic invocation. Fitting `c + kN` to the measurements recorded in `sdac.yaml`
(N=32 → 72.5 ms, N=8 → 25.9 ms at 3 seeds, batch 256) gives k ≈ 1.94 ms, c ≈ 10.4 ms, so the
N-proportional work at N=8 is ~15.5 ms of a 25.9 ms step and none of it is added or removed by
splitting. Expected cost **0-3%**.

Time a blocked `seeded_step` for SDAC at 3 seeds against the current baselines — **19.16 ms**
(HalfCheetah) and **22.15 ms** (DogRun). If the split costs more than ~5%, add the pre-designed
fallback rather than redesigning: an optional `Actor.train_context(obs, next_obs, *, critic, key)`
returning `None` by default, threaded into `generate_action`/`loss` as `context=`, with `DDPMActor`
returning the fused joint pass's two halves. Pay for that extra parameter on four actors only if the
measurement says so.

---

## Step 8 — Close the lint gaps

`scripts/lint_config.py` imports `barl_jax` and so can see the registry, but never touches
`Policy._classes` today. Add:

- **every model block's `policy:` names a registered `Policy`.** Nothing checks this now, so a model
  naming a missing class passes every test and every launcher pre-flight, then fails at
  `trainer.setup` on the compute node seconds after the GPU is allocated. Note `submit.py`'s
  `check_config` deliberately does not catch it either — `model_config.validate` never consults the
  policy registry.
- **a model's declared slots match its policy class's slots.** Add `SLOTS: tuple[str, ...]` to each
  Policy. A slot the policy builds but the YAML omits already raises at `build_component`; an *extra*
  slot in the YAML is silently resolved into the snapshot, the wandb config and the config hash with
  no component behind it — and two models sharing a class makes that hole easier to fall into.

Update CLAUDE.md's policy section (it still describes `SACPolicy` and `src/barl/`) and the model YAML
headers.

---

## The A/B harness

Not committed — a scratch script used at steps 5 and 6.

```python
# 1. git stash the change; build the OLD policy at seed s; snapshot its state
# 2. git stash pop; build the NEW one at the same seed
# 3. feed identical pre-drawn batches through barl_jax.train.train_step
# 4. compare nnx.to_flat_state leaves, report max abs diff per path
```

**Control the RNG out of the comparison.** Replace `policy.rngs.noise` on both instances with a
counter-based generator over a fixed key list. The template splits three keys where the old policies
split two, and `split(k, 3)[:2] != split(k, 2)` — without this every comparison shows spurious
divergence. Do not contort the template to preserve the key stream; control it in the harness.

Two practical traps, both hit during a previous A/B in this repo:

- `donate_argnums` on the jitted step **deletes** the input state, so keep an on-device copy via
  `jax.tree.map(jnp.copy, state)` before stepping. Going through numpy fails: the state contains
  typed PRNG keys, which raise on `np.asarray`.
- Filter typed-key leaves out of the numeric comparison (`jax.dtypes.issubdtype(x.dtype,
  jax.dtypes.prng_key)`) or convert them with `jax.random.key_data`.

Expected results, with matched keys:

| | expectation |
|---|---|
| **AMDP, DIME** | bit-identical. They already feed the bootstrap entropy to the dual and already gate the actor loss; nothing in the template differs. Any mismatch is a porting bug. |
| **SAC** | identical except `entropy.log_alpha` and everything downstream, because the dual now tracks entropy at `next_obs` instead of `obs`. The test is that the divergence is confined to that leaf on step 1. |
| **SDAC** | bit-identical at the end of step 5. After step 6 the split pass makes it non-identical by construction — compare distributionally instead: mean/std of each loss over 200 steps, plus the `explore_std` and `alpha` trajectories. |

Note bfloat16 amplifies benign reassociation: a difference that is 1e-11 in float32 shows as ~2.5e-5
in the bf16 the nets actually use. Judge "identical" against the float32 run when in doubt.

---

## New permanent tests

- **`tests/test_actor_contract.py`**, over every registered `Actor`: `generate_action` returns finite
  `(B, act_dim)` actions and a finite `(B,1)`-or-scalar entropy; `sample(obs, key) ==
  generate_action(obs, key)[0]`; key determinism and key sensitivity; `set(extra_losses(...)) <=
  set(create_optimizer())`; and — the one nothing currently pins — **`nnx.grad` of `actor.loss(obs,
  critic, alpha)` with respect to the critic is all-zero.** That is the `frozen()` contract. With the
  loss behind an ABC that any future actor implements, it is the most valuable new assertion in the
  suite: `multi_transform` partitions by param, so a leaked critic gradient would silently *add* to
  the critic's own regression rather than erroring.
- **`tests/test_policies.py`** — step 1's characterization, kept.
- **`tests/test_sdac_actor.py`** — BoN picks the argmax candidate against a stub critic with a known
  ranking; `explore_std` stays inside `[explore_min, explore_max]` for extreme `explore_raw`;
  `after_step` clamps to `±explore_raw_limit`; `explore` output stays in `[-1, 1]`; `noise_entropy`
  matches the closed-form Gaussian entropy at a known std; `optimizer_labels()` routes
  `actor.explore_raw` → `"explore"` and `actor.field.*` → `"actor"`.
- **`tests/test_policy_delay.py`** — `DELAYED = ["AMDP", "DIME"]` and `Policy.make(name)` need no
  change, they are model names. Add `UNDELAYED = ["SAC", "SDAC"]` asserting the actor moves every
  step. Note `:104` reads `policy.policy_delay` and `:112-114` reaches `nnx.merge(...)[0].actor` —
  both survive, but they are the couplings to watch.
- **`tests/test_config_tree.py`** — every model's `policy:` names a registered Policy;
  `Policy.make(m).model_name == m`; `algo_name() == m.lower()` for all five.
- **Checkpoint compatibility** — `Policy._class_for("SDAC") is MaxEntRLPolicy`; one real save/load
  round-trip in `tmp_path`; and the step-3 old-snapshot fixture.

---

## Things that will change and should not surprise you

1. **`meta/config_hash` changes for every model.** Moving params changes the resolved key strings
   (`params.num_bon` → `components.actor.num_bon`) that `trainer.py:333-338` digests. "Same config,
   different seed" grouping in wandb will not span the refactor.
2. **Sweep-axis renames.** `params.{num_bon, num_candidates, softmax_temp, explore_*,
   q_score_max_*}` all become `components.actor.*`. `validate()` rejects an override naming a
   nonexistent param, but validation runs at `trainer.setup`, not at `grid.py` expansion — so an
   archived grid spec fails *per task, after submission*. Only one archived sweep config carries
   `model_overrides` at all; the exposure is future hand-written specs.
3. **`algo_name()` silently becoming the class name** if any construction path forgets `model_name`.
   Every metric re-namespaces to `maxentrl/*` and `log.register` registers the wrong namespace, so
   metrics keep logging against the wrong step key. The assert in `make` and step 1's metric-key test
   are the two guards.
4. **Entropy shape asymmetry.** SAC/AMDP/DIME return `(B,1)`; SDAC's `noise_entropy()` is a scalar.
   `AutotuneRegulator.loss` means it and `shift=` is unused for SDAC, so both work — but
   `actor/entropy/std` becomes a constant 0 for SDAC. Consider broadcasting to `(B,1)`.
5. **`Policy._classes` no longer contains `"SAC"`/`"AMDP"`/`"DIME"`/`"SDAC"`.** No in-repo caller
   beyond `load`, but an ad-hoc script or notebook indexing it directly will break.
