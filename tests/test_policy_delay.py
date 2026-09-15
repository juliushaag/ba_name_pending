"""Loss groups follow optimizer cadences without changing state when logged.

The conditional must survive the seed vmap, skipped optimizer transforms must
preserve Adam state, and extra diagnostic evaluation must not advance actor
batch statistics or RNG streams.
"""

import gymnasium as gym
import jax
import jax.extend
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import ACT_DIM, BATCH, OBS_DIM, make_env_stats
from flax import nnx

from barl_jax.components import TRAINABLE
from barl_jax.metrics import DEBUG, PRIMARY, recording
from barl_jax.optimization import gated_losses, train_step, update_due
from barl_jax.policy import Policy

DELAYED = ["AMDP", "DIME"]


class FakeEnv:
    """Spaces only: every Policy here takes its dimensions from `env_stats`, and
    reads the spaces solely to assert the action space is a Box."""
    observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)


def make(name: str) -> Policy:
    return Policy.make(name, FakeEnv(), env_stats=make_env_stats(), seed=0)


def batch():
    rng = np.random.default_rng(0)
    f = lambda *s: jnp.asarray(rng.normal(size=s), jnp.float32)
    return {"obs": f(BATCH, OBS_DIM), "next_obs": f(BATCH, OBS_DIM),
            "act": jnp.clip(f(BATCH, ACT_DIM), -1, 1),
            "chunk_return": f(BATCH), "term": jnp.zeros((BATCH,), jnp.float32),
            "horizon": jnp.ones((BATCH,), jnp.int32)}


def test_update_due_matches_the_optimizers_own_predicate():
    """`conditionally_mask(tx, lambda step: (step + 1) % k == 0)` counts the same
    gradient steps `train_steps` does, so the two must agree step for step."""
    policy = make("AMDP")
    for step in range(9):
        policy.train_steps[...] = jnp.asarray(step, jnp.int32)
        assert bool(update_due(policy.train_steps[...], 3)) == ((step + 1) % 3 == 0)


def test_update_due_does_not_depend_on_recording():
    """Diagnostics must not change which optimizers are due."""
    policy = make("AMDP")
    policy.train_steps[...] = jnp.asarray(0, jnp.int32)   # not a multiple of 3
    assert update_due(policy.train_steps[...], 3) is not True
    with recording(PRIMARY):
        assert not bool(update_due(policy.train_steps[...], 3))


def test_gated_losses_returns_zero_off_cadence_and_the_value_on():
    policy = make("AMDP")
    def fn(module):
        return {"actor": jnp.asarray(2.5)}

    for step, due in [(0, False), (1, False), (2, True)]:
        policy.train_steps[...] = jnp.asarray(step, jnp.int32)
        value = float(gated_losses(policy, fn, ("actor",), policy.critic)["actor"])
        assert value == pytest.approx(2.5 if due else 0.0)


@pytest.mark.parametrize("name", DELAYED)
@pytest.mark.parametrize("tiers", [(), (PRIMARY,), (PRIMARY, DEBUG)],
                         ids=["unarmed", "primary", "debug"])
def test_delayed_policies_step_under_every_tier(name, tiers):
    """The unarmed variant is the one that actually branches -- and the one
    `--validate` never reaches, since it arms both tiers."""
    policy = make(name)
    optimizer = nnx.Optimizer(policy, policy.build_optimizer(lr_schedule=None), wrt=TRAINABLE)
    graphdef, state = nnx.split((policy, optimizer))

    data = batch()
    for _ in range(4):
        state, losses, _, _ = train_step(graphdef, state, data, tiers)
        assert all(np.isfinite(np.asarray(v)) for v in losses.values())


@pytest.mark.parametrize("name", DELAYED)
def test_the_actor_only_moves_on_its_cadence(name):
    """The whole point: the skipped steps are also the steps the optimizer would
    have masked, so the parameters follow exactly the trajectory they did before
    the loss was gated."""
    policy = make(name)
    optimizer = nnx.Optimizer(policy, policy.build_optimizer(lr_schedule=None), wrt=TRAINABLE)
    graphdef, state = nnx.split((policy, optimizer))

    data = batch()
    delay = policy.policy_delay
    moved = []
    for step in range(2 * delay):
        before = jax.tree.leaves(nnx.state(nnx.merge(graphdef, state)[0].actor, nnx.Param))
        state, _, _, _ = train_step(graphdef, state, data, ())
        after = jax.tree.leaves(nnx.state(nnx.merge(graphdef, state)[0].actor, nnx.Param))
        moved.append(any(not np.array_equal(np.asarray(a), np.asarray(b))
                         for a, b in zip(before, after)))

    assert moved == [(step + 1) % delay == 0 for step in range(2 * delay)]


@pytest.mark.parametrize("name", DELAYED)
def test_the_gate_survives_the_seed_vmap(name):
    """The trainer vmaps the step over seeds. `jax.vmap` lowers a `lax.cond` on a
    *batched* predicate into a select over both branches -- correct, and exactly
    as expensive as not gating at all -- so the gradient-step index has to reach
    the predicate unbatched (`optimization.gating`, `in_axes=None`). If it ever goes
    back to being read off the per-seed `train_steps` buffer, the `cond`
    disappears from the jaxpr and the saving silently goes with it.
    """
    seeds = 2
    policies = []
    for i in range(seeds):
        p = make(name)
        policies.append((p, nnx.Optimizer(p, p.build_optimizer(lr_schedule=None), wrt=TRAINABLE)))

    from barl_jax.multiseed import stack
    graphdef, state = stack(policies)

    seeded = jax.vmap(train_step, in_axes=(None, 0, 0, None, None))
    data = {k: jnp.broadcast_to(v, (seeds, *v.shape)) for k, v in batch().items()}

    jaxpr = jax.make_jaxpr(seeded, static_argnums=(0, 3))(
        graphdef, state, data, (), jnp.asarray(0, jnp.int32))
    primitives = _primitives(jaxpr.jaxpr)
    assert "cond" in primitives, f"the policy_delay branch was flattened away: {sorted(primitives)}"


def _primitives(jaxpr, seen=None) -> set:
    """Every primitive name in `jaxpr`, recursing through nested jaxprs -- the
    whole step arrives wrapped in one `jit` equation, so the top level is
    two names deep in nothing."""
    seen = set() if seen is None else seen
    for eqn in jaxpr.eqns:
        seen.add(eqn.primitive.name)
        for sub in jax.extend.core.jaxprs_in_params(eqn.params):
            _primitives(sub, seen)
    return seen


class StatefulLoss(nnx.Module):
    def __init__(self):
        self.calls = nnx.BatchStat(jnp.asarray(0, jnp.int32))
        self.weight = nnx.Param(jnp.asarray(2.0))
        self.rngs = nnx.Rngs(noise=0)

    def loss(self):
        from barl_jax.metrics import Recorder
        self.calls[...] += 1
        self.rngs.noise()
        Recorder(self).scalar("loss_calls", self.calls[...])
        return {"actor": self.weight[...] ** 2}


def test_logged_skipped_loss_preserves_module_state():
    policy = make("AMDP")
    module = StatefulLoss()
    before = jax.tree.map(_numpy_leaf, nnx.state(module))
    with recording(PRIMARY):
        losses = gated_losses(policy, lambda m: m.loss(), ("actor",), module)
    assert float(losses["actor"]) == 4.0
    from barl_jax.metrics import Metric
    nnx.pop(module, Metric)
    after = jax.tree.map(_numpy_leaf, nnx.state(module))
    for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(after)):
        np.testing.assert_array_equal(a, b)


def test_grouped_loss_runs_when_another_optimizer_is_due():
    # actor's own cadence (2) is not due at step 2, but explore's (3) is --
    # the group must still run, and "actor" gets a real value, not the zero
    # fallback, from that one shared evaluation.
    policy = MixedCadencePolicy()
    policy.train_steps[...] = jnp.asarray(2)
    module = StatefulLoss()
    def loss(m):
        return {**m.loss(), "explore": m.weight[...] ** 2}
    values = gated_losses(policy, loss, ("actor", "explore"), module)
    assert float(values["actor"]) == 4.0
    assert int(module.calls[...]) == 1


def _numpy_leaf(value):
    if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
        value = jax.random.key_data(value)
    return np.array(value, copy=True)


def _assert_state_equal(left, right, *, exact=False):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right)):
        a, b = _numpy_leaf(a), _numpy_leaf(b)
        if exact or not np.issubdtype(a.dtype, np.inexact):
            np.testing.assert_array_equal(a, b)
        else:
            np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-6)


def test_logging_does_not_change_training_state_with_stateful_actor():
    from barl_jax.config.model import ModelSpec
    from barl_jax.config.run import RunConfig
    run = RunConfig.from_dict({
        "model": "SAC", "env": "HalfCheetah",
        "model_overrides": {
            "params": {"policy_delay": 3},
            "components": {
                "actor": {"latent_dim": 8, "mlp": {
                    "norm": "batchrenorm", "dtype": "float32",
                    "norm_kwargs": {"momentum": 0.99, "epsilon": 1e-5,
                                   "r_max": 3.0, "d_max": 5.0, "warmup_steps": 0},
                    "dropout": 0.2}},
                "critic": {"latent_dim": 8, "mlp": {"dtype": "float32"}},
            },
        },
    })
    spec = ModelSpec.resolve("SAC", run)
    policy = Policy.make("SAC", FakeEnv(), env_stats=make_env_stats(), seed=0, spec=spec)
    optimizer = nnx.Optimizer(policy, policy.build_optimizer(lr_schedule=None), wrt=TRAINABLE)
    graphdef, initial = nnx.split((policy, optimizer))
    unlogged = jax.tree.map(jnp.copy, initial)
    logged = jax.tree.map(jnp.copy, initial)
    data = batch()
    for step in range(6):
        unlogged, *_ = train_step(graphdef, unlogged, data, (), jnp.asarray(step))
        logged, *_ = train_step(graphdef, logged, data, (PRIMARY, DEBUG), jnp.asarray(step))
        _assert_state_equal(unlogged, logged)


def test_optimizer_mask_freezes_adam_moments_after_a_real_update():
    policy = make("AMDP")
    tx = policy.build_optimizer(lr_schedule=lambda step: 1.0 / (step + 1))
    params = nnx.state(policy, TRAINABLE)
    state = tx.init(params)
    grads = jax.tree.map(jnp.ones_like, params)
    for step in range(6):
        before = state
        updates, state = tx.update(grads, state, params)
        for key in ("actor", "entropy_alpha"):
            # The outer schedule and mask counters advance every gradient step;
            # the inner Adam transform advances only on this partition's update.
            old_mask = before.inner_states[key].inner_state[0]
            new_mask = state.inner_states[key].inner_state[0]
            assert int(new_mask.step) == step + 1
            if (step + 1) % 3:
                _assert_state_equal(old_mask.inner_state, new_mask.inner_state, exact=True)
            else:
                assert any(not np.array_equal(_numpy_leaf(a), _numpy_leaf(b))
                           for a, b in zip(jax.tree.leaves(old_mask.inner_state),
                                           jax.tree.leaves(new_mask.inner_state)))
        if (step + 1) % 3:
            for component in ("actor", "entropy"):
                assert all(np.count_nonzero(_numpy_leaf(x)) == 0
                           for x in jax.tree.leaves(updates[component]))


def test_gate_context_restores_outer_step():
    from barl_jax.optimization import gating
    policy = make("AMDP")
    module = StatefulLoss()
    with gating(jnp.asarray(2)):
        with pytest.raises(RuntimeError), gating(jnp.asarray(0)):
            values = gated_losses(policy, lambda m: m.loss(), ("actor",), module)
            assert float(values["actor"]) == 0.0
            raise RuntimeError("inner trace failed")
        values = gated_losses(policy, lambda m: m.loss(), ("actor",), module)
        assert float(values["actor"]) == 4.0
    values = gated_losses(policy, lambda m: m.loss(), ("actor",), module)
    assert float(values["actor"]) == 0.0


class MixedCadencePolicy(nnx.Module):
    def __init__(self):
        self.train_steps = nnx.Variable(jnp.asarray(0, jnp.int32))

    def optimizer_cadence(self):
        return {"actor": 2, "explore": 3}

    def create_optimizer(self):
        return {"actor": None, "explore": None}


def test_loss_group_uses_union_of_distinct_delayed_cadences():
    policy = MixedCadencePolicy()
    module = StatefulLoss()
    def loss(m):
        return {**m.loss(), "explore": m.weight[...] ** 2}
    observed = []
    for step in range(6):
        policy.train_steps[...] = jnp.asarray(step)
        values = gated_losses(policy, loss, ("actor", "explore"), module)
        observed.append(float(values["actor"]) != 0)
    assert observed == [False, True, True, True, False, True]
    assert int(module.calls[...]) == 4


def test_seed_vmap_keeps_the_loss_gate_conditional():
    from barl_jax.multiseed import stack
    from barl_jax.optimization import gating
    policy = MixedCadencePolicy()
    graphdef, state = stack([StatefulLoss(), StatefulLoss()])
    def step_one(state, step):
        module = nnx.merge(graphdef, state)
        with gating(step):
            values = gated_losses(policy, lambda m: m.loss(), ("actor",), module)
        return nnx.state(module), values
    batched = jax.jit(jax.vmap(step_one, in_axes=(0, None)))
    # No optimizer cond is present here; this must be the loss gate itself.
    traced = jax.make_jaxpr(batched)(state, jnp.asarray(0))
    assert "cond" in _primitives(traced.jaxpr)
    state, values = batched(state, jnp.asarray(0))
    np.testing.assert_array_equal(values["actor"], [0.0, 0.0])
    np.testing.assert_array_equal(state["calls"][...], [0, 0])
    state, values = batched(state, jnp.asarray(1))
    np.testing.assert_array_equal(values["actor"], [4.0, 4.0])
    np.testing.assert_array_equal(state["calls"][...], [1, 1])
