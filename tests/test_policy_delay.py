"""`policy_delay` skips the actor's work, and skipping it changes nothing.

dime_am gates the actor and the entropy dual behind `policy_delay: 3` while the
critic steps every gradient step. This repo implemented that at the *optimizer*
(`Policy.build_optimizer` -> `optax.conditionally_mask`), so the loss ran on all
three steps and two thirds of the result was thrown away -- for DIME that is a
16-deep differentiated sampling chain, the dominant term in its step.

`gated_loss` skips the loss itself. These pin the two properties that makes
sound: the gate agrees with the optimizer's own predicate, and the branch is
only taken when nothing is being recorded (`lax.cond` traces both branches, so a
metric on one side and not the other is a pytree mismatch, not a silent zero).
"""

import gymnasium as gym
import jax
import jax.extend
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from barl_jax.metrics import DEBUG, PRIMARY, recording
from barl_jax.policy import TRAINABLE, Policy, gated_loss, update_due
from barl_jax.trainer import train_step

from conftest import ACT_DIM, BATCH, OBS_DIM, make_env_stats

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
            "rew": f(BATCH), "term": jnp.zeros((BATCH,), jnp.float32)}


def test_update_due_matches_the_optimizers_own_predicate():
    """`conditionally_mask(tx, lambda step: (step + 1) % k == 0)` counts the same
    gradient steps `train_steps` does, so the two must agree step for step."""
    policy = make("AMDP")
    for step in range(9):
        policy.train_steps[...] = jnp.asarray(step, jnp.int32)
        assert bool(update_due(policy, 3)) == ((step + 1) % 3 == 0)


def test_update_due_is_python_true_while_a_tier_is_armed():
    """A logged step must report the real loss, not the skip branch's zero."""
    policy = make("AMDP")
    policy.train_steps[...] = jnp.asarray(0, jnp.int32)   # not a multiple of 3
    assert update_due(policy, 3) is not True
    with recording(PRIMARY):
        assert update_due(policy, 3) is True


def test_gated_loss_returns_zero_off_cadence_and_the_value_on():
    policy = make("AMDP")
    called = []

    def fn(module):
        called.append(True)
        return jnp.asarray(2.5)

    for step, due in [(0, False), (1, False), (2, True)]:
        policy.train_steps[...] = jnp.asarray(step, jnp.int32)
        value = float(gated_loss(policy, 3, fn, policy.critic))
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
    the predicate unbatched (`policy.gating`, `in_axes=None`). If it ever goes
    back to being read off the per-seed `train_steps` buffer, the `cond`
    disappears from the jaxpr and the saving silently goes with it.
    """
    seeds = 2
    policies = []
    for i in range(seeds):
        p = make(name)
        policies.append((p, nnx.Optimizer(p, p.build_optimizer(lr_schedule=None), wrt=TRAINABLE)))

    from barl_jax.policy import stack
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
