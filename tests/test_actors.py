"""The `Actor` contract, plus the algebraic identities the diffusion actors'
callers depend on.

Actors emit action plans `(B, chunk, dim)`, scored flat; the owning Policy
squashes nothing further. `sample` returns the plan, not a primitive.
"""

import jax
import jax.numpy as jnp
import pytest
from conftest import ACT_DIM, ACTORS, BATCH, build, built, finite
from flax import nnx

from barl_jax.actors.sdac import DDPMActor
from barl_jax.components import TRAINABLE
from barl_jax.registration import get_component

DIFFUSION = [n for n in ACTORS if issubclass(get_component(n), DDPMActor)]

# Every registered `Actor` gets `sample` from the ABC's default now (it calls
# `generate_action(...)[0]`), so this is every actor rather than a subset --
# kept as its own name since a few (StochasticActor) still override it with a
# cheaper path that skips the log-prob `generate_action` also computes.
SAMPLERS = [n for n in ACTORS if hasattr(get_component(n), "sample")]


@pytest.mark.parametrize("name", SAMPLERS)
def test_sample_contract(name, obs, key):
    """One plan per row, finite, and a pure function of the key it was given --
    every draw takes an explicit key because there is no global RNG here."""
    a = built(name)
    action = a.sample(obs, key=key)

    assert action.shape == (BATCH, a.chunk_size, ACT_DIM)
    assert finite(action)
    assert jnp.array_equal(action, a.sample(obs, key=key))
    assert not jnp.array_equal(action, a.sample(obs, key=jax.random.PRNGKey(7)))


@pytest.mark.parametrize("name", SAMPLERS)
def test_sample_is_differentiable_wrt_params(name, obs, key):
    a = build(name)
    grads = nnx.grad(lambda actor: actor.sample(obs, key=key).sum(),
                     argnums=nnx.DiffState(0, TRAINABLE))(a)
    leaves = jax.tree.leaves(nnx.state(grads))
    assert leaves and all(finite(g) for g in leaves)
    assert any(jnp.any(g != 0) for g in leaves), "sample() reached no parameter"


@pytest.mark.parametrize("name", DIFFUSION)
def test_x0_from_eps_inverts_the_marginal(name):
    """SDAC's candidates are `x0_from_eps` of the forward marginal at a known eps."""
    a = built(name)
    x0 = jax.random.normal(jax.random.PRNGKey(3), (BATCH, a.plan_dim))
    eps = jax.random.normal(jax.random.PRNGKey(4), (BATCH, a.plan_dim))
    t = jnp.full((BATCH, 1), 0.3)

    alpha, sigma = a.alpha_sigma(t)
    assert jnp.allclose(alpha ** 2 + sigma ** 2, 1.0, atol=1e-5)
    assert jnp.allclose(a.x0_from_eps(alpha * x0 + sigma * eps, eps, t), x0, atol=1e-3)


@pytest.mark.parametrize("name", DIFFUSION)
def test_sampled_actions_can_reach_the_box_edges(name, obs, key):
    """The clipped x0 is the played action, so the sampler's range is [-1, 1]."""
    a = built(name)
    a.field.zero_init_head()
    a.field.net.head.bias[...] = jnp.full_like(a.field.net.head.bias[...], -100.0)
    action = a.sample(obs, key=key)
    assert jnp.all(jnp.abs(action) > 0.99)


def test_stochastic_actor_log_prob_and_mode(obs, key):
    """The squashed-Gaussian pieces SAC/IQL need on top of `sample`."""
    a = built("StochasticActor")
    action, x, dist = a.rsample(obs, key)

    assert x.shape == (BATCH, a.plan_dim)
    assert jnp.all(jnp.abs(action) <= 1.0)
    assert jnp.all(jnp.abs(a.mode(obs)) <= 1.0)

    lp = a.log_prob(dist, a.to_plan(action))
    assert lp.shape == (BATCH,) and finite(lp)
