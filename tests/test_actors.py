"""The `Actor` contract, plus the algebraic identities the diffusion actors'
callers depend on.

Score-based actors deliberately return a raw, unsquashed terminal state -- the
owning Policy squashes it -- so nothing here asserts a bounded action.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from barl_jax.networks.actor import DDPMActor, DiscreteDiffusionActorBase
from barl_jax.networks.registry import get_component
from barl_jax.policy import TRAINABLE

from conftest import ACT_DIM, ACTORS, BATCH, build, built, finite

DIFFUSION = [n for n in ACTORS if issubclass(get_component(n), DiscreteDiffusionActorBase)]

# The eps/score duality is a property of the *ladder*, not of the base class:
# it holds wherever `to_score` is -raw/sqrt(nu), which is every DDPM schedule.
# OUDiffusionActor parameterizes an anchored score directly (`raw - x*inv_eta2`),
# so `predict_raw` is not eps_theta there and the base's eps helpers do not apply.
DDPM_FAMILY = [n for n in ACTORS if issubclass(get_component(n), DDPMActor)]


@pytest.mark.parametrize("name", ACTORS)
def test_sample_contract(name, obs, key):
    """One action per row, finite, and a pure function of the key it was given --
    every draw takes an explicit key because there is no global RNG here."""
    a = built(name)
    action = a.sample(obs, key)

    assert action.shape == (BATCH, ACT_DIM)
    assert finite(action)
    assert jnp.array_equal(action, a.sample(obs, key))
    assert not jnp.array_equal(action, a.sample(obs, jax.random.PRNGKey(7)))


@pytest.mark.parametrize("name", ACTORS)
def test_sample_is_differentiable_wrt_params(name, obs, key):
    a = build(name)
    grads = nnx.grad(lambda actor: actor.sample(obs, key).sum(),
                     argnums=nnx.DiffState(0, TRAINABLE))(a)
    leaves = jax.tree.leaves(nnx.state(grads))
    assert leaves and all(finite(g) for g in leaves)
    assert any(jnp.any(g != 0) for g in leaves), "sample() reached no parameter"


@pytest.mark.parametrize("name", DIFFUSION)
def test_x0_from_eps_inverts_the_marginal(name):
    """`x0_from_eps` is the algebraic inverse of `q(x_n | x_0)` at a known noise.
    SDAC's candidate mechanism is exactly this identity, so it is worth pinning:
    reconstruct the x0 that each drawn eps implies from the same x_n."""
    a = built(name)
    x0 = jax.random.normal(jax.random.PRNGKey(3), (BATCH, ACT_DIM))
    eps = jax.random.normal(jax.random.PRNGKey(4), (BATCH, ACT_DIM))
    n = jnp.ones((BATCH,), jnp.int32)

    g, nu = a.marginal_coef(n)
    x_n = g * x0 + jnp.sqrt(nu) * eps
    assert jnp.allclose(a.x0_from_eps(x_n, eps, n), x0, atol=1e-3)


@pytest.mark.parametrize("name", DDPM_FAMILY)
def test_eps_and_score_parameterizations_agree(name, obs):
    """The codebase states its losses in score space and the DDPM literature in
    eps space; the base class calls them renamings and one division over any
    ladder whose `to_score` is -raw/sqrt(nu). This is that claim."""
    a = built(name)
    x = jax.random.normal(jax.random.PRNGKey(5), (BATCH, ACT_DIM))
    n = jnp.ones((BATCH,), jnp.int32)

    from_score = a.eps_from_score(a.predict_score(obs, x, n), n)
    assert jnp.allclose(from_score, a.predict_eps(obs, x, n), atol=1e-3)


@pytest.mark.parametrize("name", DIFFUSION)
def test_diffuse_and_eps_loss_shapes(name, obs, key):
    a = built(name)
    x0 = jnp.tanh(jax.random.normal(jax.random.PRNGKey(6), (BATCH, ACT_DIM)))
    n = jnp.ones((BATCH,), jnp.int32)

    assert a.diffuse(x0, n, key).shape == (BATCH, ACT_DIM)
    loss = a.eps_loss(obs, x0, n, key)
    assert loss.shape == () and finite(loss)


def test_stochastic_actor_log_prob_and_mode(obs, key):
    """The squashed-Gaussian pieces SAC/IQL need on top of `sample`."""
    a = built("StochasticActor")
    action, x, dist = a.rsample(obs, key)

    assert x.shape == (BATCH, ACT_DIM)
    assert jnp.all(jnp.abs(action) <= 1.0)
    assert jnp.all(jnp.abs(a.mode(obs)) <= 1.0)

    lp = a.log_prob(dist, action)
    assert lp.shape == (BATCH,) and finite(lp)
