"""The `Critic` contract, over every registered critic.

Everything here holds for all seven regardless of what the heads predict -- a
scalar, categorical atoms, or a smoothed histogram -- which is the property that
makes swapping a critic class a one-word edit in config/models.yaml.

Grouped a few assertions to a test on purpose: an eager `Ensemble` forward costs
real time, so each test re-uses one `raw()` rather than paying for its own.
"""

import jax
import jax.numpy as jnp
import pytest
from conftest import BATCH, CRITICS, build, built, critic_action, finite, target_expectation
from flax import nnx

from barl_jax.components import TRAINABLE, frozen
from barl_jax.critics.nclql import NoiseConditionedCritic
from barl_jax.networks.registry import get_component

# `NoiseConditionedCritic.noise_td_loss` takes a sigma ladder and a key
# in addition to the plain TD inputs, because TD at the clean level
# plus smoothing of the noisier ones is one loss over two forwards. So the shared
# contract below is the plain one, and the noise-conditioned critics get the same
# assertions through their own entry point.
PLAIN = [n for n in CRITICS if not issubclass(get_component(n), NoiseConditionedCritic)]
NOISE_CONDITIONED = [n for n in CRITICS if issubclass(get_component(n), NoiseConditionedCritic)]

REWARD = jnp.array([-2.0, -0.5, 0.5, 2.0])          # BATCH values, well inside every support

# C51 transports the whole support, so `r + gamma*v_max` has to stay inside it or
# the projection clips and equivariance genuinely stops holding. That caps a
# usable reward at v_max*(1 - gamma), which at the shipped gamma is 0.1.
SMALL_REWARD = jnp.array([-0.06, -0.02, 0.02, 0.06])


@pytest.mark.parametrize("name", CRITICS)
def test_pricing_contract(name, obs, act):
    """Shapes, finiteness, and `value()` being the pessimistic reading of the
    same forward pass `expectation()` averages."""
    c = built(name)
    a = critic_action(c, act)
    raw = c.raw(obs, a)

    assert raw.shape[:2] == (c.n_models, BATCH)
    assert c.value(obs, a).shape == (BATCH, 1)
    assert c.expectation(obs, a).shape == (BATCH, 1)
    assert finite(raw)
    # Reduced off one `raw` rather than two calls: a BatchRenorm critic in train
    # mode moves its own statistics per forward, so two calls are two functions.
    assert jnp.all(c.reduce_value(raw) <= c.expectation_from_raw(raw).mean(axis=0) + 1e-4)


@pytest.mark.parametrize("name", CRITICS)
def test_frozen_pricing_is_repeatable(name, obs, act):
    """`frozen()` has to give a critic that prices the same action the same way
    twice -- every actor loss here scores candidates it never trained on, and
    a moving normalization would shift the function under its own regression."""
    c = built(name)
    a = critic_action(c, act)
    assert jnp.array_equal(frozen(c).value(obs, a), frozen(c).value(obs, a))


ONE = jnp.ones((BATCH,))


@pytest.mark.parametrize("name", CRITICS)
def test_bellman_contract(name, obs, act):
    """The target's expectation, in whatever representation the class returns it."""
    c = build(name)
    a = critic_action(c, act)
    raw = c.raw(obs, a)
    boot = c.bootstrap_from_raw(raw)
    live, done = jnp.zeros((BATCH,)), jnp.ones((BATCH,))

    # terminated=1 cuts the bootstrap -- catches an inverted flag or a discount
    # applied on the wrong side of it.
    terminal = c.bellman_target(REWARD, done, ONE, boot)
    assert jnp.allclose(target_expectation(c, terminal), REWARD, atol=0.05)

    # Shifting every chunk_return by d shifts E[target] by d on the non-terminal branch.
    base = target_expectation(c, c.bellman_target(SMALL_REWARD, live, ONE, boot))
    moved = target_expectation(c, c.bellman_target(SMALL_REWARD + 0.02, live, ONE, boot))
    assert jnp.allclose(moved - base, 0.02, atol=1e-3)

    one_step = target_expectation(c, c.bellman_target(jnp.zeros((BATCH,)), live, ONE, boot))
    two_step = target_expectation(
        c, c.bellman_target(jnp.zeros((BATCH,)), live, jnp.full((BATCH,), 2), boot))
    assert jnp.allclose(two_step, c.gamma * one_step, atol=0.05)

    # `bellman_target` and `regression_loss` live in different classes; this is
    # the coupling between them. `target_from_raw` is the same lift applied to
    # one of the critic's own predictions (NC-LQL's smoothing loss).
    for target in (c.bellman_target(REWARD, live, ONE, boot), c.target_from_raw(raw)):
        loss = c.regression_loss(raw, target)
        assert loss.shape == () and finite(loss)


@pytest.mark.parametrize("name", [n for n in CRITICS if not issubclass(get_component(n), NoiseConditionedCritic)])
def test_bellman_target_matches_hand_computed_chunk_return(name):
    """The directive's own worked example: K=3, rewards=[1,2,3], gamma=0.5,
    nonterminal next value=8 -> chunk_return=2.75, td_target=3.75. chunk_return
    is an *input* here (the collector's own accumulation is checked separately,
    in tests/test_replay_buffer.py); this is the objective layer's half."""
    c = build(name)
    if hasattr(c, "support"):
        pytest.skip("categorical critics build the transformed support instead -- see "
                    "test_categorical_projection_matches_scalar_discount in test_critic_stats.py")
    c.gamma = 0.5
    target = c.bellman_target(jnp.array([2.75]), jnp.array([0.0]), jnp.array([3]),
                              jnp.array([[8.0]]))
    assert jnp.allclose(target, jnp.array([[3.75]]), atol=1e-5)


@pytest.mark.parametrize("name", [n for n in CRITICS if not issubclass(get_component(n), NoiseConditionedCritic)])
def test_bellman_target_k1_matches_one_step(name):
    """K=1 is an explicit value flowing through the ordinary path, not a
    fallback for an omitted horizon."""
    c = build(name)
    if hasattr(c, "support"):
        pytest.skip("categorical: see test_categorical_projection_matches_scalar_discount")
    c.gamma = 0.5
    target = c.bellman_target(jnp.array([1.0]), jnp.array([0.0]), jnp.array([1]), jnp.array([[8.0]]))
    assert jnp.allclose(target, jnp.array([[1.0 + 0.5 * 8.0]]), atol=1e-5)


@pytest.mark.parametrize("name", [n for n in CRITICS if not issubclass(get_component(n), NoiseConditionedCritic)])
def test_bellman_target_terminated_drops_bootstrap(name):
    """terminated=1 must zero the bootstrap contribution regardless of its
    value -- bootstrap is deliberately large here to make a sign/mask bug
    visible rather than accidentally passing at a small value."""
    c = build(name)
    if hasattr(c, "support"):
        pytest.skip("categorical: see test_categorical_projection_matches_scalar_discount")
    c.gamma = 0.5
    target = c.bellman_target(jnp.array([2.0]), jnp.array([1.0]), jnp.array([2]), jnp.array([[100.0]]))
    assert jnp.allclose(target, jnp.array([[2.0]]), atol=1e-5)


@pytest.mark.parametrize("name", [n for n in CRITICS if not issubclass(get_component(n), NoiseConditionedCritic)])
def test_entropy_bonus_sign_and_discount(name):
    """entropy_bonus enters exactly like the bootstrap: added before the
    n-step discount, so its contribution to the target is discount*entropy_bonus."""
    c = build(name)
    if hasattr(c, "support"):
        pytest.skip("categorical: see test_categorical_projection_matches_scalar_discount")
    c.gamma = 0.5
    with_bonus = c.bellman_target(jnp.array([0.0]), jnp.array([0.0]), jnp.array([2]),
                                  jnp.array([[0.0]]), entropy_bonus=jnp.array([[1.0]]))
    without = c.bellman_target(jnp.array([0.0]), jnp.array([0.0]), jnp.array([2]), jnp.array([[0.0]]))
    assert jnp.allclose(with_bonus - without, 0.25, atol=1e-5)  # gamma**2 == 0.25


@pytest.mark.parametrize("name", PLAIN)
def test_loss_is_differentiable(name, obs, act):
    """`loss` fills exactly `losses["critic"]`, and that scalar moves the heads."""
    c = build(name)                       # fresh: loss updates BatchRenorm state
    a = critic_action(c, act)

    def loss_fn(critic):
        losses = {}
        critic.td_loss(obs, a, obs[::-1], a, REWARD, jnp.zeros((BATCH,)), ONE, losses)
        assert set(losses) == {"critic"}
        return losses["critic"]

    grads = nnx.grad(loss_fn, argnums=nnx.DiffState(0, TRAINABLE))(c)
    leaves = jax.tree.leaves(nnx.state(grads))
    assert leaves and all(finite(g) for g in leaves)
    assert any(jnp.any(g != 0) for g in leaves), "no head received gradient"


@pytest.mark.parametrize("name", NOISE_CONDITIONED)
def test_noise_conditioned_loss_is_differentiable(name, obs, act):
    """The same contract as above through the noise-conditioned entry point: one
    `losses["critic"]` covering both the TD and the smoothing term, and it moves
    the heads."""
    # The raw action, not `critic_action`: `augment()` is what widens it to the
    # inner critic's input, and doing both would widen it twice.
    c = build(name)
    sigmas = jnp.array([0.5, 0.2, 0.05])

    def loss_fn(critic):
        losses = {}
        critic.noise_td_loss(obs, act, obs[::-1], act, REWARD, jnp.zeros((BATCH,)), ONE,
                    sigmas, jax.random.PRNGKey(0), losses)
        assert set(losses) == {"critic"}
        return losses["critic"]

    grads = nnx.grad(loss_fn, argnums=nnx.DiffState(0, TRAINABLE))(c)
    leaves = jax.tree.leaves(nnx.state(grads))
    assert leaves and all(finite(g) for g in leaves)
    assert any(jnp.any(g != 0) for g in leaves), "no head received gradient"


@pytest.mark.parametrize("name", CRITICS)
def test_polyak_target_tracks_the_live_net(name, obs, act):
    """`after_step()` blends the target tau of the way toward the live net.
    Checked against a *perturbed* live net -- the clone it starts as would make
    any blending law look correct."""
    c = build(name)
    if not c.has_target:
        pytest.skip("no target network at this critic's polyak_tau")

    nnx.update(c.q_function, jax.tree.map(lambda x: x + 1.0,
                                          nnx.state(c.q_function, nnx.Param)))
    live = nnx.state(c.q_function, nnx.Param)
    before = jax.tree.map(jnp.copy, nnx.state(c.critic_target, nnx.Param))

    c.after_step()

    expected = jax.tree.map(lambda b, l: b + c.polyak_tau * (l - b), before, live)
    for got, want in zip(jax.tree.leaves(nnx.state(c.critic_target, nnx.Param)),
                         jax.tree.leaves(expected)):
        assert jnp.allclose(got, want, atol=1e-5)


def test_bfloat16_heads_still_price(obs, act):
    """The shipped dtype for most of these, and the one SDAC runs. The contracts
    above are checked in float32; this only asserts low precision stays finite."""
    c = build("C51Critic", dtype="bfloat16")
    assert finite(c.value(obs, critic_action(c, act)))
