"""The `Critic` contract, over every registered critic and over the full
composition grid (`conftest.VARIANTS`): value head x bootstrap x normalization
x ensemble reduction x objective.

Grouped a few assertions to a test on purpose: an eager `Ensemble` forward costs
real time, so each test re-uses one `raw()` rather than paying for its own.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest
from conftest import (
    BATCH,
    CRITICS,
    HEADS,
    VARIANTS,
    build,
    build_variant,
    finite,
    make_env_stats,
    target_expectation,
)
from flax import nnx

from barl_jax.components import TRAINABLE, frozen, is_target
from barl_jax.config.serialization import merge
from barl_jax.critics import presets
from barl_jax.registration import get_component

REWARD = jnp.array([-2.0, -0.5, 0.5, 2.0])          # BATCH values, well inside every support

# C51 transports the whole support, so `r + gamma*v_max` has to stay inside it or
# the projection clips and equivariance genuinely stops holding. That caps a
# usable reward at v_max*(1 - gamma), which at the shipped gamma is 0.1.
SMALL_REWARD = jnp.array([-0.06, -0.02, 0.02, 0.06])
ONE = jnp.ones((BATCH,))
SCALAR = [n for n in HEADS if not hasattr(get_component(n), "probs")]
POLYAK = [v for v, (_, o) in VARIANTS.items() if o["target"]["kind"] == "polyak"]


def td_losses(c, obs, act):
    return c.td_loss(obs, act, obs[::-1], act, REWARD, jnp.zeros((BATCH,)), ONE,
                     key=jax.random.PRNGKey(0))


@pytest.mark.parametrize("variant", VARIANTS)
def test_pricing_contract(variant, obs, act):
    """Shapes, finiteness, and `value()` never above the expectation it reduces."""
    c = build_variant(variant)
    raw = c.raw(obs, act)

    assert raw.shape[:2] == (c.n_models, BATCH)
    assert finite(raw)
    # Reduced off one `raw`: a BatchRenorm critic in train mode moves its own
    # statistics per forward, so two calls are two functions.
    exp = c.expectation_members(raw).mean(axis=0)
    assert c.reduce_value(raw).shape == exp.shape == (BATCH, 1)
    assert jnp.all(c.reduce_value(raw) <= exp + 1e-4)
    assert jnp.array_equal(frozen(c).value(obs, act), frozen(c).value(obs, act))


@pytest.mark.parametrize("variant", VARIANTS)
def test_loss_trains_the_live_net_only(variant, obs, act):
    """One `losses["critic"]` that moves the live heads and never the target
    net, whichever bootstrap path produced it."""
    c = build_variant(variant)

    def loss_fn(critic):
        losses = td_losses(critic, obs, act)
        assert set(losses) == {"critic"}
        return losses["critic"]

    grads = nnx.grad(loss_fn, argnums=nnx.DiffState(0, nnx.Param))(build_variant(variant))
    flat = [(path, jnp.asarray(g[...])) for path, g in nnx.to_flat_state(grads)]
    live = [g for path, g in flat if not is_target(path, g)]
    target = [g for path, g in flat if is_target(path, g)]
    assert live and all(finite(g) for g in live)
    assert any(jnp.any(g != 0) for g in live), "no head received gradient"
    assert all(jnp.all(g == 0) for g in target), "the target net received gradient"
    assert bool(target) == c.has_target


@pytest.mark.parametrize("variant", POLYAK)
def test_polyak_target_tracks_the_live_net(variant):
    """`after_step()` blends the target tau of the way toward the live net, as
    it stands after any weight projection. Checked against a *perturbed* live
    net -- the clone it starts as would make any blending law look correct."""
    c = build_variant(variant)
    assert jax.tree.all(jax.tree.map(jnp.array_equal, nnx.state(c.net, nnx.Param),
                                     nnx.state(c.net_target, nnx.Param))), \
        "target does not start as the (projected) live net"

    nnx.update(c.net, jax.tree.map(lambda x: x + 1.0, nnx.state(c.net, nnx.Param)))
    before = jax.tree.map(jnp.copy, nnx.state(c.net_target, nnx.Param))
    c.after_step()
    live = nnx.state(c.net, nnx.Param)

    expected = jax.tree.map(lambda b, l: b + c.config.target.tau * (l - b), before, live)
    for got, want in zip(jax.tree.leaves(nnx.state(c.net_target, nnx.Param)),
                         jax.tree.leaves(expected)):
        assert jnp.allclose(got, want, atol=1e-5)


def test_weight_norm_projects_every_step():
    c = build("C51Critic", weight_norm=True)
    nnx.update(c.net, jax.tree.map(lambda x: 3.0 * x + 0.1, nnx.state(c.net, nnx.Param)))
    c.after_step()
    members = c.net.ensemble.members
    stem = jnp.concatenate([members.stem.kernel[...], members.stem.bias[...][:, None, :]], axis=-2)
    assert jnp.allclose(jnp.linalg.norm(stem, axis=-2), 1.0, atol=1e-5)
    assert jnp.allclose(jnp.linalg.norm(members.head.kernel[...], axis=-2), 1.0, atol=1e-5)


@pytest.mark.parametrize("overrides", [
    {"target": {"joint_forward": False}},                        # batchrenorm needs the joint batch
    {"ensemble": {"value_reduce": "median"}},
    {"target": {"kind": "ema"}},
])
def test_invalid_compositions_are_rejected(overrides):
    with pytest.raises(ValueError):
        build("C51Critic", **overrides)


@pytest.mark.parametrize("name", HEADS)
def test_learning_objective_follows_the_noise_axis(name):
    cls = get_component(name)
    assert cls.learning_objective_for(cls.config_type()) == "td"
    noisy = replace(cls.config_type(), noise=replace(cls.config_type().noise, enabled=True))
    assert cls.learning_objective_for(noisy) == "noise_td"
    assert not build(name, noise={"enabled": True}).supports_bon


@pytest.mark.parametrize("name", HEADS)
def test_bellman_contract(name, obs, act):
    """The target's expectation, in whatever representation the head returns it."""
    c = build(name)
    raw = c.raw(obs, act)
    boot = c.bootstrap(raw)
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
    assert jnp.allclose(two_step, c.config.gamma * one_step, atol=0.05)

    # The Bellman target and the smoothing target are both what `regression_loss` fits.
    for target in (c.bellman_target(REWARD, live, ONE, boot), c.soft_target(raw)):
        loss = c.regression_loss(raw, target)
        assert loss.shape == () and finite(loss)


def test_reward_scale_scales_the_return():
    c = build("DirectCritic", reward_scale=2.0)
    target = c.bellman_target(jnp.array([1.5]), jnp.array([1.0]), jnp.array([1]), jnp.array([[0.0]]))
    assert jnp.allclose(target, 3.0)


@pytest.mark.parametrize("name", SCALAR)
def test_bellman_target_matches_hand_computed_chunk_return(name):
    """K=3, rewards=[1,2,3], gamma=0.5, nonterminal next value=8 ->
    chunk_return=2.75, td_target=3.75."""
    c = build(name)
    c.config = replace(c.config, gamma=0.5)
    target = c.bellman_target(jnp.array([2.75]), jnp.array([0.0]), jnp.array([3]),
                              jnp.array([[8.0]]))
    assert jnp.allclose(target, jnp.array([[3.75]]), atol=1e-5)


@pytest.mark.parametrize("name", SCALAR)
def test_bellman_target_k1_matches_one_step(name):
    c = build(name)
    c.config = replace(c.config, gamma=0.5)
    target = c.bellman_target(jnp.array([1.0]), jnp.array([0.0]), jnp.array([1]), jnp.array([[8.0]]))
    assert jnp.allclose(target, jnp.array([[1.0 + 0.5 * 8.0]]), atol=1e-5)


@pytest.mark.parametrize("name", SCALAR)
def test_bellman_target_terminated_drops_bootstrap(name):
    """Bootstrap deliberately large so a sign/mask bug is visible."""
    c = build(name)
    c.config = replace(c.config, gamma=0.5)
    target = c.bellman_target(jnp.array([2.0]), jnp.array([1.0]), jnp.array([2]), jnp.array([[100.0]]))
    assert jnp.allclose(target, jnp.array([[2.0]]), atol=1e-5)


@pytest.mark.parametrize("name", SCALAR)
def test_entropy_bonus_sign_and_discount(name):
    """entropy_bonus enters exactly like the bootstrap: added before the
    n-step discount."""
    c = build(name)
    c.config = replace(c.config, gamma=0.5)
    with_bonus = c.bellman_target(jnp.array([0.0]), jnp.array([0.0]), jnp.array([2]),
                                  jnp.array([[0.0]]), entropy_bonus=jnp.array([[1.0]]))
    without = c.bellman_target(jnp.array([0.0]), jnp.array([0.0]), jnp.array([2]), jnp.array([[0.0]]))
    assert jnp.allclose(with_bonus - without, 0.25, atol=1e-5)  # gamma**2 == 0.25


@pytest.mark.parametrize("name", CRITICS)
def test_registered_critics_train(name, obs, act):
    c = build(name)
    assert c.value(obs, act).shape == (BATCH, 1)
    grads = nnx.grad(lambda m: td_losses(m, obs, act)["critic"],
                     argnums=nnx.DiffState(0, TRAINABLE))(c)
    assert all(finite(g) for g in jax.tree.leaves(nnx.state(grads)))


@pytest.mark.parametrize("preset", ["target_ensemble", "xqc", "nclql"])
@pytest.mark.parametrize("name", HEADS)
def test_presets_compose_with_every_head(name, preset, obs, act):
    cls = get_component(name)
    config = merge(getattr(presets, preset)(cls.config_type()), {"latent_dim": 8})
    c = cls(make_env_stats(), config, rngs=nnx.Rngs(0))
    grads = nnx.grad(lambda m: td_losses(m, obs, act)["critic"],
                     argnums=nnx.DiffState(0, TRAINABLE))(c)
    assert all(finite(g) for g in jax.tree.leaves(nnx.state(grads)))


def test_bfloat16_heads_still_price(obs, act):
    """Low precision stays finite; the laws above are dtype-agnostic."""
    c = build("C51Critic", mlp={"dtype": "bfloat16"})
    assert finite(c.value(obs, act))
