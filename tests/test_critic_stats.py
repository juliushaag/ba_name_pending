"""What the critics' numbers actually *mean*, as closed-form laws.

The contracts in test_critics.py hold for every critic by construction. These
are the ones that distinguish the representations from each other -- whether a
spread is the return's or the heads', whether a Bellman target transports a
distribution or collapses it -- and each is checked against a formula rather
than a recorded value, so the tolerances are derived, not tuned.

Nothing here trains anything: every law below is exact at initialization.
"""

from dataclasses import replace

import jax.numpy as jnp
import pytest
from conftest import BATCH, CRITICS, HEADS, build, built, finite

from barl_jax.components import frozen
from barl_jax.critics import categorical_projection, reduce_members
from barl_jax.metrics import DEBUG, PRIMARY, DebugMetric, Metric, collect, recording

ONE = jnp.ones((1,))
MULTI_MEMBER = [n for n in CRITICS if built(n).n_models > 1]


def moments(support, d):
    """(mean, std) of a distribution over `support`, trailing axis last."""
    mean = (d * support).sum(axis=-1)
    var = (d * (support - mean[..., None]) ** 2).sum(axis=-1)
    return mean, jnp.sqrt(var)


def two_point(c, half_width: int):
    """A bootstrap distribution with an exactly known mean and std: half the mass
    on each of two atoms symmetric about the centre of the support."""
    support = c.support[...]
    n = support.shape[0]
    lo, hi = n // 2 - half_width, n // 2 + half_width
    probs = jnp.zeros((1, n)).at[0, lo].set(0.5).at[0, hi].set(0.5)
    return probs, float(0.5 * (support[lo] + support[hi])), float(0.5 * (support[hi] - support[lo]))


def bin_width(c):
    return (c.v_max - c.v_min) / c.config.num_bins


# ----------------------------------------------------------------------
# The distributional / not-distributional split
# ----------------------------------------------------------------------

@pytest.mark.parametrize("half_width", [8, 16, 32])
@pytest.mark.parametrize("name", ["C51Critic"])
def test_categorical_target_transports_the_spread(name, half_width):
    """C51 is genuinely distributional: the Bellman operator moves the whole
    next-state distribution, so the target's spread is the bootstrap's scaled by
    gamma. The residual above that is the projection's own inflation, which is
    why this is a relative tolerance and not an equality."""
    c = built(name)
    boot, boot_mean, boot_std = two_point(c, half_width)

    target = c.bellman_target(jnp.zeros((1,)), jnp.zeros((1,)), ONE, boot)
    mean, std = moments(c.support[...], target)

    assert float(mean[0]) == pytest.approx(c.config.gamma * boot_mean, abs=1e-3)
    assert float(std[0]) == pytest.approx(c.config.gamma * boot_std, rel=5e-3)


@pytest.mark.parametrize("boot_value", [0.0, 2.0, -4.0])
@pytest.mark.parametrize("name", ["HLGaussCritic"])
def test_histogram_target_discards_the_spread(name, boot_value):
    """HL-Gauss is *not* distributional -- it bootstraps a scalar and only
    discretizes at the end, so the width of its target is label smoothing and
    nothing else. Pinned to the same constant whatever the bootstrap was, which
    is the whole difference from the test above."""
    c = built(name)
    target = c.bellman_target(jnp.zeros((1,)), jnp.zeros((1,)), ONE,
                              jnp.full((1, 1), boot_value))
    mean, std = moments(c.support[...], target)

    smoothing = jnp.sqrt(c.sigma ** 2 + bin_width(c) ** 2 / 12)
    assert float(mean[0]) == pytest.approx(c.config.gamma * boot_value, abs=1e-3)
    assert float(std[0]) == pytest.approx(float(smoothing), rel=1e-3)


@pytest.mark.parametrize("value", [0.0, 3.0, -5.0])
@pytest.mark.parametrize("name", ["HLGaussCritic"])
def test_histogram_of_a_scalar_is_a_discretized_gaussian(name, value):
    """`sigma` is a label-smoothing width, so the histogram a scalar lands in
    has that scalar as its mean and a variance of exactly sigma^2 plus the
    discretization's own bin^2/12 -- independent of where on the support it is."""
    c = built(name)
    hist = c.bellman_target(jnp.array([value]), jnp.ones((1,)), ONE, jnp.zeros((1, 1)))
    mean, std = moments(c.support[...], hist)

    assert float(hist.sum()) == pytest.approx(1.0, abs=1e-5)
    assert float(mean[0]) == pytest.approx(value, abs=1e-3)
    assert float(std[0]) == pytest.approx(
        float(jnp.sqrt(c.sigma ** 2 + bin_width(c) ** 2 / 12)), rel=1e-3)


# ----------------------------------------------------------------------
# Ensemble reduction
# ----------------------------------------------------------------------

RAW = jnp.array([[[1.0], [3.0], [-2.0]], [[2.0], [-1.0], [-2.0]]])      # (2 heads, 3, 1)


def test_min_and_mean_reductions():
    assert jnp.array_equal(reduce_members(RAW, RAW, "min"), RAW.min(axis=0))
    assert jnp.array_equal(reduce_members(RAW, RAW, "mean"), RAW.mean(axis=0))


def test_min_selects_whole_members_not_elementwise_minima():
    """A per-member payload follows the member with the lower expectation."""
    payload = jnp.stack([jnp.zeros((3, 2)), jnp.ones((3, 2))])       # (2 members, 3, 2)
    picked = reduce_members(payload, RAW, "min")
    lower = (RAW[1] < RAW[0]).astype(jnp.float32)                      # (3, 1): member 1 lower
    assert jnp.array_equal(picked, jnp.broadcast_to(lower, (3, 2)))


@pytest.mark.parametrize("name", MULTI_MEMBER)
def test_ensemble_heads_are_independently_initialized(name, obs, act):
    """Ensemble's own docstring names this failure: drop `nnx.split_rngs` and
    every member draws the same init key, the heads collapse to identical, the
    spread goes to 0 and pessimism silently switches off for the whole run --
    nothing errors, the curve just looks plausible."""
    c = built(name)
    spread = c.expectation_members(c.raw(obs, act))
    assert float(spread.std(axis=0).mean()) > 1e-3


# ----------------------------------------------------------------------
# The categorical projection
# ----------------------------------------------------------------------

ATOMS = jnp.linspace(-1.0, 1.0, 11)     # spacing 0.2, so 0.4 is exactly atom 7
NO_TERM = jnp.zeros((1,))


def _project(reward, probs=None, gamma=1.0):
    probs = jnp.zeros((1, 1, 11)).at[0, 0, 5].set(1.0) if probs is None else probs
    return categorical_projection(probs, ATOMS, jnp.array([reward]), jnp.full((1, 1), gamma),
                                  -1.0, 1.0)


def test_categorical_projection_matches_scalar_discount():
    """The directive's worked example (K=3, rewards=[1,2,3], gamma=0.5, next
    value=8 -> chunk_return=2.75, td_target=3.75) at the categorical layer: a
    degenerate one-atom point mass at 8.0 (nearest representable atom) projects
    to a distribution whose mean matches the scalar target exactly, proving the
    categorical and scalar paths agree on the discount without the categorical
    path ever collapsing its distribution before projecting."""
    atoms = jnp.linspace(-10.0, 10.0, 21)     # spacing 1.0, so 8.0 is exactly an atom
    next_probs = jnp.zeros((1, 1, 21)).at[0, 0, 18].set(1.0)   # point mass at atom value 8.0
    proj = categorical_projection(next_probs, atoms, jnp.array([2.75]), jnp.full((1, 1), 0.5 ** 3),
                                  -10.0, 10.0)
    mean, _ = moments(atoms, proj)
    assert float(mean[0, 0]) == pytest.approx(3.75, abs=1e-5)


def test_projection_leaves_an_on_atom_point_mass_alone():
    """The `on_atom` branch. Without it l == u there and the mass is counted
    into both, so a point mass landing exactly on an atom doubles."""
    proj = _project(0.4)
    assert jnp.allclose(proj[0, 0], jnp.zeros(11).at[7].set(1.0), atol=1e-6)


def test_projection_splits_off_atom_mass_with_the_exact_two_point_variance():
    """A point mass landing between atoms becomes the two-point distribution
    straddling it -- mean preserved, and a variance of exactly w(1-w)*spacing^2,
    which is the inflation the transport test above tolerates."""
    proj = _project(0.05)                       # 1/4 of the way from atom 5 to atom 6
    mean, std = moments(ATOMS, proj)

    assert float(mean[0, 0]) == pytest.approx(0.05, abs=1e-6)
    assert float(std[0, 0] ** 2) == pytest.approx(0.25 * 0.75 * 0.2 ** 2, rel=1e-4)


# ----------------------------------------------------------------------
# Pessimism and saturation
# ----------------------------------------------------------------------

def _logits(n, *indices):
    """(members, 1, n) logits whose softmax is a point mass per member."""
    onehot = jnp.stack([jnp.zeros((1, n)).at[0, i].set(1.0) for i in indices])
    return jnp.where(onehot > 0, 0.0, -1e9)


@pytest.mark.parametrize("name", [n for n in HEADS if n != "DirectCritic"])
def test_the_two_reduction_axes_are_independent(name, obs, act):
    """`value_reduce` prices the members' disagreement into `value()`;
    `target_reduce` puts it in the Bellman target. Neither reaches the other's
    quantity, and each does reach its own."""
    def priced(value_reduce, target_reduce):
        c = build(name, ensemble={"value_reduce": value_reduce, "target_reduce": target_reduce})
        raw = c.raw(obs, act)
        exp = c.expectation_members(raw).mean(axis=0)
        target = c.bellman_target(jnp.zeros((BATCH,)), jnp.zeros((BATCH,)), jnp.ones((BATCH,)),
                                  c.bootstrap(raw))
        return c, c.reduce_value(raw), exp, c.target_expectation(target)

    c, price, exp, mean_target = priced("min", "mean")
    _, price_min_target, _, min_target = priced("min", "min")
    _, mean_price, _, mean_target_mean_price = priced("mean", "mean")

    assert jnp.array_equal(price, price_min_target), "target_reduce moved the price"
    assert jnp.array_equal(mean_target, mean_target_mean_price), "value_reduce moved the target"
    assert float(price.mean()) < float(mean_price.mean()) - 1e-4
    assert float(min_target.mean()) < float(mean_target.mean()) - 1e-4
    assert jnp.allclose(mean_target, c.config.gamma * exp, atol=1e-3)


def test_min_target_transports_the_lower_heads_distribution():
    """Not a pointwise min over probabilities -- that is not a distribution.
    It picks the member with the lower E[Z] and moves that member's whole
    categorical."""
    c = build("C51Critic", ensemble={"target_reduce": "min"}, num_bins=11, v_min=-1.0, v_max=1.0)
    target = c.bellman_target(jnp.zeros((1,)), NO_TERM, ONE, c.bootstrap(_logits(11, 8, 2)))
    assert float(moments(c.support[...], target)[0][0]) == pytest.approx(c.config.gamma * -0.6, abs=1e-4)


def test_mean_target_takes_both_heads():
    c = build("C51Critic", ensemble={"target_reduce": "mean"}, num_bins=11, v_min=-1.0, v_max=1.0)
    target = c.bellman_target(jnp.zeros((1,)), NO_TERM, ONE, c.bootstrap(_logits(11, 8, 2)))
    assert float(moments(c.support[...], target)[0][0]) == pytest.approx(0.0, abs=1e-4)


def test_projection_saturates_once_the_reward_leaves_the_contraction_interior():
    """Documented footgun, not a bug: C51 transports the whole support, so the
    operator only preserves expectation while r + gamma*v_max stays inside it.
    Past v_max*(1 - gamma) the top atoms clip and the target reads low -- which
    is what a loose `return_max` quietly costs."""
    top = jnp.zeros((1, 1, 11)).at[0, 0, 10].set(1.0)      # every atom that carries mass is v_max
    transported = lambda r: r + 0.99 * 1.0

    inside = _project(0.005, probs=top, gamma=0.99)        # 0.005 < v_max*(1 - gamma) = 0.01
    assert float(moments(ATOMS, inside)[0][0, 0]) == pytest.approx(transported(0.005), abs=1e-4)

    outside = _project(0.5, probs=top, gamma=0.99)
    got = float(moments(ATOMS, outside)[0][0, 0])
    assert got == pytest.approx(1.0, abs=1e-4), "mass should pile on the top atom"
    assert got < transported(0.5) - 1e-3, "and read low by exactly what was clipped"


# ----------------------------------------------------------------------
# Noise conditioning
# ----------------------------------------------------------------------

@pytest.mark.parametrize("name", HEADS)
def test_noise_level_conditions_the_prediction(name, obs, act):
    """Through `frozen`, as the Langevin sampler prices: a train-mode batch
    normalization would zero a level that is constant across the batch."""
    c = frozen(build(name, noise={"enabled": True}))
    assert not jnp.allclose(c.expectation(obs, act, level=0), c.expectation(obs, act, level=3)), \
        "levels embed identically"
    assert jnp.array_equal(c.expectation(obs, act), c.expectation(obs, act, level=c.noise.clean_level))


@pytest.mark.parametrize("name", HEADS)
def test_smoothing_coefficient_is_actually_wired_in(name, obs, act, key):
    """TD at the clean level and smoothing of the noisier ones fold into one
    `losses["critic"]`, since both fit the same heads."""
    c = build(name, noise={"enabled": True})

    def loss_at(coef):
        c.config = replace(c.config, noise=replace(c.config.noise, smoothing_coef=coef))
        losses = c.td_loss(obs, act, obs[::-1], act, jnp.zeros((BATCH,)),
                           jnp.zeros((BATCH,)), jnp.ones((BATCH,)), key=key)
        assert set(losses) == {"critic"}
        return losses["critic"]

    off, on = loss_at(0.0), loss_at(1.0)
    assert finite(off) and finite(on)
    assert not jnp.allclose(off, on), "smoothing_coef changed nothing"


# ----------------------------------------------------------------------
# What the metrics claim to measure
# ----------------------------------------------------------------------

def drain(name, obs, act, **overrides) -> dict:
    """One critic loss under `recording`, and the diagnostics it sowed."""
    from flax import nnx

    c = build(name, **overrides)
    with recording(PRIMARY, DEBUG):
        c.td_loss(obs, act, obs[::-1], act, jnp.zeros((BATCH,)), jnp.zeros((BATCH,)), jnp.ones((BATCH,)))
        sown, sown_debug = nnx.pop(c, Metric, DebugMetric)
    return {**collect(sown), **collect(sown_debug)}


def test_pessimism_gap_tracks_the_reduction_that_produced_it():
    """`E[Z] - value()` itself: with two members, `min` sits exactly one
    population std below the mean, and `mean` has no gap at all."""
    reduced = lambda r: build("EnsembleCritic", ensemble={"value_reduce": r}).pessimism_gap(RAW)
    assert jnp.allclose(reduced("min"), RAW.std(axis=0), atol=1e-6)
    assert jnp.allclose(reduced("mean"), 0.0, atol=1e-6)


@pytest.mark.parametrize("name", CRITICS)
def test_every_critic_prices_no_higher_than_its_own_expectation(name, obs, act):
    """A gap is a configured choice: one-sided always, strict only for `min`
    over more than one member."""
    c = built(name)
    gap = float(c.pessimism_gap(c.raw(obs, act)).mean())
    assert gap >= -1e-4
    assert (gap > 1e-3) == (c.config.ensemble.value_reduce == "min" and c.n_models > 1)


def test_metric_families_separate_the_two_spreads(obs, act):
    """`ensemble/disagreement` is spread across heads (epistemic) and
    `return_std` is the predicted return distribution's own (aleatoric). Only a
    critic that predicts a distribution has the second -- before this, nothing
    logged it and a C51 run was indistinguishable from a scalar-ensemble run."""
    scalar = drain("EnsembleCritic", obs, act, ensemble={"value_reduce": "min"})
    categorical = drain("C51Critic", obs, act, ensemble={"value_reduce": "mean"})

    for m in (scalar, categorical):
        assert m["ensemble/disagreement"] > 1e-3

    # Disagreement is the heads' spread whatever the reduction does with it;
    # the gap is what `value_reduce` chooses to price in.
    assert scalar["ensemble/pessimism_gap"] > 1e-3
    assert categorical["ensemble/pessimism_gap"] == pytest.approx(0.0, abs=1e-4)

    assert categorical["return_std/mean"] > 0.0
    assert not any(k.startswith("return_std") for k in scalar), "a scalar critic has no return spread"
