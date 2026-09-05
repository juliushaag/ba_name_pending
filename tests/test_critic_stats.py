"""What the critics' numbers actually *mean*, as closed-form laws.

The contracts in test_critics.py hold for every critic by construction. These
are the ones that distinguish the representations from each other -- whether a
spread is the return's or the heads', whether a Bellman target transports a
distribution or collapses it -- and each is checked against a formula rather
than a recorded value, so the tolerances are derived, not tuned.

Nothing here trains anything: every law below is exact at initialization.
"""

from barl_jax.objectives import td_loss, noise_td_loss

import jax.numpy as jnp
import pytest

from barl_jax.critics import (C51Critic, EnsembleCritic, HLGaussCritic,
                                      categorical_projection)
from barl_jax.metrics import DEBUG, PRIMARY, DebugMetric, Metric, collect, recording
from barl_jax.networks.registry import get_component
from barl_jax.policies.lql import NoiseConditionedCritic

from conftest import (BATCH, CRITICS, build, built, critic_action, finite,
                      target_expectation)

_sub = lambda base: [n for n in CRITICS if issubclass(get_component(n), base)]

CATEGORICAL = _sub(C51Critic)          # bootstraps and transports a distribution
HISTOGRAM = _sub(HLGaussCritic)        # bootstraps a scalar, discretizes at the end
SCALAR = _sub(EnsembleCritic)
NOISE_CONDITIONED = _sub(NoiseConditionedCritic)


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
    probs = jnp.zeros((1, 1, n)).at[0, 0, lo].set(0.5).at[0, 0, hi].set(0.5)
    return probs, float(0.5 * (support[lo] + support[hi])), float(0.5 * (support[hi] - support[lo]))


def bin_width(c):
    return (c.v_max - c.v_min) / c.num_bins


# ----------------------------------------------------------------------
# The distributional / not-distributional split
# ----------------------------------------------------------------------

@pytest.mark.parametrize("half_width", [8, 16, 32])
@pytest.mark.parametrize("name", CATEGORICAL)
def test_categorical_target_transports_the_spread(name, half_width):
    """C51 is genuinely distributional: the Bellman operator moves the whole
    next-state distribution, so the target's spread is the bootstrap's scaled by
    gamma. The residual above that is the projection's own inflation, which is
    why this is a relative tolerance and not an equality."""
    c = built(name)
    boot, boot_mean, boot_std = two_point(c, half_width)

    target = c.bellman_target(jnp.zeros((1,)), jnp.zeros((1,)), boot)
    mean, std = moments(c.support[...], target)

    assert float(mean[0]) == pytest.approx(c.gamma * boot_mean, abs=1e-3)
    assert float(std[0]) == pytest.approx(c.gamma * boot_std, rel=5e-3)


@pytest.mark.parametrize("boot_value", [0.0, 2.0, -4.0])
@pytest.mark.parametrize("name", HISTOGRAM)
def test_histogram_target_discards_the_spread(name, boot_value):
    """HL-Gauss is *not* distributional -- it bootstraps a scalar and only
    discretizes at the end, so the width of its target is label smoothing and
    nothing else. Pinned to the same constant whatever the bootstrap was, which
    is the whole difference from the test above."""
    c = built(name)
    target = c.bellman_target(jnp.zeros((1,)), jnp.zeros((1,)),
                              jnp.full((1, 1), boot_value))
    mean, std = moments(c.support[...], target)

    smoothing = jnp.sqrt(c.sigma ** 2 + bin_width(c) ** 2 / 12)
    assert float(mean[0]) == pytest.approx(c.gamma * boot_value, abs=1e-3)
    assert float(std[0]) == pytest.approx(float(smoothing), rel=1e-3)


@pytest.mark.parametrize("value", [0.0, 3.0, -5.0])
@pytest.mark.parametrize("name", HISTOGRAM)
def test_histogram_of_a_scalar_is_a_discretized_gaussian(name, value):
    """`sigma` is a label-smoothing width, so the histogram a scalar lands in
    has that scalar as its mean and a variance of exactly sigma^2 plus the
    discretization's own bin^2/12 -- independent of where on the support it is."""
    c = built(name)
    hist = c.bellman_target(jnp.array([value]), jnp.ones((1,)), jnp.zeros((1, 1)))
    mean, std = moments(c.support[...], hist)

    assert float(hist.sum()) == pytest.approx(1.0, abs=1e-5)
    assert float(mean[0]) == pytest.approx(value, abs=1e-3)
    assert float(std[0]) == pytest.approx(
        float(jnp.sqrt(c.sigma ** 2 + bin_width(c) ** 2 / 12)), rel=1e-3)


# ----------------------------------------------------------------------
# Ensemble reduction
# ----------------------------------------------------------------------

RAW = jnp.array([[[1.0], [3.0], [-2.0]], [[2.0], [-1.0], [-2.0]]])      # (2 heads, 3, 1)


def test_rho_one_at_two_heads_is_exactly_the_min():
    """The claim in EnsembleCritic's docstring: at n_models=2 the population std
    is |q1 - q2|/2, so mean - 1.0*std is TD3's min-over-heads rather than an
    approximation of it. The 1e-3 is the sqrt(maximum(var, 1e-8)) floor, which
    is why two heads that agree exactly come back 1e-4 low."""
    c = build("EnsembleCritic", rho=1.0, n_models=2)
    assert jnp.allclose(c.reduce_value(RAW), RAW.min(axis=0), atol=1e-3)


def test_rho_zero_makes_the_price_the_expectation():
    c = build("EnsembleCritic", rho=0.0, n_models=2)
    assert jnp.allclose(c.reduce_value(RAW), c.expectation_from_raw(RAW).mean(axis=0))


@pytest.mark.parametrize("name", CRITICS)
def test_ensemble_heads_are_independently_initialized(name, obs, act):
    """Ensemble's own docstring names this failure: drop `nnx.split_rngs` and
    every member draws the same init key, the heads collapse to identical, the
    spread goes to 0 and pessimism silently switches off for the whole run --
    nothing errors, the curve just looks plausible."""
    c = built(name)
    spread = c.expectation_from_raw(c.raw(obs, critic_action(c, act))).astype(jnp.float32)
    assert float(spread.std(axis=0).mean()) > 1e-3


# ----------------------------------------------------------------------
# The categorical projection
# ----------------------------------------------------------------------

ATOMS = jnp.linspace(-1.0, 1.0, 11)     # spacing 0.2, so 0.4 is exactly atom 7
NO_TERM = jnp.zeros((1,))


def _project(reward, probs=None, gamma=1.0):
    probs = jnp.zeros((1, 1, 11)).at[0, 0, 5].set(1.0) if probs is None else probs
    return categorical_projection(probs, ATOMS, jnp.array([reward]), NO_TERM,
                                  gamma, 0.0, -1.0, 1.0)


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

def _one_head(n, index):
    return jnp.zeros((1, 1, n)).at[0, 0, index].set(1.0)


@pytest.mark.parametrize("name", CRITICS)
def test_the_two_pessimism_axes_are_independent(name, obs, act):
    """`rho` prices the heads' disagreement into `value()`; `pessimistic_target`
    puts it in the Bellman target. They are separate because the references
    disagree about them separately -- dime_am wants neither, NC-LQL's C51 critic
    wants the first without the second -- and while they were one knob that
    combination could not be expressed at all.

    Independence is the pair of cross-terms: neither knob reaches the other's
    quantity. (`pessimistic_target` is *not* inert at rho=0 for a distributional
    critic -- it selects a head's whole categorical rather than reducing with a
    spread, so it never routes through `reduce_value`.)"""
    def priced(**over):
        c = build(name, **over)
        raw = c.raw(obs, critic_action(c, act))
        return (c, c.reduce_value(raw), c.mean_expectation(raw),
                c.bellman_target(jnp.zeros((BATCH,)), jnp.zeros((BATCH,)),
                                 c.bootstrap_from_raw(raw)))

    c, price, mean_exp, off = priced(rho=1.0, pessimistic_target=False)
    _, price_on, _, on = priced(rho=1.0, pessimistic_target=True)
    _, _, _, off_neutral = priced(rho=0.0, pessimistic_target=False)

    lift = lambda t: target_expectation(c, t)
    assert jnp.array_equal(price, price_on), "pessimistic_target moved the price"
    assert jnp.array_equal(off, off_neutral), "rho moved a non-pessimistic target"

    # ... and each knob does reach its own.
    assert float(lift(on).mean()) < float(lift(off).mean()) - 1e-4
    assert jnp.allclose(lift(off), c.gamma * mean_exp.squeeze(-1), atol=1e-3)


def test_pessimistic_target_transports_the_lower_heads_distribution():
    """Not a pointwise min over probabilities -- that is not a distribution.
    It picks the head with the lower E[Z] and moves *that* head's whole
    categorical, which is what keeps the target well-posed."""
    c = build("C51Critic", pessimistic_target=True, num_atoms=11, v_min=-1.0, v_max=1.0)
    support = c.support[...]
    boot = jnp.concatenate([_one_head(11, 8), _one_head(11, 2)], axis=0)   # E = +0.6, -0.6

    target = c.bellman_target(jnp.zeros((1,)), NO_TERM, boot)
    assert float(moments(support, target)[0][0]) == pytest.approx(c.gamma * -0.6, abs=1e-4)


def test_averaging_target_takes_both_heads():
    c = build("C51Critic", pessimistic_target=False, num_atoms=11, v_min=-1.0, v_max=1.0)
    boot = jnp.concatenate([_one_head(11, 8), _one_head(11, 2)], axis=0)

    target = c.bellman_target(jnp.zeros((1,)), NO_TERM, boot)
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

@pytest.mark.parametrize("name", NOISE_CONDITIONED)
def test_augment_widens_the_action_by_the_level_embedding(name, act):
    """The inner critic is built over `act_dim + time_dim`; `action_dim` stays
    the env's, as it means everywhere else."""
    c = built(name)
    assert c.action_dim == act.shape[-1]
    assert c.augment(act, 0).shape == (BATCH, act.shape[-1] + c.time_dim)
    assert not jnp.array_equal(c.augment(act, 0), c.augment(act, 3)), "levels embed identically"


@pytest.mark.parametrize("name", NOISE_CONDITIONED)
def test_smoothing_coefficient_is_actually_wired_in(name, obs, act, key):
    """`loss` folds TD at the clean level and smoothing of the
    noisier ones into one `losses["critic"]`, since both fit the same heads."""
    sigmas = jnp.array([0.5, 0.2, 0.05])
    c = build(name)                     # one build, two coefficients -- it is a plain float

    def loss_at(coef):
        c.smoothing_coef = coef
        losses = {}
        noise_td_loss(c, obs, act, obs[::-1], act, jnp.zeros((BATCH,)),
                                 jnp.zeros((BATCH,)), sigmas, key, losses)
        assert set(losses) == {"critic"}
        return losses["critic"]

    off, on = loss_at(0.0), loss_at(1.0)
    assert finite(off) and finite(on)
    assert not jnp.allclose(off, on), "smoothing_coef changed nothing"


# ----------------------------------------------------------------------
# What the metrics claim to measure
# ----------------------------------------------------------------------

def drain(name, obs, act) -> dict:
    """One critic loss under `recording`, and the diagnostics it sowed."""
    from flax import nnx

    c = build(name)
    a = critic_action(c, act)
    with recording(PRIMARY, DEBUG):
        td_loss(c, obs, a, obs[::-1], a, jnp.zeros((BATCH,)), jnp.zeros((BATCH,)), {})
        sown, sown_debug = nnx.pop(c, Metric, DebugMetric)
    return {**collect(sown), **collect(sown_debug)}


def test_pessimism_gap_tracks_the_reduction_that_produced_it():
    """It used to be reconstructed as `rho*std`, which is only one of the
    reductions here -- a `min` over heads has no rho, so every categorical
    critic logged a constant 0. Now it is `E[Z] - value()` itself: at rho=1 with
    two heads that is exactly the population std, and at rho=0 there is no gap
    at all, which is what a std-shaped stand-in could never report."""
    assert jnp.allclose(build("EnsembleCritic", rho=1.0, n_models=2).pessimism_gap(RAW),
                        RAW.astype(jnp.float32).std(axis=0), atol=1e-3)
    assert jnp.allclose(build("EnsembleCritic", rho=0.0, n_models=2).pessimism_gap(RAW),
                        0.0, atol=1e-6)


@pytest.mark.parametrize("name", CRITICS)
def test_every_critic_prices_no_higher_than_its_own_expectation(name, obs, act):
    """A gap is a configured choice, not a property of the class: dime_am's own
    reduction is the plain mean (`policy_q_reduce_fn = jnp.mean`), which is what
    the shipped `rho: 0.0` on the C51/HL-Gauss critics reproduces. So the
    invariant is one-sided, and strictness is keyed off `rho`."""
    c = built(name)
    gap = float(c.pessimism_gap(c.raw(obs, critic_action(c, act))).mean())
    assert gap >= -1e-4
    assert (gap > 1e-3) == (c.rho > 0.0)


def test_metric_families_separate_the_two_spreads(obs, act):
    """`ensemble/disagreement` is spread across heads (epistemic) and
    `return_std` is the predicted return distribution's own (aleatoric). Only a
    critic that predicts a distribution has the second -- before this, nothing
    logged it and a C51 run was indistinguishable from a scalar-ensemble run."""
    scalar = drain("EnsembleCritic", obs, act)
    categorical = drain("C51Critic", obs, act)

    for m in (scalar, categorical):
        assert m["ensemble/disagreement"] > 1e-3

    # Disagreement is the heads' spread whatever the reduction does with it;
    # the gap is what this critic's `rho` chooses to price in. EnsembleCritic
    # ships rho: 1.0, C51Critic rho: 0.0 (dime_am's mean).
    assert scalar["ensemble/pessimism_gap"] > 1e-3
    assert categorical["ensemble/pessimism_gap"] == pytest.approx(0.0, abs=1e-4)

    assert categorical["return_std/mean"] > 0.0
    assert not any(k.startswith("return_std") for k in scalar), "a scalar critic has no return spread"
