"""The components with no network in them -- squashes, schedules, the dual --
plus the two free functions the critics and optimizers are built out of.

All cheap: nothing here compiles a model, so this file is where it is
affordable to be exhaustive.
"""

import jax
import jax.numpy as jnp
import pytest
from conftest import SCHEDULES, SQUASHES, build, built, finite
from flax import nnx

from barl_jax.critics import categorical_projection
from barl_jax.optimization import CLIP_MODES, adam, grad_clip_transform

# ----------------------------------------------------------------------
# Squash
# ----------------------------------------------------------------------

X = jnp.linspace(-1.0, 1.0, 9).reshape(3, 3)


@pytest.mark.parametrize("name", SQUASHES)
def test_squash_maps_into_the_canonical_box(name):
    y = built(name)(X)
    assert y.shape == X.shape and finite(y)
    assert jnp.all(jnp.abs(y) <= 1.0)


@pytest.mark.parametrize("name", SQUASHES)
def test_squash_inverse_round_trips(name):
    sq = built(name)
    assert jnp.allclose(sq.inverse(sq(X)), X, atol=1e-3)


@pytest.mark.parametrize("name", SQUASHES)
def test_log_abs_det_jacobian_matches_autodiff(name):
    """Both squashes hand-roll a stable form of log|dy/dx| rather than letting
    autodiff near the saturated tail produce it; this is that form checked
    against the derivative it claims to be."""
    sq = built(name)
    slope = jax.grad(lambda z: sq(z).sum())(X)       # elementwise, since sq is elementwise
    assert jnp.allclose(sq.log_abs_det_jacobian(X), jnp.log(jnp.abs(slope)), atol=1e-4)


# ----------------------------------------------------------------------
# Schedule
# ----------------------------------------------------------------------

# ConstantSchedule is the one curve that ignores `end` -- holding at `start` is
# the whole of what it does, so the span and clamp contracts below exclude it.
SPANNING = [n for n in SCHEDULES if n != "ConstantSchedule"]


def test_constant_schedule_holds_at_start():
    s = built("ConstantSchedule")
    assert s.value(0.0) == s.value(1.0) == pytest.approx(s.start, abs=1e-6)
    assert s(s.steps * 10) == pytest.approx(s.start, abs=1e-6)


@pytest.mark.parametrize("name", SPANNING)
def test_schedule_spans_start_to_end(name):
    """Every curve shape shares one parameterization: `value` runs the whole way
    from `start` to `end` over normalized progress, whatever it does between."""
    s = built(name)
    assert s.value(0.0) == pytest.approx(s.start, abs=1e-6)
    assert s.value(1.0) == pytest.approx(s.end, abs=1e-6)


@pytest.mark.parametrize("name", SCHEDULES)
def test_schedule_is_monotone_between_its_endpoints(name):
    ladder = built(name).ladder(5)
    assert ladder.shape == (5,) and finite(ladder)
    assert jnp.all(jnp.diff(ladder) <= 1e-6), "descending endpoints, ascending curve"


@pytest.mark.parametrize("name", SPANNING)
def test_schedule_step_indexing_clamps_past_the_horizon(name):
    """`steps` is an estimate in a real run, so overshooting it stays flat at
    `end` rather than extrapolating."""
    s = built(name)
    assert s(0) == pytest.approx(s.start, abs=1e-6)
    assert s(s.steps) == pytest.approx(s.end, abs=1e-6)
    assert s(s.steps * 10) == pytest.approx(s.end, abs=1e-6)


@pytest.mark.parametrize("name", SCHEDULES)
def test_schedule_warmup_ramps_to_start(name):
    """Warmup is a linear ramp 0 -> `start`, stepped so the first step is not a
    wasted zero one."""
    s = build(name, warmup=2, steps=8)
    assert s(0) == pytest.approx(s.start * 0.5, abs=1e-6)
    assert s(1) == pytest.approx(s.start, abs=1e-6)


# ----------------------------------------------------------------------
# AutotuneRegulator
# ----------------------------------------------------------------------

def test_regulator_dual_has_the_sign_sac_needs():
    """d(loss)/d(log_alpha) is `alpha * (value - target)`, so under descent alpha
    falls while the measured quantity sits above target and rises below it."""
    r = build("AutotuneRegulator", alpha_init=0.5, target=-1.0)
    assert float(r.value()) == pytest.approx(0.5, abs=1e-6)

    above = jnp.array([1.0, -1.0])                      # mean 0.0, a full 1.0 above target
    losses = {}
    r.loss(above, losses)

    (key,) = losses
    assert key in r.create_optimizer(), "dual's loss key has no optimizer"

    gap = float((above - r.target_value).mean())
    assert gap == pytest.approx(1.0, abs=1e-6)
    assert float(losses[key]) == pytest.approx(0.5 * gap, abs=1e-6)


def test_regulator_dual_gradient_scales_with_alpha():
    """dime_am's form: d(loss)/d(log_alpha) is `alpha * (value - target)`, not the
    flat `value - target` a log_alpha coefficient would give. Same sign at every
    alpha -- the magnitude is what differs, and it is what sets how far alpha
    travels before the measured value pulls it back."""
    gap = jnp.array([2.0])

    def grad_at(alpha_init):
        r = build("AutotuneRegulator", alpha_init=alpha_init, target=0.0)

        def dual_loss(reg):
            losses = {}
            reg.loss(gap, losses)
            return next(iter(losses.values()))

        grads = nnx.grad(dual_loss)(r)
        return float(grads["log_alpha"][...])

    assert grad_at(1.0) == pytest.approx(2.0, rel=1e-5)
    assert grad_at(0.01) == pytest.approx(0.02, rel=1e-5)


def test_regulator_with_a_vector_target_gives_one_alpha_per_position():
    """`MaxEntRLPolicy`'s `alpha_granularity: per_action` passes a `(chunk_size,)`
    target instead of a float -- log_alpha/value/target_value all take that
    shape, with no separate switch on the regulator itself."""
    target = jnp.array([-1.0, -2.0, -3.0])
    r = build("AutotuneRegulator", alpha_init=0.5, target=target)

    assert r.log_alpha[...].shape == (3,)
    assert jnp.allclose(r.value(), 0.5)
    assert jnp.array_equal(r.target_value, target)


def test_regulator_vector_gradient_is_independent_per_position():
    """Each log_alpha_i's gradient comes from position i's own discrepancy
    only -- a batch of per-position values, reduced by `.mean()` over *both*
    the batch and chunk axes (`(alpha * (value - target)).mean()`), must not
    smear position i's signal onto position j's alpha."""
    target = jnp.array([0.0, 0.0])
    r = build("AutotuneRegulator", alpha_init=1.0, target=target)

    # Batch of 2, chunk_size 2: position 0 sits 1.0 above target, position 1
    # sits 3.0 above -- distinct gaps, so the two log_alpha grads must differ.
    value = jnp.array([[1.0, 3.0], [1.0, 3.0]])

    def dual_loss(reg):
        losses = {}
        reg.loss(value, losses)
        return next(iter(losses.values()))

    grads = nnx.grad(dual_loss)(r)["log_alpha"][...]
    # d(loss)/d(log_alpha_i) = alpha_i * mean_b(value_{b,i} - target_i) /
    # chunk_size (the /chunk_size falls out of the mean spanning both axes) --
    # at alpha_init=1.0 that is the position's own gap halved, not the other
    # position's.
    assert grads.shape == (2,)
    assert grads[0] == pytest.approx(0.5, rel=1e-5)
    assert grads[1] == pytest.approx(1.5, rel=1e-5)


def test_regulator_tune_false_pins_alpha_at_init():
    """`tune: false` keeps `log_alpha` in the loss/gradient path (so the loss-key
    contract `train_step` relies on is untouched) but backs its optimizer key
    with `optax.set_to_zero()`, so an update never moves it off `alpha_init`."""
    r = build("AutotuneRegulator", alpha_init=0.5, target=-1.0, tune=False)
    tx = next(iter(r.create_optimizer().values()))
    state = tx.init(r.log_alpha[...])

    grads = jnp.array(3.0)                                # a nonzero gradient
    updates, _ = tx.update(grads, state)
    assert float(updates) == pytest.approx(0.0, abs=1e-12)
    assert float(r.value()) == pytest.approx(0.5, abs=1e-6)


# ----------------------------------------------------------------------
# Free functions
# ----------------------------------------------------------------------

def test_categorical_projection_conserves_mass_and_expectation():
    """The C51 Bellman projection: still a distribution afterwards, and the
    expectation it lands on is the transported one -- provided the transported
    support stays inside [v_min, v_max], which is where it is a contraction."""
    atoms = jnp.linspace(-1.0, 1.0, 11)
    probs = jnp.full((2, 4, 11), 1 / 11)                # (n_models, B, atoms)
    reward, term, gamma = jnp.array([0.1, -0.1, 0.0, 0.2]), jnp.zeros((4,)), 0.5

    proj = categorical_projection(probs, atoms, reward, term, gamma, 0.0, -1.0, 1.0)

    assert proj.shape == probs.shape
    assert jnp.allclose(proj.sum(axis=-1), 1.0, atol=1e-5)
    expected = reward + gamma * (probs * atoms).sum(axis=-1)
    assert jnp.allclose((proj * atoms).sum(axis=-1), expected, atol=1e-5)


def test_categorical_projection_is_terminal_at_term_one():
    atoms = jnp.linspace(-1.0, 1.0, 11)
    probs = jnp.full((1, 3, 11), 1 / 11)
    reward = jnp.array([0.2, 0.0, -0.4])

    proj = categorical_projection(probs, atoms, reward, jnp.ones((3,)), 0.99, 0.0, -1.0, 1.0)
    assert jnp.allclose((proj * atoms).sum(axis=-1), reward, atol=1e-5)


# ----------------------------------------------------------------------
# Component optimizers
# ----------------------------------------------------------------------

CLIPPY = {"w": jnp.array([5.0, 0.1, -9.0])}
ZEROS  = {"w": jnp.zeros((3,))}


def _clipped(mode, grads=CLIPPY, tx=None):
    """What the clip hands on: these transforms are stateless and scale-only, so
    the update they emit *is* the clipped gradient."""
    tx = tx or grad_clip_transform(1.0, mode)
    updates, _ = tx.update(grads, tx.init(ZEROS), ZEROS)
    return updates["w"]


def test_global_norm_clip_preserves_direction_and_element_clip_does_not():
    """The distinction the two modes exist for. A global-norm clip rescales, so
    one outlier coordinate still dominates the update; the element-wise clip
    bounds each coordinate, which is what actually attenuates it."""
    g = CLIPPY["w"]
    gn, el = _clipped("global_norm"), _clipped("element")

    assert jnp.allclose(gn / jnp.linalg.norm(gn), g / jnp.linalg.norm(g), atol=1e-5)
    assert jnp.isclose(jnp.linalg.norm(gn), 1.0, atol=1e-5)
    assert jnp.allclose(el, jnp.array([1.0, 0.1, -1.0]), atol=1e-5)


def test_no_clip_mode_passes_the_gradient_through():
    assert jnp.allclose(_clipped("none"), CLIPPY["w"], atol=1e-5)


def test_adam_zeroes_nans_ahead_of_its_clip():
    """Order matters: a NaN reaching clip_by_global_norm makes the norm NaN and
    scales every coordinate, not just the bad one."""
    dirty = {"w": jnp.array([jnp.nan, 0.1, -9.0])}
    for mode in CLIP_MODES:
        u = _clipped(mode, dirty, tx=adam(1e-3, 0.9, 1.0, mode))
        assert finite(u) and jnp.all(jnp.abs(u) <= 1e-3 + 1e-9)


def test_unknown_clip_mode_is_rejected():
    with pytest.raises(KeyError, match="grad_clip_mode"):
        adam(1e-3, 0.9, 1.0, "sometimes")
