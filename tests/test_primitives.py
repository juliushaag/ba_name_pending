"""The components with no network in them -- squashes, schedules, the dual --
plus the two free functions the critics and optimizers are built out of.

All cheap: nothing here compiles a model, so this file is where it is
affordable to be exhaustive.
"""

import jax
import jax.numpy as jnp
import optax
import pytest

from barl_jax.networks.critic import categorical_projection
from barl_jax.networks.optim import every_k

from conftest import SCHEDULES, SQUASHES, build, built, finite

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
    """d(loss)/d(log_alpha) is exactly `value - target`, so under descent alpha
    falls while the measured quantity sits above target and rises below it."""
    r = build("AutotuneRegulator", alpha_init=0.5, target=-1.0)
    assert float(r.value()) == pytest.approx(0.5, abs=1e-6)

    above = jnp.array([1.0, -1.0])                      # mean 0.0, a full 1.0 above target
    losses = {}
    r.loss(above, losses)

    (key,) = losses
    assert key in r.create_optimizer(), "dual's loss key has no optimizer"

    # loss = log_alpha * (value - target).mean(), so d/d(log_alpha) is that mean:
    # positive while the measured quantity sits above target, which is exactly
    # what makes gradient *descent* lower alpha there.
    gap = float((above - r.target_value).mean())
    assert gap == pytest.approx(1.0, abs=1e-6)
    assert float(losses[key]) == pytest.approx(float(r.log_alpha[...]) * gap, abs=1e-6)


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


def test_every_k_drops_the_steps_it_skips():
    """Not optax.MultiSteps: the skipped gradients are discarded rather than
    accumulated, and the inner state is frozen so momentum cannot leak through."""
    tx = every_k(optax.sgd(1.0), 3)
    params = {"w": jnp.zeros((2,))}
    state = tx.init(params)
    grads = {"w": jnp.ones((2,))}

    moved = []
    for _ in range(6):
        updates, state = tx.update(grads, state, params)
        moved.append(bool(jnp.any(updates["w"] != 0)))

    assert moved == [False, False, True, False, False, True]


def test_every_k_is_a_no_op_at_k_of_one():
    inner = optax.sgd(1.0)
    assert every_k(inner, 1) is inner
