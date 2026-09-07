"""AMDP's weighted control objective, tested beyond shapes and finite losses."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import OBS_DIM, build, make_env_stats
from flax import nnx

from barl_jax.components import Buffer
from barl_jax.metrics import Recorder


def actor_for_test(*, steps=3, squash="ErfSquash"):
    stats = make_env_stats(act_dim=1, act_low=np.array([-1.0], np.float32),
                           act_high=np.array([1.0], np.float32), chunk_size=2)
    return build("ScoreContinuousActor", env_stats=stats, max_steps=steps,
                 squash_cls=squash, q_score_max_norm=1e6, q_score_max_percentile=100.0)


class LinearField(nnx.Module):
    """A separate affine controller at each Euler time, for analytic oracles."""
    def __init__(self, gains):
        self.gains = nnx.Param(jnp.asarray(gains, jnp.float32))
        self.bias = nnx.Param(jnp.zeros(gains.shape[:2], jnp.float32))

    def __call__(self, obs, x, scaled_time):
        n = self.gains[...].shape[0]
        k = jnp.rint((scaled_time[:, 0] + 1) * n / 2).astype(jnp.int32)
        return jnp.einsum("bij,bj->bi", self.gains[...][k], x) + self.bias[...][k]


class QuadraticCritic(nnx.Module):
    def __init__(self, curvature):
        self.curvature = Buffer(jnp.asarray(curvature, jnp.float32))
        self.scale = nnx.Param(jnp.asarray(1.0))

    def value(self, obs, action):
        return -0.5 * self.scale[...] * jnp.einsum(
            "bi,ij,bj->b", action, self.curvature[...], action)[:, None]

    @property
    def metrics(self):
        return Recorder(self)


class IdentitySquash(nnx.Module):
    def __call__(self, x):
        return x

    def log_abs_det_jacobian(self, x):
        return jnp.zeros_like(x)


def assert_tree_close(a, b, **kwargs):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_allclose(x, y, **kwargs)


def test_training_pair_uses_the_actual_controlled_path():
    actor = actor_for_test()
    gains = np.broadcast_to([[0.1, 0.2], [-0.3, 0.15]], (3, 2, 2))
    actor.field = LinearField(gains)
    actor.field.bias[...] = jnp.full((3, 2), 0.1)
    obs = jnp.zeros((24, OBS_DIM))
    key = jax.random.PRNGKey(3)
    rollout_key, time_key = jax.random.split(key)
    indices = jax.random.randint(time_key, (24,), 0, 3)
    states = []
    x = jnp.zeros((24, 2))
    for k, step_key in enumerate(jax.random.split(rollout_key, 3)):
        states.append(x)
        c = x @ jnp.asarray(gains[k]).T + 0.1
        v = actor.step_var[...][k]
        x = x + v * c + jnp.sqrt(v) * jax.random.normal(step_key, x.shape)
    end, mid, tau = actor.sample_training_pair(obs, key)
    np.testing.assert_allclose(end, x, atol=1e-6)
    np.testing.assert_allclose(mid, jnp.stack(states)[indices, jnp.arange(24)], atol=1e-6)
    np.testing.assert_allclose(tau, actor.taus[...][indices])
    sampled, _ = actor.generate_action(obs, squash=False, key=rollout_key)
    np.testing.assert_array_equal(end, sampled.reshape(24, 2))

    grads = nnx.grad(lambda a: sum(x.sum() for x in a.sample_training_pair(obs, key)))(actor)
    assert all(np.count_nonzero(g) == 0 for g in jax.tree.leaves(grads))


def test_scalar_alpha_and_equal_per_action_alphas_have_identical_actor_gradients():
    actor = actor_for_test()
    actor.field = LinearField(np.full((3, 2, 2), 0.05))
    critic = QuadraticCritic([[0.3, 0.1], [0.1, 0.4]])
    obs = jnp.zeros((8, OBS_DIM))
    key = jax.random.PRNGKey(4)
    loss_grad = nnx.value_and_grad(lambda a, alpha: a.loss(obs, critic, alpha, key=key)["actor"])
    scalar = loss_grad(actor, jnp.asarray(0.7))
    vector = loss_grad(actor, jnp.full((2,), 0.7))
    assert_tree_close(scalar, vector, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("squash", ["ErfSquash", "TanhSquash"])
def test_weighted_terminal_gradient_and_parameter_isolation(squash):
    actor = actor_for_test(steps=1, squash=squash)
    actor.field = LinearField(np.zeros((1, 2, 2)))
    actor.field.bias[...] = jnp.array([[0.05, -0.04]])
    matrix = jnp.array([[0.3, 0.1], [0.1, 0.4]])
    critic = QuadraticCritic(matrix)
    obs = jnp.zeros((8, OBS_DIM))
    alpha = jnp.array([0.4, 1.7])
    key = jax.random.PRNGKey(5)
    x, _, _ = actor.sample_training_pair(obs, key)
    a = actor.squash(x)
    if squash == "ErfSquash":
        k = actor.squash.erf_k
        jac = 2 * k / np.sqrt(np.pi) * jnp.exp(-(k * x) ** 2)
        entropy_grad = jnp.zeros_like(x)
    else:
        jac = 1 - a ** 2
        entropy_grad = alpha * (x / actor.terminal_var - 2 * a)
    q_grad = -(a @ matrix) * jac
    expected = actor.sigma_fn(jnp.asarray(0.0)) * (
        actor.field.bias[...] - ((q_grad + entropy_grad) / alpha).mean(axis=0))
    actor_grad, critic_grad, alpha_grad = nnx.grad(
        lambda a, c, t: a.loss(obs, c, t, key=key)["actor"], argnums=(0, 1, 2)
    )(actor, critic, alpha)
    np.testing.assert_allclose(actor_grad["field"]["bias"][...], expected,
                               rtol=2e-5, atol=2e-6)
    assert all(np.count_nonzero(g) == 0 for g in jax.tree.leaves(critic_grad))
    np.testing.assert_array_equal(alpha_grad, jnp.zeros_like(alpha))


def test_unequal_alpha_matching_is_stationary_at_coupled_lqr_optimum(monkeypatch):
    """An independent Riccati solution, integrated exactly for quadratic data.

    Unequal A and off-diagonal terminal curvature give a nonconservative
    optimal control. It must still be a fixed point of the matching update.
    """
    actor = actor_for_test()
    actor.squash = IdentitySquash()
    alpha = jnp.array([0.4, 1.7])
    A = np.diag(np.asarray(alpha))
    terminal = np.array([[0.35, 0.15], [0.15, 0.5]])
    curvature = terminal.copy()
    gains = []
    for v in reversed(np.asarray(actor.step_var[...])):
        gain = -np.linalg.solve(A + v * curvature, curvature)
        gains.append(gain)
        curvature = curvature + v * curvature @ gain
    actor.field = LinearField(np.stack(gains[::-1]))
    # Remove the Gaussian-prior terminal term through the critic, leaving
    # exactly g(x) = x.T @ terminal @ x / 2 (up to a constant).
    critic = QuadraticCritic(terminal + A / actor.terminal_var)
    assert not np.allclose(gains[0], gains[0].T)

    n, d = actor.max_steps, actor.plan_dim
    m = n * d
    points = np.concatenate([np.eye(m), -np.eye(m)]) * np.sqrt(m)
    # Repeat a full cubature rule for each selected training time. Each rule
    # has E[eps]=0 and E[eps eps.T]=I across every step and coordinate.
    noises = jnp.asarray(np.tile(points, (n, 1)).reshape(n * 2 * m, n, d))
    selected = jnp.repeat(jnp.arange(n), 2 * m)
    obs = jnp.zeros((len(selected), OBS_DIM))
    key = jax.random.PRNGKey(6)
    rollout_key, _ = jax.random.split(key)
    step_keys = jax.random.split(rollout_key, n)

    def cubature_normal(key, shape, dtype=jnp.float32):
        index = jnp.argmax(jnp.all(step_keys == key, axis=-1))
        return noises[:, index, :].astype(dtype)

    monkeypatch.setattr(jax.random, "normal", cubature_normal)
    monkeypatch.setattr(jax.random, "randint", lambda *args, **kwargs: selected)
    gradient = nnx.grad(lambda a: a.loss(obs, critic, alpha, key=key)["actor"])
    grads = gradient(actor)
    assert all(np.max(np.abs(g)) < 3e-6 for g in jax.tree.leaves(grads))
    actor.field.gains[...] = actor.field.gains[...] + 0.1
    perturbed = gradient(actor)
    assert any(np.max(np.abs(g)) > 1e-3 for g in jax.tree.leaves(perturbed))


@pytest.mark.parametrize("granularity", ["per_chunk", "per_action"])
def test_amdp_policy_updates_with_chunked_alphas(granularity):
    from test_policies import FakeEnv, batch

    from barl_jax.components import TRAINABLE
    from barl_jax.config.tree import ConfigTree
    from barl_jax.metrics import DEBUG, PRIMARY
    from barl_jax.optimization import train_step
    from barl_jax.policy import Policy

    config = ConfigTree(overrides={
        "params.chunk_size": 2, "params.alpha_granularity": granularity,
        "components.actor.latent_dim": 8, "components.critic.latent_dim": 8,
    })
    policy = Policy.make("AMDP", FakeEnv(), config=config, env_stats=make_env_stats(), seed=0)
    policy.entropy.log_alpha[...] = jnp.log(
        jnp.array([0.4, 1.7]) if granularity == "per_action" else jnp.asarray(0.7))
    before_actor = jax.tree.leaves(nnx.state(policy.actor, nnx.Param))
    before_alpha = np.asarray(policy.alpha())
    optimizer = nnx.Optimizer(policy, policy.build_optimizer(lr_schedule=None), wrt=TRAINABLE)
    graphdef, state = nnx.split((policy, optimizer))
    data = batch()
    data["act"] = jnp.tile(data["act"], (1, 2))
    data["horizon"] = jnp.full_like(data["horizon"], 2)
    for step in range(policy.policy_delay + 1):
        state, losses, _, _ = train_step(graphdef, state, data, (PRIMARY, DEBUG), jnp.asarray(step))
        assert all(np.isfinite(v) for v in losses.values())
    updated = nnx.merge(graphdef, state)[0]
    assert not np.array_equal(before_alpha, np.asarray(updated.alpha()))
    assert any(not np.array_equal(a, b) for a, b in zip(
        before_actor, jax.tree.leaves(nnx.state(updated.actor, nnx.Param)), strict=True))
