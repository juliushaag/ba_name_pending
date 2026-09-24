"""AMDP's weighted control objective, tested beyond shapes and finite losses."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import OBS_DIM, build, make_env_stats
from flax import nnx

from barl_jax.components import Buffer
from barl_jax.metrics import Recorder


def actor_for_test(*, steps=3, batch_rep=1, squash="ErfSquash"):
    stats = make_env_stats(act_dim=1, act_low=np.array([-1.0], np.float32),
                           act_high=np.array([1.0], np.float32), chunk_size=2)
    return build("ScoreContinuousActor", env_stats=stats, max_steps=steps, batch_rep=batch_rep,
                 squash_cls=squash, q_score_max_norm=1e6, q_score_max_percentile=100.0)


class LinearField(nnx.Module):
    """A time-independent affine controller, for closed-form gradient checks --
    the bridge-sampled matching target no longer depends on a discrete Euler
    step index, so this needn't vary with `scaled_time` either."""
    def __init__(self, gain, bias):
        self.gain = nnx.Param(jnp.asarray(gain, jnp.float32))
        self.bias = nnx.Param(jnp.asarray(bias, jnp.float32))

    def __call__(self, obs, x, scaled_time):
        return x @ self.gain[...].T + self.bias[...]


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


def assert_tree_close(a, b, **kwargs):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_allclose(x, y, **kwargs)


def test_scalar_alpha_and_equal_per_action_alphas_have_identical_actor_gradients():
    actor = actor_for_test(batch_rep=3)
    actor.field = LinearField(np.full((2, 2), 0.05), np.array([0.02, -0.01]))
    critic = QuadraticCritic([[0.3, 0.1], [0.1, 0.4]])
    obs = jnp.zeros((8, OBS_DIM))
    key = jax.random.PRNGKey(4)
    loss_grad = nnx.value_and_grad(lambda a, alpha: a.loss(obs, critic, alpha, key=key)["actor"])
    scalar = loss_grad(actor, jnp.asarray(0.7))
    vector = loss_grad(actor, jnp.full((2,), 0.7))
    assert_tree_close(scalar, vector, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("squash", ["ErfSquash", "TanhSquash"])
def test_loss_matches_manual_bridge_target(squash):
    """Recomputes `loss`'s bridge-sampled target from scratch -- same PRNG
    splits, same rollout, same closed-form score/entropy gradients -- and
    checks the actor gradient against the least-squares closed form.

    A time-independent field makes this exact for any number of Euler steps
    or `batch_rep` bridge samples: the rollout endpoint is stop-gradient'd
    before it reaches the score/entropy grads or the bridge sample, so
    `u_tilde`'s evaluation at each bridge point is the only path carrying
    gradient into `field` -- this incidentally also checks that isolation.
    """
    steps, rep, B = 2, 3, 5
    actor = actor_for_test(steps=steps, batch_rep=rep, squash=squash)
    gain = np.array([[0.2, -0.1], [0.05, 0.3]])
    bias = np.array([0.05, -0.04])
    actor.field = LinearField(gain, bias)
    matrix = np.array([[0.3, 0.1], [0.1, 0.4]])
    critic = QuadraticCritic(matrix)
    obs = jnp.zeros((B, OBS_DIM))
    alpha = jnp.array([0.4, 1.7])
    key = jax.random.PRNGKey(7)

    # Mirror loss()'s own key splits and rollout exactly.
    rollout_key, tau_key, noise_key = jax.random.split(key, 3)
    step_vars = np.asarray(actor.step_var[...])
    X = np.zeros((B, 2), np.float32)
    for i, step_key in enumerate(jax.random.split(rollout_key, steps)):
      v = step_vars[i]
      u = X @ gain.T + bias
      eps = np.asarray(jax.random.normal(step_key, X.shape))
      X = X + v * u + np.sqrt(v) * eps

    a = np.asarray(actor.squash(jnp.asarray(X)))
    if squash == "ErfSquash":
      k = actor.squash.erf_k
      jac = 2 * k / np.sqrt(np.pi) * np.exp(-(k * X) ** 2)
      entropy_grad = np.zeros_like(X)
    else:
      jac = 1 - a ** 2
      entropy_grad = np.asarray(alpha) * (X / actor.terminal_var - 2 * a)
    q_grad = -(a @ matrix) * jac
    score = q_grad + entropy_grad

    score_r = np.repeat(score, rep, axis=0)
    X_r = np.repeat(X, rep, axis=0)
    N = X_r.shape[0]
    tau = np.asarray(jax.random.uniform(tau_key, (N, 1), minval=actor.config.tau_min, maxval=actor.config.tau_max))
    noise = np.asarray(jax.random.normal(noise_key, X_r.shape))
    sigma_sq = np.asarray(actor.sigma_sq_int_fn(jnp.asarray(tau)))
    ratio = sigma_sq / actor.terminal_var
    std = np.sqrt(np.maximum(sigma_sq * (1.0 - ratio), 0.0))
    a_t = ratio * X_r + std * noise
    sigma_tau = np.asarray(actor.sigma_fn(jnp.asarray(tau)))
    u = sigma_tau * (a_t @ gain.T + bias)
    target = sigma_tau * (score_r / np.asarray(alpha))
    r = u - target
    expected_gain = np.mean(r[:, :, None] * a_t[:, None, :], axis=0)
    expected_bias = np.mean(r, axis=0)

    actor_grad, critic_grad, alpha_grad = nnx.grad(
      lambda a_, c, t: a_.loss(obs, c, t, key=key)["actor"], argnums=(0, 1, 2)
    )(actor, critic, alpha)
    np.testing.assert_allclose(actor_grad["field"]["gain"][...], expected_gain, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(actor_grad["field"]["bias"][...], expected_bias, rtol=2e-5, atol=2e-6)
    assert all(np.count_nonzero(g) == 0 for g in jax.tree.leaves(critic_grad))
    np.testing.assert_array_equal(alpha_grad, jnp.zeros_like(alpha))


@pytest.mark.parametrize("granularity", ["per_chunk", "per_action"])
def test_amdp_policy_updates_with_chunked_alphas(granularity):
    from test_policies import FakeEnv, batch

    from barl_jax.components import TRAINABLE
    from barl_jax.config.run import RunConfig
    from barl_jax.metrics import DEBUG, PRIMARY
    from barl_jax.optimization import train_step
    from barl_jax.policy import Policy

    run = RunConfig.from_dict({
        "model": "AMDP", "env": "HalfCheetah",
        "model_overrides": {
            "params": {"chunk_size": 2, "alpha_granularity": granularity},
            "components": {
                "actor": {"latent_dim": 8},
                "critic": {"latent_dim": 8},
            },
        },
    })
    policy = Policy.make("AMDP", FakeEnv(), run.model_config, env_stats=make_env_stats(), seed=0)
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
