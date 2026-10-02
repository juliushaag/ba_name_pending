"""The learned categorical value support (DySEL): a Lagrangian, switchable per head."""

import math

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import ACT_DIM, BATCH, OBS_DIM, build, finite, make_env_stats
from flax import nnx

from barl_jax.components import TRAINABLE
from barl_jax.config.run import RunConfig
from barl_jax.optimization import train_step
from barl_jax.policy import Policy

ONE = jnp.ones((1,))
HEADS = ["HLGaussCritic", "C51Critic"]


def learned(head, **extra):
  return build(head, learn_support=True, **extra)


def boot(critic, batch=1, atom=None):
  """A bootstrap in the head's own form: a scalar for HL-Gauss, a distribution for C51."""
  if critic.config.class_name == "HLGaussCritic":
    return jnp.zeros((batch, 1))
  k = critic.config.num_bins
  probs = jnp.full((batch, k), 1.0 / k)
  return probs if atom is None else jnp.zeros((batch, k)).at[:, atom].set(1.0)


def gauss_leak(y, sigma, lo, hi):
  cdf = lambda x: 0.5 * (1 + math.erf((x - y) / (sigma * math.sqrt(2))))
  return cdf(lo) + 1 - cdf(hi)


@pytest.mark.parametrize("head", HEADS)
def test_support_is_a_fixed_buffer_when_switched_off(head):
  critic = build(head, learn_support=False)
  assert set(critic.create_optimizer()) == {"critic"}
  assert not hasattr(critic, "support_net") and not hasattr(critic, "dual")
  assert critic.support_losses(ONE, ONE, ONE, boot(critic)) == {}


@pytest.mark.parametrize("head", HEADS)
def test_learned_support_is_the_default(head):
  assert hasattr(build(head), "support_net")


@pytest.mark.parametrize("head", HEADS)
def test_learned_support_adds_its_own_optimizer_keys_and_parameters(head):
  critic = learned(head)
  assert set(critic.create_optimizer()) == {"critic", "support", "support_dual"}
  assert critic.param_labels(("support_net", "endpoints")) == "support"
  assert critic.param_labels(("dual",)) == "support_dual"
  assert critic.param_labels(("net", "ensemble")) is None
  np.testing.assert_allclose(critic.support_net.bounds(), (-10.0, 10.0))


def test_hlgauss_geometry_follows_the_endpoints():
  critic = learned("HLGaussCritic", num_bins=10)
  critic.support_net.endpoints[...] = jnp.asarray([4.0, -6.0])    # unsorted on purpose
  edges, _, sigma = critic._geometry(*critic.support_net.bounds())
  np.testing.assert_allclose(edges, np.linspace(-6, 4, 11), atol=1e-5)
  np.testing.assert_allclose(critic.support_values(), np.linspace(-5.5, 3.5, 10), atol=1e-5)
  assert float(sigma) == pytest.approx(max(0.75 * 1.0, 0.3))


def test_c51_atoms_span_the_endpoints_inclusive():
  critic = learned("C51Critic", num_bins=11)
  critic.support_net.endpoints[...] = jnp.asarray([4.0, -6.0])
  edges, _, sigma = critic._geometry(*critic.support_net.bounds())
  assert edges is None
  np.testing.assert_allclose(critic.support_values(), np.linspace(-6, 4, 11), atol=1e-5)
  assert float(sigma) == pytest.approx(max(0.75 * 1.0, 0.3))


@pytest.mark.parametrize("head", HEADS)
def test_sigma_is_floored_in_absolute_units(head):
  critic = learned(head, num_bins=100, support_init_min=-1.0, support_init_max=1.0)
  assert float(critic._geometry(*critic.support_net.bounds())[2]) == pytest.approx(0.3)


@pytest.mark.parametrize("head", HEADS)
@pytest.mark.parametrize("y", [-3.0, 0.0, 4.5, 9.0, 12.0])
def test_leak_is_the_gaussian_mass_outside_the_support(head, y):
  """A terminal transition has discount 0, so every shifted atom lands on y and
  both heads' leaks are the one Gaussian's mass outside the support."""
  critic = learned(head)
  sigma = float(critic._geometry(*critic.support_net.bounds())[2])
  losses = critic.support_losses(jnp.array([y]), jnp.ones((1,)), ONE, boot(critic))
  leak = -float(losses["support_dual"]) + critic.config.support_eps   # lambda = dual_init = 1
  assert leak == pytest.approx(gauss_leak(y, sigma, -10.0, 10.0), abs=1e-6)


def test_c51_leak_is_a_mixture_over_the_shifted_atoms():
  critic = learned("C51Critic")
  atoms = np.asarray(critic.support_values())
  lo, hi = -10.0, 10.0
  sigma = float(critic._geometry(lo, hi)[2])
  probs = jnp.zeros((1, atoms.size)).at[0, 0].set(0.25).at[0, -1].set(0.75)
  reward = 2.0
  losses = critic.support_losses(jnp.array([reward]), jnp.zeros((1,)), ONE, probs)
  leak = -float(losses["support_dual"]) + critic.config.support_eps
  gamma = critic.config.gamma
  expected = sum(p * gauss_leak(reward + gamma * atoms[i], sigma, lo, hi)
                 for p, i in ((0.25, 0), (0.75, -1)))
  assert leak == pytest.approx(expected, abs=1e-6)


def test_c51_projection_follows_the_moved_atoms():
  critic = learned("C51Critic", num_bins=21)
  critic.support_net.endpoints[...] = jnp.asarray([-2.0, 6.0])
  atoms = np.asarray(critic.support_values())
  j = 14
  probs = jnp.zeros((1, atoms.size)).at[0, j].set(1.0)
  target = critic.bellman_target(jnp.array([0.1]), jnp.zeros((1,)), ONE, probs)
  assert float(target.sum()) == pytest.approx(1.0, abs=1e-5)
  mean = float((target * critic.support_values()).sum())
  assert mean == pytest.approx(0.1 + critic.config.gamma * atoms[j], abs=1e-4)


def grads_for(critic, y):
  def total(critic):
    losses = critic.support_losses(jnp.full((BATCH,), y), jnp.ones((BATCH,)), jnp.ones((BATCH,)),
                                   boot(critic, BATCH))
    return sum(losses.values())
  return nnx.grad(total)(critic)


@pytest.mark.parametrize("head", HEADS)
def test_violated_constraint_widens_the_support_and_raises_the_multiplier(head):
  critic = learned(head)
  grads = grads_for(critic, y=10.3)         # one sigma beyond v_max = 10: mass leaks
  lo_grad, hi_grad = np.asarray(grads.support_net.endpoints[...])
  assert hi_grad < 0                        # descending the loss raises v_max
  assert float(grads.dual[...]) < 0         # descending the *negated* term raises lambda


@pytest.mark.parametrize("head", HEADS)
def test_slack_constraint_shrinks_the_support_and_lowers_the_multiplier(head):
  critic = learned(head)
  grads = grads_for(critic, y=0.0)          # nothing leaks, leak < eps
  lo_grad, hi_grad = np.asarray(grads.support_net.endpoints[...])
  assert hi_grad > 0 and lo_grad < 0        # the width penalty pulls both inward
  assert float(grads.dual[...]) > 0


@pytest.mark.parametrize("head", HEADS)
def test_cross_entropy_does_not_move_the_support_or_multiplier(head):
  critic = learned(head)
  s, a = jnp.zeros((BATCH, OBS_DIM)), jnp.zeros((BATCH, ACT_DIM))
  def ce(critic):
    raw = critic.raw(s, a)
    target = critic.bellman_target(jnp.ones((BATCH,)), jnp.zeros((BATCH,)), jnp.ones((BATCH,)),
                                   boot(critic, BATCH))
    return critic.regression_loss(raw, target)
  grads = nnx.grad(ce)(critic)
  assert not np.any(np.asarray(grads.support_net.endpoints[...]))
  assert float(grads.dual[...]) == 0


@pytest.mark.parametrize("head", HEADS)
def test_cross_entropy_is_scaled_by_inverse_alpha(head):
  plain = build(head, learn_support=False)
  scaled = build(head, learn_support=True, support_alpha=0.25)
  raw = jax.random.normal(jax.random.key(0), (2, 5, scaled.config.num_bins))
  target = jax.nn.softmax(jax.random.normal(jax.random.key(1), (5, scaled.config.num_bins)))
  np.testing.assert_allclose(scaled.regression_loss(raw, target),
                             plain.regression_loss(raw, target) / 0.25, rtol=1e-5)


def sac_policy(head, **critic):
  run = RunConfig.from_dict({
    "model": "SAC", "env": "HalfCheetah",
    "model_overrides": {"components": {
      "actor": {"latent_dim": 8, "mlp": {"dtype": "float32"}},
      "critic": {"class_name": head, "latent_dim": 8, "learn_support": True,
                 "mlp": {"dtype": "float32"}, **critic}}},
  })

  class Env:
    observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)

  return Policy.make("SAC", Env(), run.model_config, env_stats=make_env_stats(), seed=0)


def data(scale):
  rng = np.random.default_rng(0)
  f = lambda *s: jnp.asarray(rng.normal(size=s), jnp.float32)
  return {"obs": f(BATCH, OBS_DIM), "next_obs": f(BATCH, OBS_DIM),
          "act": jnp.clip(f(BATCH, ACT_DIM), -1, 1), "chunk_return": scale + 0 * f(BATCH),
          "term": jnp.ones((BATCH,), jnp.float32), "horizon": jnp.ones((BATCH,), jnp.int32)}


@pytest.mark.parametrize("head", HEADS)
def test_policy_routes_each_parameter_group_to_its_own_optimizer(head):
  policy = sac_policy(head)
  labels = {".".join(map(str, path)): key
            for path, key in nnx.to_flat_state(policy.optimizer_labels())}
  assert labels["critic.support_net.endpoints"] == "support"
  assert labels["critic.dual"] == "support_dual"
  assert {k for k in labels.values()} >= {"critic", "support", "support_dual", "actor"}


@pytest.mark.parametrize("head", HEADS)
def test_training_expands_the_support_towards_returns_it_cannot_cover(head):
  policy = sac_policy(head, support_lr=0.05)
  optimizer = nnx.Optimizer(policy, policy.build_optimizer(lr_schedule=None), wrt=TRAINABLE)
  graphdef, state = nnx.split((policy, optimizer))
  batch = data(scale=10.5)                  # terminal returns just beyond v_max = 10
  for step in range(300):
    state, losses, *_ = train_step(graphdef, state, batch, (), jnp.asarray(step))
  assert set(losses) == {"critic", "actor", "entropy_alpha", "support", "support_dual"}
  trained, _ = nnx.merge(graphdef, state)
  lo, hi = trained.critic.support_net.bounds()
  assert float(hi) > 10.0 and finite(jnp.stack([lo, hi]))
  assert float(trained.critic.dual[...]) >= 0.0
