"""The paper-faithful SDAC (`actors/sdac_paper.py`): the schedule's algebra, the
sampler's contracts, and which loss reaches which parameter."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import ACT_DIM, BATCH, build, built, finite
from flax import nnx

from barl_jax.actors.sdac_paper import (
  POST_LOGVAR,
  SQRT_ALPHA_BAR,
  SQRT_ONE_MINUS,
  SQRT_RECIP,
  SQRT_RECIPM1,
  diffusion_coefficients,
)
from barl_jax.components import TRAINABLE


def test_schedule_identities():
  c = diffusion_coefficients(20, 0.8, 1e-4, 0.999).astype(np.float64)
  assert np.allclose(c[SQRT_ALPHA_BAR] ** 2 + c[SQRT_ONE_MINUS] ** 2, 1.0, atol=1e-5)
  # 1/alpha_bar = 1 + (1/alpha_bar - 1); the table is float32 and reaches ~4e2, so compare relatively.
  assert np.allclose(c[SQRT_RECIP] ** 2, c[SQRT_RECIPM1] ** 2 + 1.0, rtol=1e-4)
  assert np.all(np.isfinite(c)) and np.all(np.diff(c[SQRT_ALPHA_BAR]) < 0)
  assert c[POST_LOGVAR, 0] == pytest.approx(np.log(1e-20), rel=1e-5)


def test_x0_is_recovered_from_the_forward_marginal_at_the_true_eps():
  a = built("PaperDDPMActor")
  coef = a.coef[...]
  x0 = jax.random.uniform(jax.random.PRNGKey(0), (6, ACT_DIM), minval=-1, maxval=1)
  eps = jax.random.normal(jax.random.PRNGKey(1), (6, ACT_DIM))
  for n in range(a.config.num_steps):
    x_t = coef[SQRT_ALPHA_BAR, n] * x0 + coef[SQRT_ONE_MINUS, n] * eps
    assert jnp.allclose(x_t * coef[SQRT_RECIP, n] - eps * coef[SQRT_RECIPM1, n], x0, atol=2e-3)


class LinearCritic(nnx.Module):
  """value(s, a) = a[:, 0], so the best candidate is the one with the largest first coordinate."""
  def value(self, s, a):
    return a[:, :1]


def test_actions_stay_in_the_box_with_and_without_a_critic(obs, key):
  a = built("PaperDDPMActor")
  for critic in (None, LinearCritic()):
    action, entropy = a.generate_action(obs, critic=critic, key=key)
    assert action.shape == (BATCH, a.chunk_size, ACT_DIM) and finite(action)
    assert jnp.all(jnp.abs(action) <= 1.0) and entropy is None


def test_best_of_n_prefers_the_higher_value_candidate(key):
  a = build("PaperDDPMActor")
  big = jnp.tile(jax.random.normal(jax.random.PRNGKey(5), (1, a.obs_dim)), (512, 1))
  first = lambda n: dataclasses.replace(a.config, num_particles=n)
  a.config = first(1)
  one = a.generate_action(big, critic=LinearCritic(), key=key)[0][:, 0, 0].mean()
  a.config = first(32)
  best = a.generate_action(big, critic=LinearCritic(), key=key)[0][:, 0, 0].mean()
  assert best > one


def test_exploration_noise_scales_with_alpha_and_noise_scale(obs, key):
  a = build("PaperDDPMActor")
  a.field.zero_init_head()
  a.config = dataclasses.replace(a.config, noise_scale=1e-9)
  quiet = a.sample(obs, key=key)
  a.config = dataclasses.replace(a.config, noise_scale=0.1)
  loud = a.sample(obs, key=key)
  assert jnp.abs(loud - quiet).max() > 1e-3
  a.log_alpha[...] = jnp.log(jnp.asarray(1e-6))
  assert jnp.abs(a.sample(obs, key=key) - quiet).max() < 1e-3


def test_alpha_loss_descends_towards_the_target_entropy():
  """loss = log_alpha * (H(alpha) - target): alpha falls above the target and rises below it,
  and the gradient vanishes where the noise entropy equals the target."""
  a = build("PaperDDPMActor")
  critic, cfg, d = build("DirectCritic"), a.config, a.plan_dim
  obs = jnp.zeros((2, a.obs_dim))

  def grad_at(alpha):
    a.log_alpha[...] = jnp.log(jnp.asarray(alpha, jnp.float32))
    g = nnx.grad(lambda m: m.loss(obs, critic, 1.0, key=jax.random.PRNGKey(0))["alpha"],
                 argnums=nnx.DiffState(0, TRAINABLE))(a)
    return float(g.log_alpha[...])

  balanced = np.sqrt(np.exp(-2 * cfg.target_entropy_scale) / (2 * np.pi * np.e)) / cfg.noise_scale
  assert grad_at(cfg.alpha_init) > 0 > grad_at(balanced / 10)
  assert abs(grad_at(balanced)) < 1e-3 * d


def test_each_loss_reaches_only_its_own_parameters(obs, key):
  a = built("PaperDDPMActor")
  critic = build("DirectCritic")

  def grads(name):
    g = nnx.grad(lambda m: m.loss(obs, critic, 1.0, key=key)[name],
                 argnums=nnx.DiffState(0, TRAINABLE))(a)
    return g.log_alpha[...], jax.tree.leaves(nnx.state(g.field))

  alpha_of_actor, field_of_actor = grads("actor")
  alpha_of_alpha, field_of_alpha = grads("alpha")
  assert alpha_of_actor == 0 and any(jnp.any(x != 0) for x in field_of_actor)
  assert alpha_of_alpha != 0 and all(jnp.all(x == 0) for x in field_of_alpha)
  assert all(finite(x) for x in field_of_actor)


def test_policy_declares_a_cadence_per_optimizer_key():
  from test_policies import make
  policy = make("SDACPaper")
  cadence = policy.optimizer_cadence()
  assert cadence == {"actor": policy.policy_delay, "alpha": policy.params.delay_alpha_update}
  assert set(cadence) <= set(policy.create_optimizer())
