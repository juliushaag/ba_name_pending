"""Time encoder dimensions, numerical schedules, and actor time conventions."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import ACT_DIM, BATCH, OBS_DIM, build, finite
from flax import nnx

from barl_jax.config.run import RunConfig
from barl_jax.networks.diffusion import controlled_sde, reverse_chain
from barl_jax.networks.time_embedding import (
  CosTimeConfig,
  FourierTimeConfig,
  SinusoidalTimeConfig,
)


@pytest.mark.parametrize("dim", [2, 3, 4, 5, 6, 7, 8])
def test_sinusoidal_dimensions_are_finite_and_keep_leading_axes(dim):
  embedding = SinusoidalTimeConfig(time_dim=dim).build(rngs=nnx.Rngs(0))
  t = jnp.array([[[0.0], [0.2]], [[0.7], [1.0]]])
  result = embedding(t)
  assert result.shape == (2, 2, dim) and finite(result)
  np.testing.assert_allclose(jax.jit(embedding)(t), result, rtol=2e-5, atol=1e-5)
  assert jnp.any(embedding(jnp.array([[0.2]])) != embedding(jnp.array([[0.7]])))
  half = dim // 2
  np.testing.assert_array_equal(result[0, 0, :half], 0)
  np.testing.assert_array_equal(result[0, 0, half:2 * half], 1)
  if dim % 2:
    np.testing.assert_array_equal(result[..., -1], 0)


def test_sinusoidal_preserves_existing_even_width_frequency_convention():
  embedding = SinusoidalTimeConfig(time_dim=6).build(rngs=nnx.Rngs(0))
  t = np.array([[0.1], [0.7]], np.float32)
  angle = 1000 * t * np.array([1.0, 0.01, 0.0001], np.float32)
  np.testing.assert_allclose(embedding(jnp.asarray(t)),
                             np.concatenate([np.sin(angle), np.cos(angle)], -1), atol=1e-5)


@pytest.mark.parametrize("spacing", ["geometric", "linear"])
@pytest.mark.parametrize("learn_freqs", [False, True])
def test_fourier_frequency_grid_time_gradients_and_parameter_ownership(spacing, learn_freqs):
  config = FourierTimeConfig(time_dim=5, num_freqs=4, spacing=spacing, learn_freqs=learn_freqs)
  embedding = config.build(rngs=nnx.Rngs(0))
  expected = (np.geomspace if spacing == "geometric" else np.linspace)(0.1, 100, 4)
  np.testing.assert_allclose(embedding.freqs[...], expected, rtol=1e-5)
  assert isinstance(embedding.freqs, nnx.Param) == learn_freqs
  t = jnp.array([[[0.2], [0.7]], [[0.4], [0.9]]])
  assert embedding(t).shape == (2, 2, 5)
  derivative = jax.grad(lambda tau: embedding(tau).sum())(t)
  assert finite(derivative) and jnp.any(derivative != 0)
  grads = nnx.grad(lambda e: e(t).sum())(embedding)
  assert finite(grads.phase[...]) and jnp.any(grads.phase[...] != 0)
  if learn_freqs:
    assert finite(grads.freqs[...]) and jnp.any(grads.freqs[...] != 0)
  else:
    assert "freqs" not in grads


@pytest.mark.parametrize("config", [
  SinusoidalTimeConfig(time_dim=0), SinusoidalTimeConfig(time_dim=1),
  CosTimeConfig(time_dim=4), FourierTimeConfig(time_dim=0),
  FourierTimeConfig(num_freqs=0), FourierTimeConfig(hidden=0),
  FourierTimeConfig(spacing="invalid"), FourierTimeConfig(freq_min=0),
  FourierTimeConfig(freq_min=101), FourierTimeConfig(freq_max=math.inf),
])
def test_invalid_time_encoder_configurations_fail_before_network_build(config):
  assert config.problems()
  with pytest.raises(ValueError):
    config.build(rngs=nnx.Rngs(0))


def test_cos_encoder_keeps_its_declared_single_channel():
  embedding = CosTimeConfig().build(rngs=nnx.Rngs(0))
  t = jnp.array([[0.0], [math.pi / 2], [math.pi]])
  assert embedding(t).shape == t.shape
  np.testing.assert_allclose(embedding(t).ravel(), [1.0, 0.0, -1.0], atol=1e-6)


class _IdentityTime(nnx.Module):
  def __call__(self, t):
    return t


class _EchoCondition(nnx.Module):
  def __call__(self, inputs, conditioning):
    return jnp.broadcast_to(conditioning[-1], inputs[0].shape)


@pytest.mark.parametrize("name", ["ScoreContinuousActor", "DISActor", "DDPMActor"])
def test_time_coordinates_and_broadcasting_reach_the_diffusion_field(name):
  actor = build(name)
  actor.field.time_emb = _IdentityTime()
  actor.field.net = _EchoCondition()
  obs, x = jnp.zeros((BATCH, OBS_DIM)), jnp.zeros((BATCH, ACT_DIM))
  if name == "ScoreContinuousActor":
    np.testing.assert_array_equal(actor.control(obs, x, 0.25), -0.5)
    per_row = jnp.linspace(0, 1, BATCH)
    np.testing.assert_allclose(actor.control(obs, x, per_row),
                               jnp.broadcast_to((2 * per_row - 1)[:, None], x.shape))
  elif name == "DISActor":
    np.testing.assert_array_equal(actor.control_at(obs, x, 3), 3)
    run = nnx.jit(lambda a, step: a.control_at(obs, x, step))
    np.testing.assert_array_equal(run(actor, jnp.asarray(5)), 5)
  else:
    np.testing.assert_array_equal(actor.predict_eps(obs, x, 0.25), 0.25)
    per_row = jnp.linspace(actor.config.t_min, actor.config.t_max, BATCH)
    np.testing.assert_allclose(actor.predict_eps(obs, x, per_row),
                               jnp.broadcast_to(per_row[:, None], x.shape))


@pytest.mark.parametrize("sigma_min", [0.01, 1e-6])
def test_amdp_step_variances_match_the_integral_and_keep_small_late_steps(sigma_min):
  actor = build("ScoreContinuousActor", sigma_min=sigma_min, max_steps=64,
                tau_min=0.2, tau_max=0.9)
  times = np.asarray(actor.taus[...], np.float64)
  log_r = math.log(actor.config.sigma_max / sigma_min)
  expected = actor.config.sigma_max ** 2 * (
    np.exp(-2 * log_r * times[:-1]) - np.exp(-2 * log_r * times[1:]))
  assert bool(jnp.all(actor.step_var[...] > 0))
  np.testing.assert_allclose(actor.step_var[...], expected, rtol=1e-5)
  np.testing.assert_allclose(actor.step_var[...].sum(), actor.terminal_var, rtol=1e-6)
  t = jnp.linspace(0.3, 0.8, 7)
  derivative = jax.vmap(jax.grad(actor.sigma_sq_int_fn))(t)
  np.testing.assert_allclose(derivative, actor.sigma_fn(t) ** 2, rtol=2e-6)
  endpoint = jnp.ones((BATCH, ACT_DIM))
  noise = jnp.full_like(endpoint, 3)
  np.testing.assert_array_equal(actor.bridge_sample(endpoint, actor.config.tau_min, noise), 0)
  np.testing.assert_array_equal(actor.bridge_sample(endpoint, actor.config.tau_max, noise), endpoint)


def test_dime_step_schedule_is_positive_decreasing_and_starts_at_configured_dt():
  actor = build("DISActor", num_steps=7, dt_init=0.2, friction_init=2.0)
  coefficients = [actor.step_coef(i) for i in range(actor.config.num_steps)]
  eta = jnp.stack([item[0] for item in coefficients])
  scale = jnp.stack([item[1] for item in coefficients])
  assert bool(jnp.all(eta > 0)) and bool(jnp.all(jnp.diff(eta, axis=0) < 0))
  np.testing.assert_allclose(scale ** 2, 2 * eta, rtol=1e-6)
  assert float(eta[0, 0]) == pytest.approx(0.1, rel=2e-4)


@pytest.mark.parametrize("t_min, t_max", [(1e-3, 0.8), (1e-7, 0.8), (1e-7, 4e-7)])
def test_sdac_schedule_and_reverse_steps_match_high_precision_reference(t_min, t_max):
  actor = build("DDPMActor", num_steps=3, t_min=t_min, t_max=t_max, clip_sampler=False)
  actor.field.time_emb = _IdentityTime()
  actor.field.net = _EchoCondition()
  times = np.asarray(actor.t_grid[...], np.float64)
  assert times[0] == pytest.approx(t_min) and times[-1] == pytest.approx(t_max)
  assert np.all(np.diff(times) > 0)
  s = actor.config.cosine_s
  alpha = np.cos(np.pi / 2 * (times + s) / (1 + s)) / math.cos(np.pi / 2 * s / (1 + s))
  actual_alpha, actual_sigma = actor.alpha_sigma(jnp.asarray(times, jnp.float32))
  np.testing.assert_allclose(actual_alpha, alpha, rtol=1e-6)
  np.testing.assert_allclose(actual_sigma, np.sqrt(1 - alpha ** 2), rtol=1e-5)
  assert bool(jnp.all(actual_sigma > 0))
  x = jnp.full((BATCH, ACT_DIM), 0.3)
  obs = jnp.zeros((BATCH, OBS_DIM))
  key = jax.random.key(9)
  for n in (1, 3):
    ab_n, ab_prev = alpha[n] ** 2, alpha[n - 1] ** 2
    a = ab_n / ab_prev
    beta = 1 - a
    x0 = (np.asarray(x) - np.sqrt(1 - ab_n) * times[n]) / alpha[n]
    expected = (np.sqrt(ab_prev) * beta * x0 + np.sqrt(a) * (1 - ab_prev) * np.asarray(x)) / (1 - ab_n)
    if n > 1:
      expected += np.sqrt(beta) * np.asarray(jax.random.normal(key, x.shape))
    np.testing.assert_allclose(actor.reverse_step(obs, x, n, key), expected, rtol=1e-5, atol=1e-6)
    compiled = nnx.jit(lambda a, step: a.reverse_step(obs, x, step, key))
    np.testing.assert_allclose(compiled(actor, jnp.asarray(n)), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("model, values", [
  ("AMDP", {"max_steps": 0}), ("AMDP", {"tau_min": 0.8, "tau_max": 0.2}),
  ("AMDP", {"sigma_min": 0}), ("AMDP", {"sigma_max": 0.001}),
  ("DIME", {"num_steps": 0}), ("DIME", {"dt_init": 0}),
  ("DIME", {"dt_s": -1}), ("DIME", {"friction_init": 0}),
  ("SDAC", {"num_steps": 0}), ("SDAC", {"t_min": 0}),
  ("SDAC", {"t_max": 1}), ("SDAC", {"t_min": 0.8, "t_max": 0.2}),
])
def test_invalid_diffusion_time_configuration_is_reported_before_training(model, values):
  with pytest.raises(ValueError):
    RunConfig.from_dict({"model": model, "env": "HalfCheetah",
      "model_overrides": {"components": {"actor": values}}})
  actor_config = RunConfig.defaults(model, "HalfCheetah").model_config.components.actor
  with pytest.raises(ValueError):
    build(actor_config.class_name, **values)


def test_chain_helpers_visit_every_time_index_in_the_correct_order():
  def reverse_step(x, n, key):
    return x * 10 + n

  x = reverse_chain(reverse_step, jnp.zeros((1, 1)), 4, jax.random.key(0))
  np.testing.assert_array_equal(x, 4321)

  def step(x, i, key):
    return x * 10 + i, jnp.ones((1, 1))

  state, aux = controlled_sde(step, jnp.zeros((1, 1)), 4, jax.random.key(0))
  np.testing.assert_array_equal(state, 123)
  np.testing.assert_array_equal(aux, 4)


def test_chain_helpers_give_each_step_its_own_split_key():
  key = jax.random.key(3)
  step = lambda x, i, k: (x + jax.random.normal(k, x.shape), jnp.zeros_like(x))
  state, _ = controlled_sde(step, jnp.zeros((2, 3)), 5, key)
  expected = sum(jax.random.normal(k, (2, 3)) for k in jax.random.split(key, 5))
  np.testing.assert_allclose(state, expected, rtol=1e-5, atol=1e-6)
