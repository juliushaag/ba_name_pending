"""Activation-aware initialization, small heads, and algorithm exceptions."""

import math
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest
from conftest import build
from flax import nnx

from barl_jax.config.serialization import construct, to_dict
from barl_jax.networks.base import ACTIVATIONS, Linear
from barl_jax.networks.mlp import MLPConfig
from barl_jax.networks.simba import AdaLNSimbaConfig, SimbaConfig
from barl_jax.networks.time_embedding import FourierTimeConfig


def expected_variance(activation, inputs, outputs):
  if activation is None:
    return 1.0 / inputs
  if activation == "tanh":
    return (5.0 / 3.0) ** 2 / ((inputs + outputs) / 2)
  if activation in ("relu", "relu6"):
    return 2.0 / inputs
  if activation == "leakyrelu":
    return 2.0 / ((1.0 + 0.01 ** 2) * inputs)
  # Independently integrate the smooth activation's second moment, rather
  # than reuse the initializer's gain table.
  points, weights = np.polynomial.hermite.hermgauss(128)
  values = np.asarray(ACTIVATIONS[activation](jnp.asarray(math.sqrt(2) * points)))
  moment = np.dot(weights, values.astype(np.float64) ** 2) / math.sqrt(math.pi)
  return 1.0 / (moment * inputs)


def assert_variance(linear, activation=None, gain=1.0):
  kernel = np.asarray(linear.kernel[...])
  expected = gain ** 2 * expected_variance(activation, *kernel.shape)
  assert kernel.var() == pytest.approx(expected, rel=0.04)


@pytest.mark.parametrize("activation", [None, *ACTIVATIONS])
def test_linear_initialization_matches_activation_and_keeps_float32_parameters(activation):
  linear = Linear(256, 512, activation=activation, dtype=jnp.bfloat16, rngs=nnx.Rngs(0))
  assert_variance(linear, activation)
  assert jnp.all(linear.bias[...] == 0)
  assert linear.kernel[...].dtype == linear.bias[...].dtype == jnp.float32


CONFIGS = [MLPConfig(), MLPConfig(residual=True), SimbaConfig(), AdaLNSimbaConfig()]


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("activation", ["relu", "tanh", "silu"])
def test_network_initializes_activated_layers_and_bare_projections_differently(config, activation):
  config = replace(config, activation=activation)
  net = config.build(256, 64, 256, 1, cond_dim=128, rngs=nnx.Rngs(0))
  if isinstance(config, MLPConfig):
    assert_variance(net.stem, None if config.residual else activation)
    assert_variance(net.blocks[0].lin, activation)
  else:
    assert_variance(net.stem)
    assert_variance(net.blocks[0].up, activation)
    assert_variance(net.blocks[0].down)
  if isinstance(config, AdaLNSimbaConfig):
    # This embedding always uses SiLU, regardless of the block activation.
    assert_variance(net.cond_embed, "silu")
  assert_variance(net.head, gain=0.01)


@pytest.mark.parametrize("config", [*CONFIGS, MLPConfig(final_activation=True, norm="none")])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "float16"])
def test_head_gain_scales_only_the_head_and_roundtrips(config, dtype):
  config = replace(config, dtype=dtype)
  full_config = replace(config, head_gain=1.0)
  small = config.build(5, 3, 16, 2, cond_dim=4, rngs=nnx.Rngs(0))
  full = full_config.build(5, 3, 16, 2, cond_dim=4, rngs=nnx.Rngs(0))
  np.testing.assert_allclose(small.head.kernel[...], 0.01 * full.head.kernel[...], rtol=1e-6)
  for (path, param), (other_path, other_param) in zip(
      nnx.to_flat_state(nnx.state(small, nnx.Param)),
      nnx.to_flat_state(nnx.state(full, nnx.Param))):
    assert path == other_path
    if path != ("head", "kernel"):
      np.testing.assert_array_equal(param[...], other_param[...])
    if path[-1] == "bias":
      assert jnp.all(param[...] == 0)
  assert construct(type(config), to_dict(config)) == config


def test_fourier_embedding_uses_gelu_hidden_gain_and_unscaled_linear_output():
  embedding = FourierTimeConfig(num_freqs=128, hidden=256, time_dim=128).build(rngs=nnx.Rngs(0))
  assert_variance(embedding.dense1, "gelu")
  assert_variance(embedding.dense2)


@pytest.mark.parametrize("gain", [0.0, -0.1, float("inf"), float("-inf"), float("nan")])
def test_invalid_head_gains_are_rejected(gain):
  config = MLPConfig(head_gain=gain)
  assert any("head_gain" in problem for problem in config.problems())
  with pytest.raises(ValueError, match="gain"):
    config.build(5, 3, 8, 1, rngs=nnx.Rngs(0))


@pytest.mark.parametrize("actor_name", ["ScoreContinuousActor", "DISActor"])
def test_controlled_diffusion_keeps_exactly_zero_heads(actor_name):
  actor = build(actor_name)
  assert jnp.all(actor.field.net.head.kernel[...] == 0)
  assert jnp.all(actor.field.net.head.bias[...] == 0)
