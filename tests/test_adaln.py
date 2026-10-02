"""AdaLN algebra, zero initialization, and diffusion conditioning paths."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from conftest import ACT_DIM, BATCH, OBS_DIM, build, finite
from flax import nnx

from barl_jax.config.run import RunConfig
from barl_jax.config.serialization import construct, to_dict
from barl_jax.networks.norm import AdaLN
from barl_jax.networks.simba import AdaLNSimBa, AdaLNSimbaConfig


@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16, jnp.float16])
def test_adaln_matches_affine_free_layernorm_and_modulation(gated, dtype):
  epsilon = 1e-4
  norm = AdaLN(4, 3, gated=gated, epsilon=epsilon, dtype=dtype, rngs=nnx.Rngs(0))
  x = jnp.array([[1.0, 3.0, -2.0, 0.5], [-1.0, 0.0, 4.0, 2.0]], dtype=dtype)
  c = jnp.array([[0.3, -0.7, 0.2], [-0.5, 0.8, 1.0]], dtype=dtype)
  xf = np.asarray(x, dtype=np.float32)
  expected_norm = (xf - xf.mean(-1, keepdims=True)) / np.sqrt(
    xf.var(-1, keepdims=True) + epsilon)
  initial = norm(x, c)
  h, gate = initial if gated else (initial, None)
  tolerance = 1e-6 if dtype == jnp.float32 else 2e-2
  np.testing.assert_allclose(h.astype(jnp.float32), expected_norm,
                             atol=tolerance, rtol=tolerance)
  assert not jax.tree.leaves(nnx.state(norm.norm, nnx.Param))
  assert all(p.dtype == jnp.float32 for p in jax.tree.leaves(nnx.state(norm, nnx.Param)))
  assert h.dtype == dtype
  if gated:
    assert jnp.array_equal(gate, jnp.zeros_like(gate))

  width = (3 if gated else 2) * 4
  norm.modulation.kernel[...] = jnp.arange(3 * width, dtype=jnp.float32).reshape(3, width) / 100
  norm.modulation.bias[...] = jnp.linspace(-0.2, 0.3, width)
  modulation = np.asarray(norm.modulation(c), dtype=np.float32)
  shift, scale, *expected_gate = np.split(modulation, 3 if gated else 2, axis=-1)
  output = norm(x, c)
  h, gate = output if gated else (output, None)
  np.testing.assert_allclose(h.astype(jnp.float32), expected_norm * (1 + scale) + shift,
                             atol=tolerance, rtol=tolerance)
  if gated:
    np.testing.assert_array_equal(gate.astype(jnp.float32), expected_gate[0])


def test_zero_gate_learns_before_the_residual_branch():
  net = AdaLNSimbaConfig().build(5, 3, 8, 2, cond_dim=4, rngs=nnx.Rngs(0))
  block = net.blocks[0]
  x = jax.random.normal(jax.random.key(1), (BATCH, 8))
  c = jax.random.normal(jax.random.key(2), (BATCH, 8))
  grads = nnx.grad(lambda b: jnp.square(b(x, c)).sum())(block)
  assert jnp.array_equal(block(x, c), x)
  assert jnp.all(grads.up.kernel[...] == 0)
  assert jnp.all(grads.down.kernel[...] == 0)
  assert jnp.all(grads.norm.modulation.kernel[..., :16] == 0)
  assert jnp.any(grads.norm.modulation.kernel[..., 16:] != 0)


@pytest.mark.parametrize("name", ["ScoreContinuousActor", "DISActor", "DDPMActor"])
def test_diffusion_adaln_learns_observation_and_time_conditioning(name):
  actor = build(name)
  field = actor.field
  assert isinstance(field.net, AdaLNSimBa)
  assert field.net.stem.kernel.shape[0] == ACT_DIM
  assert field.net.cond_embed.kernel.shape[0] == OBS_DIM + actor.config.time.time_dim
  obs = jax.random.normal(jax.random.key(3), (BATCH, OBS_DIM))
  x = jax.random.normal(jax.random.key(4), (BATCH, ACT_DIM))
  t = jnp.full((BATCH, 1), 0.2)
  other_t = jnp.full_like(t, 0.7)
  target = obs[:, :ACT_DIM] + t

  # AMDP and DIME also zero the head, delaying modulation gradients by one
  # update. Exercise real updates rather than manually opening their gates.
  optimizer = nnx.Optimizer(field, optax.adam(1e-2), wrt=nnx.Param)
  for _ in range(4):
    loss, grads = nnx.value_and_grad(lambda f: jnp.square(f(obs, x, t) - target).mean())(field)
    assert finite(loss) and all(finite(g) for g in jax.tree.leaves(grads))
    optimizer.update(field, grads)
  eager = field(obs, x, t)
  assert not jnp.allclose(eager, field(-obs, x, t))
  assert not jnp.allclose(eager, field(obs, x, other_t))
  assert jnp.any(field.net.blocks[0].norm.modulation.kernel[...] != 0)
  assert jnp.any(field.net.final_norm.modulation.kernel[...] != 0)
  np.testing.assert_allclose(nnx.jit(lambda f, s, a, tau: f(s, a, tau))(field, obs, x, t),
                             eager, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("model", ["AMDP", "DIME", "SDAC", "SAC", "TD3", "NCLQL"])
def test_only_diffusion_models_default_to_adaln_and_configs_roundtrip(model):
  config = RunConfig.from_dict({"model": model, "env": "HalfCheetah"}).model_config
  actor = config.components.actor
  if hasattr(actor, "mlp"):
    assert isinstance(actor.mlp, AdaLNSimbaConfig) == (model in ("AMDP", "DIME", "SDAC"))
    assert construct(type(actor), to_dict(actor)) == actor
  assert not isinstance(config.components.critic.mlp, AdaLNSimbaConfig)


def test_adaln_requires_conditioning_and_layer_normalization():
  with pytest.raises(ValueError, match="conditioning input"):
    AdaLNSimbaConfig().build(5, 3, 8, 2, rngs=nnx.Rngs(0))
  net = AdaLNSimbaConfig().build(5, 3, 8, 2, cond_dim=4, rngs=nnx.Rngs(0))
  with pytest.raises(ValueError, match="conditioning input"):
    net((jnp.ones((BATCH, 5)),))
  for norm in ("none", "batch", "batchrenorm"):
    config = AdaLNSimbaConfig(norm=norm)
    assert config.problems()
    with pytest.raises(ValueError, match="requires norm='layer'"):
      config.build(5, 3, 8, 2, cond_dim=4, rngs=nnx.Rngs(0))
