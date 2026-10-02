"""Langevin sampling on a known action-value landscape."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import build, make_env_stats
from flax import nnx

from barl_jax.components import Buffer
from barl_jax.metrics import DEBUG, collect, recording


class _QuadraticCritic(nnx.Module):
  """An interior optimum: Q(a) = -||a||² / 2, at every noise level."""

  def __init__(self):
    self.noise = nnx.Module()
    self.noise.sigmas = Buffer(jnp.geomspace(0.1, 0.001, 8))

  def expectation(self, obs, action, *, level):
    return -0.5 * jnp.square(action).sum(axis=-1, keepdims=True)


def _sample(dim, **config):
  stats = make_env_stats(act_dim=dim, act_low=np.full(dim, -1., np.float32),
                         act_high=np.full(dim, 1., np.float32))
  actor = build("LangevinActor", env_stats=stats, **config)
  critic = _QuadraticCritic()
  obs = jnp.zeros((256, actor.obs_dim), jnp.float32)
  key = jax.random.PRNGKey(2)
  with recording(DEBUG):
    action, entropy = actor.generate_action(obs, critic=critic, key=key)
  diagnostics = collect(nnx.pop(actor, nnx.Intermediate))
  assert entropy is None
  assert action.shape == (256, 1, dim)
  assert bool(jnp.all(jnp.isfinite(action)))
  assert bool(jnp.all(jnp.abs(action) <= 1.))
  return action, diagnostics


@pytest.mark.parametrize("dim", [1, 6, 38])
def test_coarse_step_cap_avoids_saturation_at_an_interior_optimum(dim):
  action, capped = _sample(dim)
  _, reference = _sample(dim, max_step_size=None)
  assert float(reference["chain/clipped_frac_l0"]) > 0.95
  assert float(capped["chain/clipped_frac_l0"]) < 0.02
  assert float((jnp.abs(action) >= 1. - 1.e-4).mean()) < 0.02
  # A finite chain should also move toward this critic's known optimum.
  assert float(jnp.square(action).mean()) < 1. / 3.


def test_step_cap_scales_drift_and_noise_together_and_preserves_fine_steps():
  _, values = _sample(6)
  for i in range(8):
    alpha = float(values[f"chain/step_size_l{i}"])
    assert alpha <= 1.e-3 * (1. + 1.e-6)
    assert float(values[f"chain/noise_scale_l{i}"]) == pytest.approx(np.sqrt(alpha))
    assert float(values[f"chain/drift_norm_l{i}"]) == pytest.approx(
      0.5 * alpha * 500., rel=1.e-4)
  assert float(values["chain/step_size_l0"]) == pytest.approx(1.e-3)
  assert float(values["chain/step_size_l7"]) == pytest.approx(1.e-4)


def test_disabled_step_cap_restores_reference_schedule():
  _, values = _sample(6, max_step_size=None)
  assert float(values["chain/step_size_l0"]) == pytest.approx(1.)
  assert float(values["chain/drift_norm_l0"]) == pytest.approx(250.)
  assert float(values["chain/noise_scale_l0"]) == pytest.approx(1.)


@pytest.mark.parametrize("limit", [0., -1., float("inf"), float("nan")])
def test_invalid_step_cap_is_rejected(limit):
  with pytest.raises(ValueError, match="max_step_size"):
    build("LangevinActor", max_step_size=limit)
