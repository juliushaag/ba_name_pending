import jax
from flax import nnx

from barl_jax.components import Buffer
from barl_jax.networks.schedule import Schedule
from barl_jax.registration import get_component


class NoiseConditioning(nnx.Module):
  """NCLQL's sigma ladder, shared with the Langevin sampler. The last level
  is the clean one; noisy training levels are drawn from the ones before it."""

  def __init__(self, env_stats, config):
    schedule_cls = get_component(config.schedule.class_name)
    if not issubclass(schedule_cls, Schedule):
      raise TypeError("noise.schedule must name a Schedule")
    self.sigmas = Buffer(schedule_cls(env_stats, config.schedule).ladder())
    if self.sigmas.shape[0] < 2:
      raise ValueError("noise-conditioned TD requires at least two noise levels")

  @property
  def clean_level(self) -> int:
    return self.sigmas.shape[0] - 1

  def perturb(self, action, key):
    level_key, noise_key = jax.random.split(key)
    levels = jax.random.randint(level_key, (action.shape[0],), 0, self.clean_level)
    noise = jax.random.normal(noise_key, action.shape, action.dtype)
    return action + self.sigmas[...][levels][:, None] * noise, levels
