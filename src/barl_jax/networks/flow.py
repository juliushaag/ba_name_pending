import math

import jax.numpy as jnp


def sinusoidal_emb(t, time_dim: int):
  shape = t.shape[:-1]
  t = t.reshape(-1)
  half = time_dim // 2
  freqs = jnp.exp(-math.log(10000) * jnp.arange(half, dtype=jnp.float32) / (half - 1))
  args = t[:, None] * 1000.0 * freqs[None]
  emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
  return emb.reshape(*shape, time_dim)
