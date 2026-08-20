import math

import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.policy import Buffer


def sinusoidal_emb(t, time_dim: int):
  shape = t.shape[:-1]
  t = t.reshape(-1)
  half = time_dim // 2
  freqs = jnp.exp(-math.log(10000) * jnp.arange(half, dtype=jnp.float32) / (half - 1))
  args = t[:, None] * 1000.0 * freqs[None]
  emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
  return emb.reshape(*shape, time_dim)


class TimeEmbedding(nnx.Module):
  """A scalar time/level `(B, 1)` -> `(B, time_dim)`, in one of three kinds.

  * `sin` -- `sinusoidal_emb`: fixed transformer-style frequencies, no
    parameters. barl's original, and the default so nothing moves unasked.
  * `fourier` -- dime_am's `FeatureProjector` (diffusion_AM/ctrl_net.py):
    `num_freqs` frequencies fixed on a geometric grid over `freq_range`, a
    *trainable* zero-init phase, sin/cos of `t*freq + phase`, optionally
    concatenated with the raw `t`, then `Dense -> gelu -> Dense`.
  * `learned` -- the same, with the frequencies themselves trainable. Only the
    two differ: which of `freqs`/`phase` carries gradient.

  The reference's time settings are `num_freqs 16`, `freq_range (0.1, 100.0)`,
  `hidden 32`, `time_dim 16`, `concat_input` on."""

  def __init__(self, time_dim: int, kind: str = "sin", num_freqs: int = 16,
               freq_min: float = 0.1, freq_max: float = 100.0, hidden: int = 32,
               concat_input: bool = True, *, rngs: nnx.Rngs):
               
    from barl_jax.networks.mlp import Linear   # circular at module import time

    assert kind in ("sin", "fourier", "learned"), \
      f"{kind} not in available time embeddings ('sin', 'fourier', 'learned')"

    self.kind         = kind
    self.time_dim     = int(time_dim)
    self.concat_input = bool(concat_input)

    if kind == "sin":
      return

    freqs = jnp.geomspace(freq_min, freq_max, num_freqs, dtype=jnp.float32)
    self.freqs = nnx.Param(freqs) if kind == "learned" else Buffer(freqs)
    self.phase = nnx.Param(jnp.zeros((num_freqs,), jnp.float32))

    in_dim = 2 * num_freqs + (1 if concat_input else 0)
    self.dense1 = Linear(in_dim, hidden, rngs=rngs)
    self.dense2 = Linear(hidden, self.time_dim, rngs=rngs)

  def __call__(self, t):
    if self.kind == "sin":
      return sinusoidal_emb(t, self.time_dim)

    shape = t.shape[:-1]
    flat  = t.reshape(-1, 1)

    angle = flat * self.freqs[...] + self.phase[...]
    feats = jnp.concatenate([jnp.sin(angle), jnp.cos(angle)], axis=-1)
    if self.concat_input:
      feats = jnp.concatenate([flat, feats], axis=-1)

    return self.dense2(jax.nn.gelu(self.dense1(feats))).reshape(*shape, self.time_dim)
