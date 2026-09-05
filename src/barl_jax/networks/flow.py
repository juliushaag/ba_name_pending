import math

import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.components import Buffer

# ---------------------------------------------------------------------------
# TIME_STYLES -- the named time/level-embedding surfaces a component *selects*
# from (`time_emb = Config("fourier", style=TIME_STYLES)`).
# ---------------------------------------------------------------------------

_TIME_SIN_16 = dict(time_dim=8, time_embed="sin",
                    time_kwargs={"num_freqs": 16, "freq_min": 0.1, "freq_max": 100.0,
                                 "hidden": 32, "concat_input": True, "spacing": "geometric"})

TIME_STYLES = {
  "sin_16": _TIME_SIN_16,
  "sin_20": {**_TIME_SIN_16,
             "time_kwargs": {**_TIME_SIN_16["time_kwargs"], "num_freqs": 20}},
  # dime_am's FeatureProjector -- 16 geometric frequencies and a trainable
  # zero-init phase, Dense -> gelu -> Dense out to time_dim 16. The NC-LQL
  # level coder.
  "fourier": {**_TIME_SIN_16, "time_dim": 16, "time_embed": "fourier"},
  # The AM control net's coder (ctrl_net.py FeatureProjector at
  # configs/alg/am.yaml).
  "ctrl":    dict(time_dim=16, time_embed="fourier",
                  time_kwargs={"num_freqs": 16, "freq_min": 0.1, "freq_max": 100.0,
                               "hidden": 32, "concat_input": True, "spacing": "geometric"}),
  # PISGRADNet's coder -- 256 *linearly* spaced frequencies, a trainable
  # zero-init phase, sin/cos with no raw `t`, out to time_dim 256.
  "pisgrad": dict(time_dim=256, time_embed="fourier",
                  time_kwargs={"num_freqs": 256, "freq_min": 0.1, "freq_max": 100.0,
                               "hidden": 256, "concat_input": False, "spacing": "linear"}),
}


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
  `hidden 32`, `time_dim 16`, `concat_input` on, geometric spacing; PISGRADNet's
  (see `spacing`) are 256 linearly spaced frequencies into a 256-wide coder with
  no raw `t` concatenated."""

  def __init__(self, time_dim: int, *, kind: str, num_freqs: int,
               freq_min: float, freq_max: float, hidden: int,
               concat_input: bool, spacing: str, rngs: nnx.Rngs):

    from barl_jax.networks.mlp import Linear  # circular at module import time

    assert kind in ("sin", "fourier", "learned"), \
      f"{kind} not in available time embeddings ('sin', 'fourier', 'learned')"

    self.kind         = kind
    self.time_dim     = int(time_dim)
    self.concat_input = bool(concat_input)

    if kind == "sin":
      return

    assert spacing in ("geometric", "linear"), \
      f"{spacing} not in available frequency spacings ('geometric', 'linear')"

    # dime_am's two time coders disagree here and the difference is not
    # cosmetic: ctrl_net.FeatureProjector puts its frequencies on a geometric
    # grid, PISGRADNet's `timestep_coeff` on a linear one (jnp.linspace(0.1,
    # 100, num_hid)), so a linear grid spends most of its ladder on the high
    # frequencies a 16-step index needs.
    space = jnp.geomspace if spacing == "geometric" else jnp.linspace
    freqs = space(freq_min, freq_max, num_freqs, dtype=jnp.float32)
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
