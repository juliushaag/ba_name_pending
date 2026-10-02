import math
from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.components import Buffer
from barl_jax.config.serialization import variant_registry
from barl_jax.networks.base import Linear

TIME_CONFIGS, register_time = variant_registry()


@dataclass(frozen=True)
class TimeEmbeddingConfig:
  """Selects and builds the coder from a scalar time/level `(B, 1)` to
  `(B, time_dim)`; `class_name` names the registered subclass."""
  variants: ClassVar[dict[str, type]] = TIME_CONFIGS

  class_name: str = ""
  time_dim: int = 8

  def problems(self) -> list[str]:
    return [] if self.time_dim >= 1 else ["time_dim must be >= 1"]

  def build(self, *, rngs: nnx.Rngs) -> nnx.Module:
    raise NotImplementedError


@register_time
@dataclass(frozen=True)
class SinusoidalTimeConfig(TimeEmbeddingConfig):
  """Fixed transformer-style frequencies, no parameters."""
  class_name: str = "Sinusoidal"

  def problems(self):
    return [] if self.time_dim >= 2 else ["Sinusoidal time_dim must be >= 2"]

  def build(self, *, rngs):
    return SinusoidalEmbedding(self.time_dim)


@register_time
@dataclass(frozen=True)
class FourierTimeConfig(TimeEmbeddingConfig):
  """dime_am's `FeatureProjector` (diffusion_AM/ctrl_net.py): `num_freqs`
  frequencies on a `spacing` grid over [freq_min, freq_max], a trainable zero-init
  phase, sin/cos of `t*freq + phase`, optionally with the raw `t`, then
  `Dense -> gelu -> Dense`. `learn_freqs` makes the frequencies trainable too.

  The two dime_am coders differ in `spacing`: ctrl_net's frequencies are geometric,
  PISGRADNet's linear (256 of them, `concat_input` off), which spends most of its
  ladder on the high frequencies."""
  class_name: str = "Fourier"
  time_dim: int = 16
  num_freqs: int = 16
  freq_min: float = 0.1
  freq_max: float = 100.0
  hidden: int = 32
  concat_input: bool = True
  spacing: str = "geometric"
  learn_freqs: bool = False

  def problems(self):
    problems = super().problems()
    if self.num_freqs < 1 or self.hidden < 1:
      problems.append("num_freqs and hidden must be >= 1")
    if self.spacing not in ("geometric", "linear"):
      problems.append("spacing must be 'geometric' or 'linear'")
    if not (math.isfinite(self.freq_min) and math.isfinite(self.freq_max)
            and self.freq_min <= self.freq_max):
      problems.append("frequency bounds must be finite with freq_min <= freq_max")
    if self.spacing == "geometric" and self.freq_min <= 0:
      problems.append("geometric frequencies require freq_min > 0")
    return problems

  def build(self, *, rngs):
    return FourierEmbedding(self, rngs=rngs)


@register_time
@dataclass(frozen=True)
class CosTimeConfig(TimeEmbeddingConfig):
  """Scalar `cos(t)`, the floq reference's time feature."""
  class_name: str = "Cos"
  time_dim: int = 1

  def problems(self):
    return [] if self.time_dim == 1 else ["Cos time_dim must be 1"]

  def build(self, *, rngs):
    _validate_config(self)
    return CosEmbedding()


def _validate_config(config):
  problems = config.problems()
  if problems:
    raise ValueError(f"{config.class_name} time embedding: " + "; ".join(problems))


class CosEmbedding(nnx.Module):
  time_dim = 1

  def __call__(self, t):
    return jnp.cos(t)


def sinusoidal_emb(t, time_dim: int):
  if time_dim < 2:
    raise ValueError("Sinusoidal time_dim must be >= 2")
  shape = t.shape[:-1]
  t = t.reshape(-1)
  half = time_dim // 2
  freqs = jnp.exp(-math.log(10000) * jnp.arange(half, dtype=jnp.float32) / max(half - 1, 1))
  args = t[:, None] * 1000.0 * freqs[None]
  emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
  if time_dim % 2:
    emb = jnp.pad(emb, ((0, 0), (0, 1)))
  return emb.reshape(*shape, time_dim)


class SinusoidalEmbedding(nnx.Module):

  def __init__(self, time_dim: int):
    _validate_config(SinusoidalTimeConfig(time_dim=time_dim))
    self.time_dim = int(time_dim)

  def __call__(self, t):
    return sinusoidal_emb(t, self.time_dim)


class FourierEmbedding(nnx.Module):

  def __init__(self, config: FourierTimeConfig, *, rngs: nnx.Rngs):
    _validate_config(config)
    self.time_dim = int(config.time_dim)
    self.concat_input = bool(config.concat_input)

    space = jnp.geomspace if config.spacing == "geometric" else jnp.linspace
    freqs = space(config.freq_min, config.freq_max, config.num_freqs, dtype=jnp.float32)
    self.freqs = nnx.Param(freqs) if config.learn_freqs else Buffer(freqs)
    self.phase = nnx.Param(jnp.zeros((config.num_freqs,), jnp.float32))

    in_dim = 2 * config.num_freqs + (1 if config.concat_input else 0)
    self.dense1 = Linear(in_dim, config.hidden, rngs=rngs, activation="gelu")
    self.dense2 = Linear(config.hidden, self.time_dim, rngs=rngs)

  def __call__(self, t):
    shape = t.shape[:-1]
    flat = t.reshape(-1, 1)

    angle = flat * self.freqs[...] + self.phase[...]
    feats = jnp.concatenate([jnp.sin(angle), jnp.cos(angle)], axis=-1)
    if self.concat_input:
      feats = jnp.concatenate([flat, feats], axis=-1)

    return self.dense2(jax.nn.gelu(self.dense1(feats))).reshape(*shape, self.time_dim)
