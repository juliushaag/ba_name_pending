from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.networks.base import (
  Linear,
  NetworkConfig,
  concat,
  make_dropout,
  register_network,
)
from barl_jax.networks.norm import AdaLN


@register_network
@dataclass(frozen=True)
class SimbaConfig(NetworkConfig):
  """SimBa (Lee et al., ICLR 2025): a linear embedding, `layers` pre-norm residual
  blocks whose hidden layer is `expansion` times wider, and a final norm. The
  observation running-statistics normalizer is the policy's, not this net's."""
  class_name: str = "SimBa"
  activation: str = "relu"
  expansion: int = 4

  def build(self, input_dim, output_dim, latent_dim, layers, *, cond_dim=0, rngs):
    return SimBa(input_dim + cond_dim, output_dim, latent_dim, layers, config=self, rngs=rngs)


@register_network
@dataclass(frozen=True)
class AdaLNSimbaConfig(SimbaConfig):
  """SimBa whose block norms are adaptive: `conditioning` is embedded once and every
  block takes a (shift, scale, gate) from it, as in DiT. Only `inputs` enter the stem."""
  class_name: str = "AdaLNSimBa"

  def problems(self):
    problems = super().problems()
    if self.norm != "layer":
      problems.append("AdaLNSimBa requires norm='layer': its normalization is adaptive LayerNorm")
    return problems

  def build(self, input_dim, output_dim, latent_dim, layers, *, cond_dim=0, rngs):
    problems = self.problems()
    if problems:
      raise ValueError("AdaLNSimBa: " + "; ".join(problems))
    return AdaLNSimBa(input_dim, cond_dim, output_dim, latent_dim, layers, config=self, rngs=rngs)


class SimbaBlock(nnx.Module):
  """x + Linear(act(Linear(Norm(x))))."""

  def __init__(self, latent_dim: int, act, norm, *, expansion: int, dropout: float,
               activation: str, rngs: nnx.Rngs, dtype=jnp.float32):
    self.act, self.norm = act, norm
    self.up = Linear(latent_dim, expansion * latent_dim, rngs=rngs, dtype=dtype,
                     activation=activation)
    self.down = Linear(expansion * latent_dim, latent_dim, rngs=rngs, dtype=dtype)
    self.dropout = make_dropout(dropout, rngs)

  def __call__(self, x):
    h = x if self.norm is None else self.norm(x)
    h = self.act(self.up(h))
    if self.dropout is not None:
      h = self.dropout(h)
    return x + self.down(h)


class SimBa(nnx.Module):

  def __init__(self, input_dim: int, output_dim: int, latent_dim: int, layers: int,
               *, config: SimbaConfig, rngs: nnx.Rngs):
    act, self.dtype = config.resolve()
    mk = lambda dim: config.make_norm(dim, rngs=rngs)
    self.stem = Linear(input_dim, latent_dim, rngs=rngs, dtype=self.dtype)
    self.blocks = nnx.List([
      SimbaBlock(latent_dim, act, mk(latent_dim), expansion=config.expansion,
                 dropout=config.dropout, activation=config.activation,
                 rngs=rngs, dtype=self.dtype) for _ in range(layers)])
    self.stream_norm = mk(latent_dim)
    self.head = Linear(latent_dim, output_dim, rngs=rngs, dtype=self.dtype, gain=config.head_gain)

  @property
  def hidden_linears(self):
    return [self.stem, *(lin for block in self.blocks for lin in (block.up, block.down))]

  def __call__(self, inputs, conditioning=()):
    h = self.stem(concat(inputs, conditioning).astype(self.dtype))
    for block in self.blocks:
      h = block(h)
    if self.stream_norm is not None:
      h = self.stream_norm(h)
    return self.head(h)


class AdaLNSimbaBlock(nnx.Module):
  """x + gate(c) * Linear(act(Linear(AdaLN(x, c)))); the gate starts at 0, so the
  block is the identity at init."""

  def __init__(self, latent_dim: int, cond_dim: int, act, *, expansion: int, dropout: float,
               epsilon: float, activation: str, rngs: nnx.Rngs, dtype=jnp.float32):
    self.act = act
    self.norm = AdaLN(latent_dim, cond_dim, gated=True, epsilon=epsilon, rngs=rngs, dtype=dtype)
    self.up = Linear(latent_dim, expansion * latent_dim, rngs=rngs, dtype=dtype,
                     activation=activation)
    self.down = Linear(expansion * latent_dim, latent_dim, rngs=rngs, dtype=dtype)
    self.dropout = make_dropout(dropout, rngs)

  def __call__(self, x, c):
    h, gate = self.norm(x, c)
    h = self.act(self.up(h))
    if self.dropout is not None:
      h = self.dropout(h)
    return x + gate * self.down(h)


class AdaLNSimBa(nnx.Module):

  def __init__(self, input_dim: int, cond_dim: int, output_dim: int, latent_dim: int,
               layers: int, *, config: AdaLNSimbaConfig, rngs: nnx.Rngs):
    if cond_dim <= 0:
      raise ValueError("AdaLNSimBa needs a conditioning input; use it in a diffusion field")
    act, self.dtype = config.resolve()
    self.act = act
    eps = config.norm_kwargs.get("epsilon", 1e-5)
    self.stem = Linear(input_dim, latent_dim, rngs=rngs, dtype=self.dtype)
    self.cond_embed = Linear(cond_dim, latent_dim, rngs=rngs, dtype=self.dtype, activation="silu")
    self.blocks = nnx.List([
      AdaLNSimbaBlock(latent_dim, latent_dim, act, expansion=config.expansion,
                      dropout=config.dropout, epsilon=eps, activation=config.activation,
                      rngs=rngs, dtype=self.dtype)
      for _ in range(layers)])
    self.final_norm = AdaLN(latent_dim, latent_dim, gated=False, epsilon=eps, rngs=rngs,
                            dtype=self.dtype)
    self.head = Linear(latent_dim, output_dim, rngs=rngs, dtype=self.dtype, gain=config.head_gain)

  @property
  def hidden_linears(self):
    return [self.stem, *(lin for block in self.blocks for lin in (block.up, block.down))]

  def __call__(self, inputs, conditioning=()):
    if not conditioning:
      raise ValueError("AdaLNSimBa needs a conditioning input")
    h = self.stem(concat(inputs).astype(self.dtype))
    c = jax.nn.silu(self.cond_embed(concat(conditioning).astype(self.dtype)))
    for block in self.blocks:
      h = block(h, c)
    return self.head(self.final_norm(h, c))
