from dataclasses import dataclass

import jax.numpy as jnp
from flax import nnx

from barl_jax.networks.base import Linear, NetworkConfig, make_dropout, register_network


@register_network
@dataclass(frozen=True)
class SimbaConfig(NetworkConfig):
  """SimBa (Lee et al., ICLR 2025): a linear embedding, `layers` pre-norm residual
  blocks whose hidden layer is `expansion` times wider, and a final norm. The
  observation running-statistics normalizer is the policy's, not this net's."""
  class_name: str = "SimBa"
  activation: str = "relu"
  expansion: int = 4

  def build(self, input_dim, output_dim, latent_dim, layers, *, rngs):
    return SimBa(input_dim, output_dim, latent_dim, layers, config=self, rngs=rngs)


class SimbaBlock(nnx.Module):
  """x + Linear(act(Linear(Norm(x))))."""

  def __init__(self, latent_dim: int, act, norm, *, expansion: int, dropout: float,
               rngs: nnx.Rngs, dtype=jnp.float32):
    self.act, self.norm = act, norm
    self.up = Linear(latent_dim, expansion * latent_dim, rngs=rngs, dtype=dtype)
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
                 dropout=config.dropout, rngs=rngs, dtype=self.dtype) for _ in range(layers)])
    self.stream_norm = mk(latent_dim)
    self.head = Linear(latent_dim, output_dim, rngs=rngs, dtype=self.dtype)

  @property
  def hidden_linears(self):
    return [self.stem, *(lin for block in self.blocks for lin in (block.up, block.down))]

  def __call__(self, *x):
    h = x[0] if len(x) == 1 else jnp.concatenate(x, axis=-1)
    h = self.stem(h.astype(self.dtype))
    for block in self.blocks:
      h = block(h)
    if self.stream_norm is not None:
      h = self.stream_norm(h)
    return self.head(h)
