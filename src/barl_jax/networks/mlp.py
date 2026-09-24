from dataclasses import dataclass

import jax.numpy as jnp
from flax import nnx

from barl_jax.networks.base import (
  Linear,
  NetworkConfig,
  act_norm,
  make_dropout,
  normalize_weights,  # noqa: F401  (re-exported: the projection is defined for any trunk)
  register_network,
)


@register_network
@dataclass(frozen=True)
class MLPConfig(NetworkConfig):
  class_name: str = "MLP"
  norm_input: bool = False
  norm_stem: bool = False
  norm_after_act: bool = True
  final_activation: bool = False
  residual: bool = False

  def build(self, input_dim, output_dim, latent_dim, layers, *, rngs):
    return MLP(input_dim, output_dim, latent_dim, layers, config=self, rngs=rngs)


class Block(nnx.Module):
  """Linear -> dropout -> activation/norm, optionally wrapped in a skip."""

  def __init__(self, latent_dim: int, act, norm, *, dropout: float, norm_after_act: bool,
               residual: bool, rngs: nnx.Rngs, dtype=jnp.float32):
    self.lin = Linear(latent_dim, latent_dim, rngs=rngs, dtype=dtype)
    self.act, self.norm = act, norm
    self.dropout = make_dropout(dropout, rngs)
    self.norm_after_act, self.residual = norm_after_act, residual

  def __call__(self, x):
    h = self.lin(x)
    if self.dropout is not None:
      h = self.dropout(h)
    h = act_norm(h, self.act, self.norm, self.norm_after_act)
    return x + h if self.residual else h


class MLP(nnx.Module):
  """Stem, `layers` x (Linear -> act/norm), head. With `residual` the blocks carry
  skips, the stem is a bare embedding and the stream is normalized before the head.

  `dropout` follows each hidden Linear, before its activation/norm (DroQ, Hiraoka
  et al., 2022); it is off at eval via `nnx.Module.eval()`. `norm_input` and
  `norm_stem` normalize the raw input and the stem output -- what CrossQ-style
  nets need so the joint current+next forward shares statistics."""

  def __init__(self, input_dim: int, output_dim: int, latent_dim: int, layers: int,
               *, config: MLPConfig, rngs: nnx.Rngs):
    act, dtype = config.resolve()
    self.act, self.dtype = act, dtype
    self.residual = config.residual
    self.final_activation = config.final_activation
    self.norm_after_act = config.norm_after_act

    mk = lambda dim: config.make_norm(dim, rngs=rngs)
    self.input_norm = mk(input_dim) if config.norm_input else None
    self.stem = Linear(input_dim, latent_dim, rngs=rngs, dtype=dtype)
    self.stem_norm = mk(latent_dim) if config.norm_stem else None
    self.blocks = nnx.List([
      Block(latent_dim, act, mk(latent_dim), dropout=config.dropout,
            norm_after_act=config.norm_after_act, residual=config.residual,
            rngs=rngs, dtype=dtype) for _ in range(layers)])
    # A skip stream's scale grows with depth; the head needs it normalized.
    self.stream_norm = mk(latent_dim) if config.residual else None
    self.head = Linear(latent_dim, output_dim, rngs=rngs, dtype=dtype)
    self.head_norm = mk(output_dim) if config.final_activation else None

  @property
  def hidden_linears(self):
    return [self.stem, *(block.lin for block in self.blocks)]

  def __call__(self, *x):
    h = x[0] if len(x) == 1 else jnp.concatenate(x, axis=-1)
    h = h.astype(self.dtype)
    if self.input_norm is not None:
      h = self.input_norm(h)

    h = act_norm(self.stem(h), None if self.residual else self.act,
                 self.stem_norm, self.norm_after_act)
    for block in self.blocks:
      h = block(h)
    if self.stream_norm is not None:
      h = self.stream_norm(h)

    h = self.head(h)
    if self.final_activation:
      h = act_norm(h, self.act, self.head_norm, True)
    return h
