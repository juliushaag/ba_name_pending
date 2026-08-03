
from typing import Optional

import torch
from torch import nn

import gymnasium as gym

import numpy as np

_ACTIVATIONS = dict(
  silu =nn.SiLU,
  relu=nn.ReLU,
  gelu=nn.GELU,
)

class ResidualBlock(nn.Module):
  """One pre-norm hidden layer: `x + Linear(ACT(LN(x)))`. The norm sits on the
  branch rather than after its Linear, so the skip carries an unnormalized
  identity and a branch whose activation has a one-sided range (ReLU, SiLU)
  cannot walk the stream's mean upward once per layer."""

  def __init__(self, latent_dim : int, ACT : type, layernorm : bool):
    super().__init__()

    self.block = nn.Sequential(
      *([nn.LayerNorm(latent_dim)] if layernorm else []),
      ACT(),
      nn.Linear(latent_dim, latent_dim),
    )

  def forward(self, x : torch.Tensor) -> torch.Tensor:
    return x + self.block(x)


class MLP(nn.Module):

  def __init__(self, input_dim : int, output_dim : int, latent_dim : int, hidden_layers : int, activation : str = "silu", layernorm=True, final_activation=False, residual=False):
    super().__init__()

    assert activation in _ACTIVATIONS, f"{activation} not in available activations {_ACTIVATIONS.keys()}"

    ACT = _ACTIVATIONS[activation]

    self.input_dim = input_dim
    self.layernorm = layernorm
    self.output_dim = output_dim
    self.latent_dim = latent_dim
    self.hidden_layers = hidden_layers
    self.final_activation = final_activation
    self.residual = residual

    self.layers = nn.Sequential()

    # A residual stem is a bare linear embedding of the stream -- the activation
    # the plain path puts here would be immediately followed by each block's own.
    self.layers.append(nn.Linear(self.input_dim, self.latent_dim))
    if not self.residual:
      self.layers.append(ACT())

    for _ in range(self.hidden_layers):
      if self.residual:
        # One Linear per block, same as the plain path, so `residual` swaps the
        # wiring rather than the width -- the two differ only by the head norm.
        self.layers.append(ResidualBlock(self.latent_dim, ACT, self.layernorm))
        continue

      self.layers.append(nn.Linear(self.latent_dim, self.latent_dim))
      if self.layernorm:
        self.layers.append(nn.LayerNorm(self.latent_dim))

      self.layers.append(ACT())

    # Nothing has normalized the residual stream itself, only each branch's
    # input, so without this the head sees a scale that grows with depth.
    if self.residual and self.layernorm:
      self.layers.append(nn.LayerNorm(self.latent_dim))

    self.layers.append(nn.Linear(self.latent_dim, self.output_dim))

    if self.final_activation:
      if self.layernorm:
        self.layers.append(nn.LayerNorm(self.output_dim))

      self.layers.append(ACT())

  def forward(self, *x : torch.Tensor):
    if len(x) == 1:
      x = x[0]
    else:
      x = torch.cat(x, dim=-1)
    x =  self.layers(x)
    return x