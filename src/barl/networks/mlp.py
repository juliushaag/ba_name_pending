
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

class MLP(nn.Module):

  def __init__(self, input_dim : int, output_dim : int, latent_dim : int, hidden_layers : int, activation : str = "silu", layernorm=True, final_activation=False):
    super().__init__()

    assert activation in _ACTIVATIONS, f"{activation} not in available activations {_ACTIVATIONS.keys()}"

    ACT = _ACTIVATIONS[activation]

    self.input_dim = input_dim
    self.layernorm = layernorm
    self.output_dim = output_dim
    self.latent_dim = latent_dim
    self.hidden_layers = hidden_layers
    self.final_activation = final_activation

    self.layers = nn.Sequential()
    self.layers.extend([nn.Linear(self.input_dim, self.latent_dim), ACT()])
      
    for _ in range(self.hidden_layers):
      self.layers.append(nn.Linear(self.latent_dim, self.latent_dim))
      if self.layernorm:
        self.layers.append(nn.LayerNorm(self.latent_dim))
  
      self.layers.append(ACT())
    
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