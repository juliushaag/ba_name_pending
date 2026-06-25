
from typing import Optional

import torch
from torch import nn

import gymnasium as gym

import numpy as np

_ACTIVATIONS = dict(
  silu = nn.SiLU,
  relu=nn.ReLU,
)

class MLP(nn.Module):

  def __init__(self, input_dim : int, output_dim : int, latent_dim : int, hidden_layers : int, activation : str = "silu", output_space : Optional[gym.Space] = None, layernorm=True, final_activation=False):
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
      

    self.output_fn = self._select_output(output_space) if output_space is not None else nn.Identity()

  def _select_output(self, output_space : gym.Space):
    
    if isinstance(output_space, gym.spaces.Box):
      low = torch.tensor(output_space.low, dtype=torch.float32)
      high = torch.tensor(output_space.high, dtype=torch.float32)

      if (np.absolute(low) - np.absolute(high)).abs().sum() == 0: # sym

        scale = torch.tensor((output_space.high - output_space.low) / 2.0)
        bias  = torch.tensor((output_space.high + output_space.low) / 2.0)

        self.register_buffer("output_scale", scale)
        self.register_buffer("output_bias", bias)

        self.output_fn = lambda x: torch.tanh(x) * self.output_scale + self.output_bias

      else:
        self.output_fn = lambda x: (
            low + 0.5 * (torch.tanh(x) + 1.0) * (high - low)
        )

    elif isinstance(output_space, gym.spaces.Discrete):
      self.output_fn = lambda x: torch.softmax(x, dim=-1)

    else:
      raise NotImplementedError(f"No output function defined for {output_space}")

  def forward(self, *x : torch.Tensor):
    if len(x) == 1:
      x = x[0]
    else:
      x = torch.cat(x, dim=-1)
    x =  self.layers(x)
    x = self.output_fn(x)
    return x