import math

import torch
from torch import nn

from barl.networks.registry import register_component



@register_component
class AutotuneRegulator(nn.Module):

  def __init__(self, name : str, lr : float = 1e-3, alpha_init : float = 1e-2, max_alpha : float = 5.0, min_alpha : float = 1e-4, target : float = -1.0, **kwargs):

    super().__init__(**kwargs)  

    self.name = name

    self.register_buffer("log_alpha", torch.tensor([math.log(alpha_init)])) 

    self.register_buffer("lr", torch.tensor([lr]))
    self.register_buffer("target_value", torch.tensor([target]))
    self.register_buffer("max_alpha_log", torch.tensor(math.log(max_alpha)))
    self.register_buffer("min_alpha_log", torch.tensor(math.log(min_alpha)))

  def value(self) -> torch.Tensor:
    return self.log_alpha.exp()

  
  def target(self) -> torch.Tensor:
    return self.target_value

  @torch.no_grad()
  def loss(self, value : torch.Tensor, log : dict, losses : dict):

    self.log_alpha = self.log_alpha - self.lr * (value - self.target_value).mean()
    self.log_alpha.clamp_(self.min_alpha_log, self.max_alpha_log)

    log[f"stats/{self.name}_value"] = value.mean()
    log[f"stats/{self.name}_scale"] = self.log_alpha.exp()
    
  def create_optimizer(self):
    return {}