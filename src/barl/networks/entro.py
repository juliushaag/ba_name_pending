import math

import torch
from torch import nn

from barl.networks.registry import register_component



@register_component
class EntropyRegulator(nn.Module):

  def __init__(self, action_dim : int, alpha_init : float = 1e-2, max_alpha : float = 5.0, min_alpha : float = 1e-4, entropy_factor : float = -1.0, **kwargs):

    super().__init__(**kwargs)

    self.log_alpha     = nn.Parameter(torch.tensor(math.log(alpha_init)))

    self.register_buffer("target_entropy", torch.tensor([entropy_factor * action_dim]))
    self.register_buffer("max_alpha_log", torch.tensor(math.log(max_alpha)))
    self.register_buffer("min_alpha_log", torch.tensor(math.log(min_alpha)))

  def clamp_log_alpha(self):
    with torch.no_grad():
      self.log_alpha.clamp_(self.min_alpha_log, self.max_alpha_log)


  def value(self) -> torch.Tensor:
    return self.log_alpha.exp().detach()


  def loss(self, entropy : torch.Tensor, log : dict, losses : dict):
    self.clamp_log_alpha()

    losses["alpha"] = (self.log_alpha * (entropy.detach() - self.target_entropy)).mean()

    log["stats/entropy"] = entropy.detach().mean()
    log["stats/alpha"] = self.log_alpha.detach().exp()
    
  def create_optimizer(self):
    return { "alpha" : torch.optim.Adam([self.log_alpha], lr=1e-3)}