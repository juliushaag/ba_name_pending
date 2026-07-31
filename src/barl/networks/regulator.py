import math

import torch
from torch import nn

from barl.networks.registry import register_component

from barl.policy import Component


@register_component
class AutotuneRegulator(Component, nn.Module):

  def __init__(self, env_stats = None, lr : float = 3e-4, alpha_init : float = 1e-2, max_alpha : float = 5.0, min_alpha : float = 1e-4, target : float = -1.0, **kwargs):

    super().__init__(env_stats, **kwargs)

    self.lr = float(lr)

    self.log_alpha = nn.Parameter(torch.tensor([math.log(alpha_init)]), requires_grad=True)

    self.register_buffer("target_value", torch.tensor([target]))
    self.register_buffer("max_alpha_log", torch.tensor(math.log(max_alpha)))
    self.register_buffer("min_alpha_log", torch.tensor(math.log(min_alpha)))

  def value(self) -> torch.Tensor:
    return self.log_alpha.exp()

  
  def target(self) -> torch.Tensor:
    return self.target_value

  def loss(self, value : torch.Tensor, losses : dict):

    losses[f"{self.slot}_alpha"] =  self.log_alpha * (value.detach() - self.target_value).mean()

    self.metrics.scalar("alpha", self.log_alpha.detach().exp())
    self.metrics.scalar("target", self.target_value)
    self.metrics.dist("value", value.detach())
    self.metrics.dist("value_scaled", value.detach() * self.value())

  def create_optimizer(self):
    # Keyed by slot, so AMDP's two regulators stay distinct without a `name` kwarg.
    return { f"{self.slot}_alpha" : torch.optim.Adam([self.log_alpha], lr=self.lr, fused=True) }