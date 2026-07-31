import math
from abc import ABC, abstractmethod

import torch
from torch import nn

from barl.networks.registry import register_component

from barl.policy import Component


class Squash(Component, nn.Module, ABC):
  """Maps an actor's unconstrained output to a canonical [-1, 1]-per-dimension
  action and back (SAC's tanh, AMDP's erf, ...)."""

  def __init__(self, env_stats=None, **kwargs):
    """Absorbs kwargs a concrete Squash doesn't need -- e.g. AMDP always
    passes `erf_k=...` regardless of which class the squash slot resolves to."""
    super().__init__(env_stats)

  @abstractmethod
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Raw actor output -> canonical [-1, 1]-per-dimension action."""
    ...

  @abstractmethod
  def inverse(self, y: torch.Tensor) -> torch.Tensor:
    """Canonical [-1, 1]-per-dimension action -> raw actor output."""
    ...

  @abstractmethod
  def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
    """log|dy/dx|, elementwise; `y` accepted to avoid a redundant forward() if already computed."""
    ...


@register_component
class TanhSquash(Squash):
  """Standard tanh squash (Haarnoja et al., 2018 SAC appendix C)."""

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)

  def inverse(self, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.atanh(y.clamp(-1 + eps, 1 - eps))

  def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor = None, eps: float = 1e-6) -> torch.Tensor:
    if y is None:
      y = self.forward(x)
    return torch.log((1 - y.pow(2)).clamp_min(eps))


@register_component
class ErfSquash(Squash):
  """erf squash (Thilges et al. 2026 Eq. 14), scaled by `erf_k` -- a fixed
  (not learned) function of the owning actor's noise schedule, chosen so
  this squash's log-Jacobian cancels out of AMDP's direct-score regression
  (see AMDP)."""

  def __init__(self, env_stats=None, erf_k: torch.Tensor | float = 1.0, **kwargs):
    super().__init__(env_stats, **kwargs)
    if not torch.is_tensor(erf_k):
      erf_k = torch.tensor(float(erf_k))
    self.register_buffer("erf_k", erf_k)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.erf(self.erf_k * x).clamp(-1.0, 1.0)

  def inverse(self, y: torch.Tensor, eps: float = 0.05) -> torch.Tensor:
    return torch.erfinv(y.clamp(-1 + eps, 1 - eps)) / self.erf_k

  def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
    return math.log(2 / math.sqrt(math.pi)) + torch.log(self.erf_k) - (self.erf_k * x).pow(2)
