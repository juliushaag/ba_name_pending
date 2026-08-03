import torch
from torch import nn


class BatchRenorm1d(nn.Module):
  """Batch Renormalization (Ioffe, 2017) over a (B, C) activation.

  Plain BatchNorm normalizes by the *minibatch* moments at train time and by
  the running ones at eval time, so the function the critic learns is not the
  function it is later evaluated with -- which is what makes BatchNorm
  notoriously unstable in off-policy RL, where the same network is queried on
  a fresh single-transition batch every env step. BatchRenorm closes that gap:
  it still normalizes with the batch moments, then applies an affine
  correction (r, d) that maps them onto the running ones, with r/d clipped to
  a window that opens over `warmup_steps`. Training starts as BatchNorm and
  converges to the eval-time normalization, so train and eval agree.

  This is the normalization CrossQ (Bhatt et al., ICLR 2024) relies on to
  drop target networks entirely."""

  def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.01,
               r_max: float = 3.0, d_max: float = 5.0, warmup_steps: int = 100_000):
    super().__init__()

    self.eps          = eps
    self.momentum     = momentum
    self.r_max        = r_max
    self.d_max        = d_max
    self.warmup_steps = warmup_steps

    self.weight = nn.Parameter(torch.ones(num_features))
    self.bias   = nn.Parameter(torch.zeros(num_features))

    self.register_buffer("running_mean", torch.zeros(num_features))
    self.register_buffer("running_var",  torch.ones(num_features))
    self.register_buffer("step",         torch.zeros((), dtype=torch.long))

  def _limits(self) -> tuple[torch.Tensor, torch.Tensor]:
    """r and d open linearly from (1, 0) -- i.e. exactly BatchNorm -- to
    (r_max, d_max) over warmup_steps, so early training isn't corrected
    towards running statistics that are still meaningless."""
    t = (self.step.float() / self.warmup_steps).clamp(max=1.0)
    return 1.0 + t * (self.r_max - 1.0), t * self.d_max

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if not self.training:
      normed = (x - self.running_mean) / (self.running_var + self.eps).sqrt()
      return normed * self.weight + self.bias

    batch_mean = x.mean(dim=0)
    batch_var  = x.var(dim=0, unbiased=False)
    batch_std  = (batch_var + self.eps).sqrt()

    with torch.no_grad():
      running_std = (self.running_var + self.eps).sqrt()
      r_max, d_max = self._limits()
      r = (batch_std / running_std).clamp(1.0 / r_max, r_max)
      d = ((batch_mean - self.running_mean) / running_std).clamp(-d_max, d_max)

      self.running_mean.lerp_(batch_mean, self.momentum)
      self.running_var.lerp_(batch_var,  self.momentum)
      self.step += 1

    normed = (x - batch_mean) / batch_std * r + d
    return normed * self.weight + self.bias
