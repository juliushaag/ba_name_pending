import copy

import torch
from torch import nn


class EMA(nn.Module):
  """Exponential moving average wrapper. Owns a frozen `target` copy of
  `source` as a proper submodule so state_dict/load_state_dict cover it."""

  def __init__(self, source: nn.Module, start_decay: float = 0.9, end_decay: float = 0.99, warmup_steps: int = 5000):
    super().__init__()

    assert 0.0 <= start_decay <= end_decay <= 1.0

    self.source = source
    self.target = copy.deepcopy(source).requires_grad_(False)

    self.register_buffer("decay",        torch.tensor(start_decay))
    self.register_buffer("_step_count",  torch.tensor(0))

    self._start_decay  = start_decay
    self._end_decay    = end_decay
    self._warmup_steps = warmup_steps

  @torch.no_grad()
  def step(self):
    self._step_count += 1

    # Stays in pure tensor ops (no .item()) so this composes with vmap.
    t = torch.clamp(self._step_count.float() / self._warmup_steps, max=1.0)
    self.decay.copy_(self._start_decay + t * (self._end_decay - self._start_decay))

    tau = self.decay
    for p_t, p_s in zip(self.target.parameters(), self.source.parameters()):
      p_t.copy_(tau * p_t + (1.0 - tau) * p_s)  # .data in-place ops are disallowed under vmap
    for b_t, b_s in zip(self.target.buffers(), self.source.buffers()):
      b_t.copy_(b_s)

  def forward(self, *args, **kwargs):
    return self.target(*args, **kwargs)


class Shadow(nn.Module):
  """Hard-copy snapshot wrapper -- EMA's trust-region counterpart.
  on_epoch_start() fully overwrites `target` with `source`'s current weights
  instead of blending; named after the Policy lifecycle hook meant to drive it."""

  def __init__(self, source: nn.Module):
    super().__init__()

    self.source = source
    self.target = copy.deepcopy(source).eval().requires_grad_(False)
    
  @torch.no_grad()
  def on_epoch_start(self):
    for p_t, p_s in zip(self.target.parameters(), self.source.parameters()):
      p_t.copy_(p_s)
    for b_t, b_s in zip(self.target.buffers(), self.source.buffers()):
      b_t.copy_(b_s)

  def forward(self, *args, **kwargs):
    return self.target(*args, **kwargs)