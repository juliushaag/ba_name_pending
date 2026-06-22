import copy

import torch
from torch import nn


class EMA(nn.Module):
  """
  Exponential moving average wrapper.

  Owns a frozen copy (target) of the source module as a proper submodule,
  so state_dict / load_state_dict cover it automatically. The decay scalar
  and step counter are registered buffers for the same reason.

  Usage:
      self.q_ema = EMA(self.q_function, end_decay=1 - tau)

      # each train step:
      self.q_ema.step()

      # inference (runs the frozen target):
      q_val = self.q_ema(obs, action)
  """

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

    # linearly ramp decay from start to end over warmup_steps
    t = min(self._step_count.item() / self._warmup_steps, 1.0)
    self.decay.fill_(self._start_decay + t * (self._end_decay - self._start_decay))

    tau = self.decay.item()
    for p_t, p_s in zip(self.target.parameters(), self.source.parameters()):
      p_t.data.copy_(tau * p_t.data + (1.0 - tau) * p_s.data)
    for b_t, b_s in zip(self.target.buffers(), self.source.buffers()):
      b_t.copy_(b_s)

  def forward(self, *args, **kwargs):
    return self.target(*args, **kwargs)
