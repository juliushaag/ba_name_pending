import copy
import enum

import torch
from torch import nn


class ReduceMethod(enum.Enum):
  MIN = 0
  MAX = 1
  MEAN = 2
  MEANSTD = 3


class Ensemble(nn.Module):

  def __init__(self, n: int, cls : type, *args, **kwargs):
    super().__init__()
    
    self.models = nn.ModuleList([
      cls(*args, **kwargs) for _ in range(n)
    ])

    self.n_models = n

  def forward(self, *args, reduce : ReduceMethod = None, **kwargs) -> list[torch.Tensor]:
    results = list(self.models[i](*args, **kwargs) for i in range(self.n_models))

    results = torch.stack(results, dim=0)
    if reduce == None:
      return results
    if reduce == ReduceMethod.MEAN:
      return results.mean(0)
    elif reduce == ReduceMethod.MIN:
      return results.min(0).values
    elif reduce == ReduceMethod.MAX:
      return results.max(0).values
    elif reduce == ReduceMethod.MEANSTD:
      return results.mean(0), results.std(0)
    
    raise NotImplementedError("Reduce method not implemented: ", reduce)

