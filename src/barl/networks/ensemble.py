import copy
import enum

import torch
from torch import nn
from torch.func import functional_call, stack_module_state, vmap


class ReduceMethod(enum.Enum):
  MIN = 0
  MAX = 1
  MEAN = 2
  MEANSTD = 3


class Ensemble(nn.Module):
  """
  Runs n independently-initialized copies of cls(*args, **kwargs) as a single
  vmapped call instead of a Python loop over n forward passes.
  """

  def __init__(self, n: int, cls : type, *args, **kwargs):
    super().__init__()

    models = [cls(*args, **kwargs) for _ in range(n)]

    params, buffers = stack_module_state(models)

    self._params = nn.ParameterDict({k.replace(".", "__"): v for k, v in params.items()})
    self._buffer_names = list(buffers.keys())
    for k, v in buffers.items():
      self.register_buffer(k.replace(".", "__"), v)

    # meta-device template used by functional_call to describe the module's shape.
    # Set via object.__setattr__ so nn.Module doesn't register it as a submodule --
    # it must stay on the meta device even after .to()/.cuda() move the real params.
    object.__setattr__(self, "_template", copy.deepcopy(models[0]).to("meta"))

    self.n_models = n

  def _stacked_params_and_buffers(self):
    params  = {k.replace("__", "."): v for k, v in self._params.items()}
    buffers = {k: getattr(self, k.replace(".", "__")) for k in self._buffer_names}
    buffers = {k.replace("__", "."): v for k, v in buffers.items()}
    return params, buffers

  def forward(self, *args, reduce : ReduceMethod = None, **kwargs) -> list[torch.Tensor]:
    params, buffers = self._stacked_params_and_buffers()

    # autocast dtype casting doesn't propagate through vmap, so replicate it manually
    # to match the dtype behavior of a plain per-model forward loop under autocast.
    device_type = args[0].device.type if args else "cpu"
    if torch.is_autocast_enabled(device_type):
      amp_dtype = torch.get_autocast_dtype(device_type)
      params = {k: v.to(amp_dtype) for k, v in params.items()}
      args   = tuple(a.to(amp_dtype) if torch.is_tensor(a) and a.is_floating_point() else a for a in args)
      kwargs = {k: (v.to(amp_dtype) if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in kwargs.items()}

    def call_model(params, buffers, *args, **kwargs):
      return functional_call(self._template, (params, buffers), args, kwargs)

    results = vmap(call_model, in_dims=(0, 0) + (None,) * len(args))(params, buffers, *args, **kwargs)

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
