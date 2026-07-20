


import torch


def bfloat16():
  return torch.amp.autocast("cuda", torch.bfloat16)