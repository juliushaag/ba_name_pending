


import torch


def bfloat16():
  return torch.amp.autocast("cuda", torch.bfloat16)

def float32():
  return torch.amp.autocast("cuda", torch.float32)