import copy
import math
from typing import Optional

import numpy as np
import torch
from torch import nn

from abc import ABC, abstractmethod

import gymnasium as gym
import torch.optim.lr_scheduler as sched

class Policy(nn.Module, ABC):
  def __init__(self, env : gym.Env):
    super().__init__()

    action_space = env.action_space
    observation_space = env.observation_space

    if isinstance(action_space, gym.spaces.Discrete):
      self.action_dim = action_space.n
      self.action_discrete = True
    elif isinstance(action_space, gym.spaces.Box):
      self.action_dim = np.prod(action_space.shape).item()
      self.action_discrete = False

      self.actor_log_std = nn.Parameter(torch.zeros(self.action_dim))
      self.register_buffer("act_low",  torch.as_tensor(action_space.low,  dtype=torch.float32))
      self.register_buffer("act_high", torch.as_tensor(action_space.high, dtype=torch.float32))
    else:
      raise NotImplementedError(f"{action_space} type not supported ... yet")

    self.observation_shape = env.observation_space.shape
    self.observation_dim = np.prod(self.observation_shape)

    self.register_buffer("obs_std",  torch.ones(observation_space.shape[-1]))

    self.register_buffer("obs_mean", torch.zeros(observation_space.shape[-1]))

    self.ema_buffers = list()

  def register_ema(self, source : nn.Module, start_decay = 0.9, end_decay = 0.99) -> nn.Module:
    target = copy.deepcopy(source).requires_grad_(False).eval()
    self.ema_buffers.append((source, target, torch.tensor(start_decay), start_decay, end_decay))
    return target
  
  def update_ema(self):

    for source, target, decay, start, end in self.ema_buffers:
      decay.fill_(min(decay.item() + (end - start) / 5000, end))
      self._ema_soft_update(target, source, decay)

  def set_obs_normalization(self, mean, std):

    self.obs_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
    
    self.obs_std.copy_(torch.as_tensor(std, dtype=torch.float32))

  def normalize_obs(self, obs : torch.Tensor) -> torch.Tensor:
    return (obs - self.obs_mean) / self.obs_std  
  
  @staticmethod
  def _ema_soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
      target_param.data.copy_(tau * target_param.data + (1 - tau) * source_param.data)

    for be, bm in zip(target.buffers(), source.buffers()):
      be.copy_(bm)
      
  @property
  def device(self):
    return next(self.parameters()).device
  
  @abstractmethod
  def create_optimizer(self) -> torch.optim.Optimizer:
    ...

  def create_scheduler(self, optim : torch.optim.Optimizer, epochs : int, min_lr = 3e-5, base_lr = 3e-4, warmup_epochs = 8) -> torch.optim.lr_scheduler.LRScheduler:
    min_lr_ratio = min_lr / base_lr if base_lr > 0 else 0.0
    
    def lr_lambda(epoch: int) -> float:
      if epoch < warmup_epochs:
        # Linear warmup: 0 -> 1
        return (float(epoch) + 1) / float(max(1, warmup_epochs))
      else:
        # Cosine decay: 1 -> min_lr_ratio
        progress = (epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        progress = min(1.0, progress)  # Clamp to avoid going below min_lr
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
  
    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    
  def num_params(self):
    return sum(p.numel() for p in self.parameters() if p.requires_grad)
  
  @abstractmethod
  def train_step(self, batch : torch.DictType) -> tuple[torch.Tensor, dict]:
    """
    Takes in a dict with the following keys

    observations  : [Batch, Horizon, ObservationDim]
    actions       : [Batch, Horizon, ActionDim]
    """
    ...

  @abstractmethod
  @torch.inference_mode
  def inference_step(self, observation : torch.Tensor, history : Optional[dict] = None) -> torch.Tensor:
    ...

  @torch.inference_mode()
  def evaluate(self, env : gym.Env, runs = 3):

    was_training = False
    if self.training:
      was_training = True

    run_data = list()
    self.eval()

    for i in range(runs):
      obs, _ = env.reset()
      done = False

      data = dict(
        frames=list(),
        rewards=list()
      )


      while not done:
        action = self.inference_step(torch.from_numpy(obs).float().unsqueeze(0))

        obs, reward, term, trunc, _ = env.step(action[0].cpu().numpy())    
        done = term or trunc
        frame = env.render()

        data["rewards"].append(reward)
        data["frames"].append(frame)

      data["terminated"] = term
      run_data.append(data)
    
    if was_training:
      self.train()

    return run_data



  

