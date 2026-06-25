import math
from typing import Optional, Any

import numpy as np
import torch
from torch import nn

from abc import ABC, abstractmethod

import gymnasium as gym
import torch.optim.lr_scheduler as sched


class HParam:

  def __init__(self, default: Any, description: str, range: tuple = None, choices: list = None):
    self.default     = default
    self.description = description
    self.range       = range    # (min, max) for numeric params
    self.choices     = choices  # exhaustive set of valid values

  def validate(self, name: str, value: Any):
    if self.choices is not None and value not in self.choices:
      raise ValueError(f"HParam '{name}': {value!r} not in valid choices {self.choices}")
    if self.range is not None:
      lo, hi = self.range
      if not (lo <= value <= hi):
        raise ValueError(f"HParam '{name}': {value!r} out of range [{lo}, {hi}]")

  def __repr__(self):
    parts = [f"default={self.default!r}", f"desc={self.description!r}"]
    if self.range   is not None: parts.append(f"range={self.range}")
    if self.choices is not None: parts.append(f"choices={self.choices}")
    return f"HParam({', '.join(parts)})"


class Policy(nn.Module, ABC):

  _classes = dict()

  # ------------------------------------------------------------------
  # HParam helpers
  # ------------------------------------------------------------------

  @classmethod
  def _hparam_descriptors(cls) -> dict[str, HParam]:
    """Collect all HParam class attributes walking the MRO."""
    out = {}
    for klass in reversed(cls.__mro__):
      for k, v in vars(klass).items():

        if isinstance(v, HParam):
          out[k] = v
    return out

  @classmethod
  def resolve_hparams(cls, kwargs: dict) -> dict:
    """
    Merge caller-supplied kwargs with class-level HParam defaults,
    validate every value, and return a plain {name: value} dict.
    """
    descriptors = cls._hparam_descriptors()
    resolved = {}
    for name, hp in descriptors.items():
      value = kwargs.get(name, hp.default)
      hp.validate(name, value)
      resolved[name] = value
      setattr(cls, name, value)

    unknown = set(kwargs) - set(descriptors)
    if unknown:
      raise TypeError(f"{cls.__name__} got unexpected hparams: {unknown}")

    return resolved

  @classmethod
  def describe_hparams(cls):
    """Print a human-readable table of all hyperparameters."""
    descriptors = cls._hparam_descriptors()
    if not descriptors:
      print(f"{cls.__name__}: no registered HParams")
      return
    col = max(len(k) for k in descriptors) + 2
    print(f"\n{cls.__name__} hyperparameters:")
    print(f"  {'name':<{col}} {'default':<12} {'range/choices':<20} description")
    print(f"  {'-'*col} {'-'*12} {'-'*20} -----------")
    for name, hp in descriptors.items():
      constraint = ""
      if hp.range   is not None: constraint = f"{hp.range[0]}..{hp.range[1]}"
      if hp.choices is not None: constraint = str(hp.choices)
      print(f"  {name:<{col}} {str(hp.default):<12} {constraint:<20} {hp.description}")
    print()

  @classmethod
  def sample_hparams(cls, rng: np.random.Generator = None) -> dict:
    """
    Sample one value per HParam from its range/choices uniformly.
    Falls back to the default for params with no range or choices.
    """
    if rng is None:
      rng = np.random.default_rng()
    descriptors = cls._hparam_descriptors()
    out = {}
    for name, hp in descriptors.items():
      if hp.choices is not None:
        out[name] = rng.choice(hp.choices)
      elif hp.range is not None:
        lo, hi = hp.range
        if isinstance(hp.default, int):
          out[name] = int(rng.integers(lo, hi + 1))
        else:
          out[name] = float(rng.uniform(lo, hi))
      else:
        out[name] = hp.default
    return out

  # ------------------------------------------------------------------
  # Registry
  # ------------------------------------------------------------------

  @staticmethod
  def make(name: str, env: gym.Env, *args, **kwargs) -> "Policy":
    if name not in Policy._classes:
      raise KeyError(f"{name} not in registered policies: {list(Policy._classes.keys())}")
    return Policy._classes[name](env, *args, **kwargs)

  @staticmethod
  def available() -> list[str]:
    return list(Policy._classes.keys())

  # ------------------------------------------------------------------
  # Construction
  # ------------------------------------------------------------------

  def __init__(self, env: gym.Env, **kwargs):
    super().__init__()

    action_space      = env.action_space
    observation_space = env.observation_space

    if isinstance(action_space, gym.spaces.Discrete):
      self.action_dim      = action_space.n
      self.action_discrete = True
    elif isinstance(action_space, gym.spaces.Box):
      self.action_dim      = np.prod(action_space.shape).item()
      self.action_discrete = False

      self.register_buffer("act_low",  torch.as_tensor(action_space.low,  dtype=torch.float32))
      self.register_buffer("act_high", torch.as_tensor(action_space.high, dtype=torch.float32))
    else:
      raise NotImplementedError(f"{action_space} type not supported ... yet")

    self.observation_shape = env.observation_space.shape
    self.observation_dim   = np.prod(self.observation_shape)

    self.register_buffer("obs_std",  torch.ones(observation_space.shape[-1]))
    self.register_buffer("obs_mean", torch.zeros(observation_space.shape[-1]))

    self.hparams: dict = self.resolve_hparams(kwargs)
    

  # ------------------------------------------------------------------
  # Checkpointing
  # ------------------------------------------------------------------

  def save(self, path: str):
    torch.save({
      "policy_class": self.__class__.__name__,
      "state_dict":   self.state_dict(),
      "hparams":      self.hparams,
    }, path)

  @staticmethod
  def load(type: str, path: str, env: gym.Env) -> "Policy":
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    policy = Policy._classes[type](env, **ckpt["hparams"])
    policy.load_state_dict(ckpt["state_dict"])
    return policy

  @staticmethod
  def load_any(path: str, env: gym.Env) -> "Policy":
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    return Policy.load(ckpt["policy_class"], path, env)

  # ------------------------------------------------------------------
  # Observation normalisation
  # ------------------------------------------------------------------

  def set_obs_normalization(self, mean, std):
    self.obs_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
    self.obs_std.copy_(torch.as_tensor(std,  dtype=torch.float32))

  def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
    return (obs - self.obs_mean) / self.obs_std

  # ------------------------------------------------------------------
  # Misc
  # ------------------------------------------------------------------

  @property
  def device(self):
    return next(self.parameters()).device

  def num_params(self):
    return sum(p.numel() for p in self.parameters() if p.requires_grad)

  def create_scheduler(self, optim: torch.optim.Optimizer, epochs: int, min_lr=3e-5, base_lr=3e-4, warmup_epochs=8) -> torch.optim.lr_scheduler.LRScheduler:
    min_lr_ratio = min_lr / base_lr if base_lr > 0 else 0.0

    def lr_lambda(epoch: int) -> float:
      if epoch < warmup_epochs:
        return (float(epoch) + 1) / float(max(1, warmup_epochs))
      else:
        progress    = (epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        progress    = min(1.0, progress)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

  # ------------------------------------------------------------------
  # Abstract interface
  # ------------------------------------------------------------------

  @abstractmethod
  def create_optimizer(self) -> torch.optim.Optimizer:
    ...

  @abstractmethod
  def train_step(self, batch: torch.DictType, frozen : "Policy"= None) -> tuple[torch.Tensor, dict]:
    ...

  @abstractmethod
  @torch.inference_mode()
  def inference_step(self, observation: torch.Tensor, history: Optional[dict] = None) -> torch.Tensor:
    ...

  # ------------------------------------------------------------------
  # Evaluation
  # ------------------------------------------------------------------

  @torch.inference_mode()
  def evaluate(self, env: gym.Env, runs=3):
    was_training = self.training
    self.eval()
    run_data = []

    for _ in range(runs):
      obs, _ = env.reset()
      done   = False
      data   = dict(frames=[], rewards=[])

      while not done:
        action = self.inference_step(torch.from_numpy(obs).float().unsqueeze(0))
        obs, reward, term, trunc, _ = env.step(action[0].cpu().numpy())
        done = term or trunc
        data["rewards"].append(reward)
        data["frames"].append(env.render())

      data["terminated"] = term
      run_data.append(data)

    if was_training:
      self.train()

    return run_data


class FlowPolicy(Policy):

  def __init__(self, env, flow_steps : int, **kwargs):
    super().__init__(env, **kwargs)

    self.flow_steps = flow_steps
  
  @abstractmethod
  def step_flow(self, obs : torch.Tensor, action : torch.Tensor, dt : torch.Tensor, inference : bool = False):
    ...

  def sinusoidal_emb(self, t, time_dim):
    shape = t.shape[:-1]  # all dims except the trailing 1
    t = t.reshape(-1)
    half = time_dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
    args = t[:, None] * 1000.0 * freqs[None]
    time_emb = torch.cat([args.sin(), args.cos()], dim=-1)
    return time_emb.reshape(*shape, time_dim)

  @torch.no_grad()
  def generate_action(self, obs: torch.Tensor, z : torch.Tensor):
    B   = obs.shape[0]
    dt  = 1.0 / self.flow_steps

    with torch.amp.autocast("cuda", torch.bfloat16):
      z = z if z is not None else torch.randn((B, self.action_dim), device=self.device)
      for i in range(self.flow_steps):
        t     = torch.full((B, 1), i * dt, device=self.device)
        v     = self.step_flow(obs, z, t, inference=True)
        v     = torch.nan_to_num(v, 0.0, 0.0, 0.0)
        z_mid = z + v * (dt / 2)
        v_mid = self.step_flow(obs, z_mid, t + dt / 2, inference=True)
        v_mid     = torch.nan_to_num(v_mid, 0.0, 0.0, 0.0)
        z     = z + v_mid * dt

    return z.clamp(self.act_low, self.act_high)

  @torch.inference_mode()
  def inference_step(self, observation: torch.Tensor, z : torch.Tensor = None):
    return self.generate_action(observation.to(self.device), z)

def register_policy(cls: type):
  name = cls.__name__
  if name in Policy._classes:
    raise ValueError(f"Policy {name} already registered")
  Policy._classes[name] = cls
  return cls



