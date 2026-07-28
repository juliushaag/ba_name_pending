import math
import time
from typing import Optional

import numpy as np
import torch
from torch import nn

from abc import ABC, abstractmethod

import gymnasium as gym
import torch.optim.lr_scheduler as sched


class Actor(ABC):
  """A network that can produce actions -- pure architecture; every
  algorithm-level decision (loss, squash choice, etc.) stays on the owning
  Policy's train_step."""

  @abstractmethod
  def sample(self, s, critic: "Critic" = None) -> torch.Tensor:
    """`critic` is unused by generative actors; implicit/Q-derived actors
    (e.g. greedy-discrete) need it to compute an action at all."""
    ...

  @abstractmethod
  def create_optimizer(self) -> dict:
    ...

  @property
  def device(self):
    return next(self.parameters()).device


class Critic(ABC):
  """A network that can price (s,a) -- pure architecture; Bellman targets
  and the regression loss itself live in each Policy's train_step."""

  @abstractmethod
  def value(self, s, a) -> torch.Tensor: ...

  @abstractmethod
  def target_value(self, s, a) -> torch.Tensor:
    """Polyak-averaged target net output."""
    ...

  @abstractmethod
  def create_optimizer(self) -> dict:
    ...

  @property
  def device(self):
    return next(self.parameters()).device


class Policy(nn.Module, ABC):

  _classes = dict()

  # True for strictly on-policy algorithms (e.g. PPO) -- scripts/online_learning.py
  # clears the replay buffer after each rollout's training pass when set.
  on_policy = False

  # ------------------------------------------------------------------
  # Registry
  # ------------------------------------------------------------------

  @staticmethod
  def make(name: str, env: gym.Env, *args, **kwargs) -> "Policy":
    from barl.model_config import resolve_model_name
    name, profile = resolve_model_name(name)
    if profile is not None:
      kwargs.setdefault("profile", profile)

    if name not in Policy._classes:
      raise KeyError(f"{name} not in registered policies: {list(Policy._classes.keys())}")
    return Policy._classes[name](env, *args, **kwargs)

  @staticmethod
  def available() -> list[str]:
    return list(Policy._classes.keys())

  # ------------------------------------------------------------------
  # Construction
  # ------------------------------------------------------------------

  def __init__(self, env: gym.Env, profile: str = None, **kwargs):
    super().__init__(**kwargs)

    from barl.model_config import get_overrides

    # nn.Module's attribute dicts don't exist until super().__init__() returns.
    self._profile     = profile
    self._raw_kwargs  = kwargs
    self._overrides   = get_overrides()  # captured so a checkpoint reconstructs under them too

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
      self.register_buffer("act_scale", (self.act_high - self.act_low) / 2)
      self.register_buffer("act_bias",  (self.act_high + self.act_low) / 2)
    else:
      raise NotImplementedError(f"{action_space} type not supported ... yet")

    self.observation_shape = env.observation_space.shape
    self.observation_dim   = np.prod(self.observation_shape)

    self.register_buffer("obs_std",  torch.ones(observation_space.shape[-1]))
    self.register_buffer("obs_mean", torch.zeros(observation_space.shape[-1]))

  def resolve_component(self, slot: str, **explicit_kwargs) -> tuple[type, dict]:
    """Resolves a named slot's class + kwargs from config/models.yaml (see
    model_config.py), then layers `self._raw_kwargs` and `explicit_kwargs`
    on top (highest precedence -- for algorithm-fixed choices like
    EnsembleCritic's `pessimistic_score` that must never be tunable)."""
    from barl.model_config import resolve_slot
    cls, kwargs = resolve_slot(type(self).__name__, slot, self._profile)
    return cls, {**kwargs, **self._raw_kwargs, **explicit_kwargs}

  def resolve_params(self, **explicit_kwargs) -> dict:
    """Resolves this Policy's own top-level tunable kwargs (read directly
    off `self` in train_step, not part of any actor/critic slot) from
    config/models.yaml's `params:` block (see model_config.py), then layers
    `self._raw_kwargs` and `explicit_kwargs` on top -- same precedence as
    resolve_component()."""
    from barl.model_config import resolve_params
    params = resolve_params(type(self).__name__, self._profile)
    return {**params, **self._raw_kwargs, **explicit_kwargs}

  def build_component(self, slot: str, gate_cls: type, *args, **explicit_kwargs):
    """Resolves + constructs a named slot (actor/critic/squash/...), gated to
    `gate_cls`: the resolved class must be `gate_cls` or a subclass (e.g.
    QCriticBase admits MeanStdEnsembleCritic, MeanEnsembleCritic,
    MinEnsembleCritic, QRDistributionalCritic and C51Critic alike)."""
    cls, kwargs = self.resolve_component(slot, **explicit_kwargs)
    assert issubclass(cls, gate_cls), (
      f"config/models.yaml: {type(self).__name__}'s {slot!r} slot names {cls.__name__}, "
      f"which is not a subclass of {gate_cls.__name__} -- {type(self).__name__} can only "
      f"work with {gate_cls.__name__} (or a subclass of it) there."
    )
    return cls(*args, **kwargs)

  def on_epoch_start(self):
    self._epoch_start_time = time.perf_counter()

  def on_epoch_end(self):
    ...

  def epoch_time(self) -> float:
    """Wall-clock seconds since the last on_epoch_start() call."""
    return time.perf_counter() - self._epoch_start_time

  # ------------------------------------------------------------------
  # Checkpointing
  # ------------------------------------------------------------------

  def save(self, path: str):
    torch.save({
      "policy_class": self.__class__.__name__,
      "state_dict":   self.state_dict(),
      "profile":      self._profile,
      "overrides":    self._overrides,
    }, path)

  @staticmethod
  def load(type: str, path: str, env: gym.Env) -> "Policy":
    from barl.model_config import set_overrides
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    # The run's dotted overrides are part of the architecture, so they have to
    # be back in place before the components are constructed -- and they beat
    # the resuming run's own, which the weights would not match. A checkpoint
    # predating this key keeps whatever the resuming run set.
    if "overrides" in ckpt:
      set_overrides(ckpt["overrides"])
    policy = Policy._classes[type](env, profile=ckpt.get("profile"))
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
  # Action scaling
  # ------------------------------------------------------------------

  # Every continuous actor trains against a canonical per-dimension [-1, 1]
  # output, independent of the env's real bounds; scale/unscale_action are
  # the one-place affine remap, used only at env.step() call sites.

  def scale_action(self, raw_action: torch.Tensor) -> torch.Tensor:
    """Canonical [-1, 1]-per-dimension actor output -> this env's actual action bounds."""
    if self.action_discrete:
      return raw_action
    return raw_action * self.act_scale + self.act_bias

  def unscale_action(self, action: torch.Tensor) -> torch.Tensor:
    """Inverse of scale_action: real env-scale action -> canonical [-1, 1]."""
    if self.action_discrete:
      return action
    return (action - self.act_bias) / self.act_scale

  # ------------------------------------------------------------------
  # Misc
  # ------------------------------------------------------------------

  @property
  def device(self):
    return next(self.parameters()).device

  def num_params(self):
    return sum(p.numel() for p in self.parameters() if p.requires_grad)

  @torch.inference_mode()
  def measure_latency(self, n: int = 10) -> float:
    """Rough per-epoch regression signal, not a rigorous benchmark."""
    was_training = self.training
    self.eval()

    dummy = torch.zeros(1, *self.observation_shape, device=self.device)
    self.inference_step(dummy)  # warm-up: lazy CUDA init/kernel autotune

    start = time.perf_counter()
    for _ in range(n):
      self.inference_step(dummy)
    elapsed = time.perf_counter() - start

    if was_training:
      self.train()

    return 1000 * elapsed / n

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
  def train_step(self, batch: torch.DictType) -> tuple[dict, dict]:
    """Returns (losses, stats) -- `losses` keys must exactly match create_optimizer()'s keys."""
    ...

  @abstractmethod
  @torch.inference_mode()
  def inference_step(self, observation: torch.Tensor, step_info: Optional[dict] = None) -> tuple[torch.Tensor, dict]:
    """Returns (action, extras) where extras is a dict of per-step named tensors (e.g. entropy bounds) to log/persist."""
    ...

  # ------------------------------------------------------------------
  # Evaluation
  # ------------------------------------------------------------------

  @torch.inference_mode()
  def evaluate(self, env: gym.Env, runs=3, **inference_kwargs):
    was_training = self.training
    self.eval()
    run_data = []

    for _ in range(runs):
      obs, _ = env.reset()
      done   = False
      data   = dict(frames=[], rewards=[])

      while not done:
        action, _ = self.inference_step(torch.from_numpy(obs).float().unsqueeze(0), **inference_kwargs)
        obs, reward, term, trunc, _ = env.step(self.scale_action(action)[0].cpu().numpy())
        done = term or trunc
        data["rewards"].append(reward)
        data["frames"].append(env.render())

      data["terminated"] = term
      run_data.append(data)

    if was_training:
      self.train()

    return run_data


def register_policy(cls: type):
  name = cls.__name__
  if name in Policy._classes:
    raise ValueError(f"Policy {name} already registered")
  Policy._classes[name] = cls
  return cls



