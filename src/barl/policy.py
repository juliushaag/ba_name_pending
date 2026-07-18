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


class HParamMixin:
  """
  Shared HParam declaration/resolution machinery. Mixed into Policy, Actor and
  Critic so hyperparameters can be declared on whichever of the three actually
  owns them (e.g. discount/pessimism on a Critic, distillation weight on an
  Actor) rather than forcing everything onto the outer Policy.
  """

  @classmethod
  def _hparam_descriptors(cls) -> dict[str, HParam]:
    """
    Collect all HParam class attributes walking the MRO, cached per-class in
    _hparams so later instances can still find the descriptors even after
    resolve_hparams overwrites the class attributes with resolved values.
    """
    cached = cls.__dict__.get("_hparams")
    if cached is not None:
      return cached

    out = {}
    for klass in reversed(cls.__mro__):
      for k, v in vars(klass).items():

        if isinstance(v, HParam):
          out[k] = v

    cls._hparams = out
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

    unknown = set(kwargs) - set(descriptors)
    if unknown:
      raise TypeError(f"{cls.__name__} got unexpected hparams: {unknown}")

    return resolved

  def apply_hparams(self, kwargs: dict) -> dict:
    """Resolve kwargs against this class's HParam descriptors and set each
    as an instance attribute. Shared by Policy/Actor/Critic __init__s so the
    resolve+setattr boilerplate lives in one place."""
    self.hparams: dict = self.resolve_hparams(kwargs)
    for name, value in self.hparams.items():
      setattr(self, name, value)
    return self.hparams

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


class Actor(ABC):
  """
  A network that can produce actions. Pure architecture -- no algorithm
  logic lives here: every Policy owns its full policy-improvement step
  directly in train_step (Bellman targets, advantage weighting, clipped
  surrogates, adjoint matching, ...), calling into whatever this actor
  exposes (sample(), and for bespoke actors, whatever else their algorithm
  needs, e.g. step_flow()). Concrete actors are configured by architecture
  only (hidden_dim, layers, ...), never by algorithm hyperparameters --
  those are HParams on the owning Policy.
  """

  @abstractmethod
  def sample(self, s, critic: "Critic" = None) -> torch.Tensor:
    """
    `critic` is optional and unused by generative actors (Gaussian/diffusion/
    flow). Implicit/Q-derived actors (greedy-discrete, Langevin) need it to
    compute the action at all -- they have no generative net of their own.
    """
    ...

  @abstractmethod
  def create_optimizer(self) -> dict:
    ...

  def on_epoch_end(self):
    """Hook for actors with a periodic (non-Polyak) reference snapshot, e.g.
    a hard-copied trust-region policy. No-op by default."""
    ...

  @property
  def device(self):
    return next(self.parameters()).device


class MaxEntActor(Actor):
  """
  Actor subtype that carries its own entropy dual and can price a soft-Q
  Bellman target. `sample_with_entropy` is only meaningful for actors that
  are actually entropy-regularized (e.g. FlowSDEActor), so it lives here
  rather than as a zero-bonus default on every Actor -- policies with a
  soft-Q Bellman target (e.g. DAMPolicy) call it directly when building
  that target, instead of forcing every actor to carry an alpha dual.
  """

  @abstractmethod
  def sample_with_entropy(self, s) -> tuple[torch.Tensor, torch.Tensor]:
    """(action, alpha-scaled soft-entropy bonus) for the soft-Q Bellman
    target (reward + gamma * (target_q + bonus))."""
    ...


class Critic(ABC):
  """
  A network that can price (s,a). Pure architecture -- no algorithm logic:
  Bellman targets, discounting, and the regression loss itself all live in
  each Policy's train_step. Concrete critics are configured by architecture
  only (hidden_dim, layers, ensemble size, distributional-or-not, ...).
  """

  @abstractmethod
  def value(self, s, a) -> torch.Tensor: ...

  @abstractmethod
  def score(self, s, a) -> torch.Tensor:
    """∇_a Q(s,a) score of the EBM modeled by the critic"""
    ...

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


class Policy(nn.Module, HParamMixin, ABC):

  _classes = dict()

  # True for strictly on-policy algorithms (e.g. PPO): scripts/online_learning.py
  # checks this to clear the replay buffer after each rollout's training pass
  # instead of letting old-policy trajectories linger across rollouts the way
  # every off-policy critic in this framework relies on.
  on_policy = False

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

    self.apply_hparams(kwargs)

  def on_epoch_start(self):
    ...

  
  def on_epoch_end(self):
    ...

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
  def train_step(self, batch: torch.DictType) -> tuple[dict, dict]:
    """
    Computes every loss term for this batch. Returns (losses, stats):
    `losses` keyed exactly matching create_optimizer()'s keys (the training
    script zero_grads/backwards/steps each one), `stats` a separate dict of
    whatever else is worth logging (already detached).
    """
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
        obs, reward, term, trunc, _ = env.step(action[0].cpu().numpy())
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



