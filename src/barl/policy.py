import math
import time
from typing import Optional

import numpy as np
import torch
from torch import nn

from abc import ABC, abstractmethod

import gymnasium as gym
import torch.optim.lr_scheduler as sched

from barl.env_stats import EnvStats
from barl.metrics import DEBUG, PRIMARY, Metrics


class Component:
  """Mixin for anything Policy.build_component() constructs. Three jobs: consume
  the run's EnvStats, which every component takes as its first positional
  argument; give the owning Policy a uniform lifecycle to fan out to -- it
  registers each slot it builds and forwards all four hooks below, so no Policy
  has to hand-roll a critic's Polyak update from its own train_step; and hold a
  `metrics` sink already namespaced to this component's slot, so writing
  `self.metrics.dist("q", pred)` lands at `critic/q/*` without the component
  ever naming itself."""

  def __init__(self, env_stats: EnvStats = None, **kwargs):
    super().__init__(**kwargs)
    self.env_stats  = env_stats if env_stats is not None else EnvStats()
    # The two every component wants, hoisted out of the dict. A component that
    # needs them *before* super().__init__() (any critic building its
    # q_function first) reads env_stats["obs_dim"]/["act_dim"] directly.
    self.obs_dim    = self.env_stats.get("obs_dim")
    self.action_dim = self.env_stats.get("act_dim")
    # Both are overwritten by build_component with the real slot; the defaults
    # only keep a component constructed outside a Policy usable.
    self.slot       = type(self).__name__.lower()
    self.metrics    = Metrics()

  def before_step(self):
    """Called once per gradient step, before the Policy's train_step."""
    ...

  def after_step(self):
    """Called once per gradient step, after every optimizer has stepped."""
    ...

  def on_epoch_start(self):
    ...

  def on_epoch_end(self):
    ...


class Actor(Component, ABC):
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


class Critic(Component, ABC):
  """A network that can price (s,a) -- pure architecture; Bellman targets
  and the regression loss itself live in each Policy's train_step."""

  @abstractmethod
  def value(self, s, a, target: bool = False) -> torch.Tensor:
    """The only accessor an algorithm needs: a scalar price for (s, a), off the
    live net or (`target=True`) the Polyak-averaged one. How that scalar is
    arrived at -- ensemble reduction, quantile mean, categorical expectation,
    pessimism -- is the implementation's business, never the caller's."""
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

  def __init__(self, env: gym.Env, profile: str = None, env_stats: EnvStats = None, **kwargs):
    super().__init__(**kwargs)

    from barl.model_config import get_overrides

    # nn.Module's attribute dicts don't exist until super().__init__() returns.
    self._profile     = profile
    self._raw_kwargs  = kwargs
    self._overrides   = get_overrides()  # captured so a checkpoint reconstructs under them too

    # A named param, not part of **kwargs -- _raw_kwargs is layered over every
    # slot's resolved config, and env_stats is not a config value.
    self.env_stats = env_stats if env_stats is not None else EnvStats.from_env(env)

    # Every slot build_component() constructs, in build order -- what the
    # lifecycle hooks below fan out over. A plain dict, so the components stay
    # registered as submodules only through the attribute each Policy assigns.
    self._components: dict[str, Component] = {}

    # `_metrics` is the root every component's slot-namespaced view writes into;
    # `metrics` is this algorithm's own view, so a Policy writing
    # `self.metrics.scalar("clip_frac", x)` lands at `ppo/clip_frac`.
    self._metrics = Metrics()
    self.metrics  = self._metrics.child(self.algo_name())
    self._batch_metrics   = self._metrics.child("batch")
    self._optimizer_slots = None

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
    on top (highest precedence -- for algorithm-fixed choices that must never
    be tunable)."""
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

  def build_component(self, slot: str, gate_cls: type, *args,
                      env_stats: EnvStats = None, **explicit_kwargs):
    """Resolves + constructs a named slot (actor/critic/squash/...), gated to
    `gate_cls`: the resolved class must be `gate_cls` or a subclass (e.g.
    QCriticBase admits MeanStdEnsembleCritic, MeanEnsembleCritic,
    MinEnsembleCritic, QRDistributionalCritic and C51Critic alike).
    `env_stats` overrides the run's own for a slot another component wraps and
    feeds a transformed input (NC-LQL's level-conditioned Q sees a widened
    action) -- pass an `EnvStats.derive()` of the run's, never an unrelated one."""
    cls, kwargs = self.resolve_component(slot, **explicit_kwargs)
    assert issubclass(cls, gate_cls), (
      f"config/models.yaml: {type(self).__name__}'s {slot!r} slot names {cls.__name__}, "
      f"which is not a subclass of {gate_cls.__name__} -- {type(self).__name__} can only "
      f"work with {gate_cls.__name__} (or a subclass of it) there."
    )
    component = cls(env_stats if env_stats is not None else self.env_stats, *args, **kwargs)
    component.slot    = slot
    component.metrics = self._metrics.child(slot)
    self._components[slot] = component
    return component

  # ------------------------------------------------------------------
  # Metrics
  # ------------------------------------------------------------------

  @classmethod
  def algo_name(cls) -> str:
    """This algorithm's logging namespace -- `SACPolicy` -> `sac`, `AMDP` -> `amdp`."""
    name = cls.__name__.lower()
    return name[:-len("policy")] if name.endswith("policy") and len(name) > len("policy") else name

  def optimizer_slots(self) -> dict[str, str]:
    """Optimizer key -> owning component slot, by asking each slot which keys
    it creates. Keys owned by no component (SDAC's `explore`) map to this
    algorithm's own namespace. The throwaway optimizers are cheap -- torch
    allocates no state until the first step()."""
    if self._optimizer_slots is None:
      owners = {}
      for slot, component in self._components.items():
        create = getattr(component, "create_optimizer", None)  # a Squash has none
        for key in (create() if create else {}):
          owners[key] = slot
      self._optimizer_slots = owners
    return self._optimizer_slots

  def metrics_key(self, optimizer_key: str, field: str) -> str:
    """`critic/loss` when the slot owns one optimizer key, `actor/loss/flow`
    when it owns several -- so the common case stays flat."""
    owners = self.optimizer_slots()
    if optimizer_key not in owners:  # no component owns it -- always name it
      return f"{self.algo_name()}/{field}/{optimizer_key}"
    slot   = owners[optimizer_key]
    shared = sum(1 for o in owners.values() if o == slot) > 1
    return f"{slot}/{field}/{optimizer_key}" if shared else f"{slot}/{field}"

  def record_optimizer(self, optimizer_key: str, loss, lr, grad_norm, params=None):
    """The `opt` family. It lives here rather than on Metrics because the
    optimizer-key -> slot map does, so a loss lands under the component that
    owns it: `critic/loss`, `critic/lr`, `critic/grad_norm`."""
    metrics = self._metrics
    metrics.scalar(self.metrics_key(optimizer_key, "loss"), loss.detach())
    metrics.scalar(self.metrics_key(optimizer_key, "lr"), lr)
    metrics.scalar(self.metrics_key(optimizer_key, "grad_norm"), grad_norm)

    if params is None or not metrics.active(DEBUG):
      return
    grads = [p.grad.detach().float() for p in params if p.grad is not None]
    if not grads:
      return
    # Reduced per-tensor -- a flat cat() of every gradient is a full copy of the
    # model, which is exactly what we don't want on a diagnostic path.
    total   = sum(g.numel() for g in grads)
    zeros   = sum((g == 0).sum() for g in grads)
    nonfin  = sum((~torch.isfinite(g)).sum() for g in grads)
    grad_max = torch.stack([g.abs().max() for g in grads]).max()
    metrics.scalar(self.metrics_key(optimizer_key, "grad_max"), grad_max, tier=DEBUG)
    metrics.scalar(self.metrics_key(optimizer_key, "grad_frac_zero"), zeros / total, tier=DEBUG)
    metrics.scalar(self.metrics_key(optimizer_key, "grad_frac_nonfinite"), nonfin / total, tier=DEBUG)

  def record_batch(self, batch: dict):
    """The `batch/*` family -- what every algorithm trains against, recorded
    once here rather than restated by each train_step."""
    self._batch_metrics.dist("reward", batch["rew"])
    self._batch_metrics.dist("action", batch["act"])
    
  # ------------------------------------------------------------------
  # Component lifecycle
  # ------------------------------------------------------------------

  # Fanned out over every slot build_component() made, so a component owns its
  # own per-step/per-epoch bookkeeping (a critic's Polyak update, a Shadow
  # resync) instead of each Policy remembering to drive it from train_step.

  @property
  def components(self) -> dict[str, "Component"]:
    return self._components

  def before_step(self, tiers: tuple = ()):
    """`tiers` arms the metric sink for this step (see barl.metrics) -- an
    unarmed family is a no-op, so a step that won't be logged never pays for
    its reductions."""
    self._metrics.arm(*tiers)
    for component in self._components.values():
      component.before_step()

  def after_step(self):
    for component in self._components.values():
      component.after_step()

  def drain_metrics(self) -> tuple[dict, dict]:
    """(primary, debug) for the step just finished, each already on the CPU."""
    return self._metrics.drain()

  def on_epoch_start(self):
    self._epoch_start_time = time.perf_counter()
    for component in self._components.values():
      component.on_epoch_start()

  def on_epoch_end(self):
    for component in self._components.values():
      component.on_epoch_end()

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
  def load(type: str, path: str, env: gym.Env, env_stats: EnvStats = None) -> "Policy":
    from barl.model_config import set_overrides
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    # The run's dotted overrides are part of the architecture, so they have to
    # be back in place before the components are constructed -- and they beat
    # the resuming run's own, which the weights would not match. A checkpoint
    # predating this key keeps whatever the resuming run set.
    if "overrides" in ckpt:
      set_overrides(ckpt["overrides"])
    # env_stats is a function of the env, so it is rebuilt rather than persisted.
    policy = Policy._classes[type](env, profile=ckpt.get("profile"), env_stats=env_stats)
    policy.load_state_dict(ckpt["state_dict"])
    return policy

  @staticmethod
  def load_any(path: str, env: gym.Env, env_stats: EnvStats = None) -> "Policy":
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    return Policy.load(ckpt["policy_class"], path, env, env_stats)

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

  @torch.no_grad()
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
  def train_step(self, batch: torch.DictType) -> dict:
    """Returns `losses`, whose keys must exactly match create_optimizer()'s.
    Diagnostics go to `self.metrics` / a component's own `self.metrics`, not a
    return value -- the training loop drains them once per step."""
    ...

  @abstractmethod
  @torch.no_grad()
  def inference_step(self, observation: torch.Tensor, step_info: Optional[dict] = None) -> tuple[torch.Tensor, dict]:
    """Returns (action, extras) where extras is a dict of per-step named tensors (e.g. entropy bounds) to log/persist."""
    ...

  # ------------------------------------------------------------------
  # Evaluation
  # ------------------------------------------------------------------

  @torch.no_grad()
  def evaluate(self, env: gym.Env, runs=3, videos: int = None, render_env: gym.Env = None,
               **inference_kwargs):
    """`runs` episodes total: `runs - videos` unrendered on `env` plus `videos`
    rendered on `render_env` (defaulting to `env`), all counting towards the
    reward statistics. `videos=None` renders every run.

    Both may be a single gym.Env or a VectorEnv, and being able to hand this a
    VectorEnv is the point: a single env spends ~90% of eval blocked on batch-1
    device round-trips rather than on the physics. The rendered episodes stay on
    their own env because env.render() costs ~140x the env step and a VectorEnv
    can only render all of its sub-envs at once."""
    was_training = self.training
    self.eval()

    n_videos = runs if videos is None else min(videos, runs)
    run_data  = self._run_episodes(env, runs - n_videos, render=False, **inference_kwargs)
    run_data += self._run_episodes(render_env if render_env is not None else env,
                                   n_videos, render=True, **inference_kwargs)

    if was_training:
      self.train()

    return run_data

  @torch.no_grad()
  def _run_episodes(self, env: gym.Env, n: int, render: bool, **inference_kwargs) -> list[dict]:
    """`n` complete episodes, in parallel waves of `num_envs` if `env` is a
    VectorEnv. A slot that finishes before the rest of its wave is reset and
    stops recording -- autoreset_mode=DISABLED asserts rather than letting a
    finished slot be stepped again -- so a wave costs its longest episode."""
    if n <= 0:
      return []

    is_vec = isinstance(env, gym.vector.VectorEnv)
    n_envs = env.num_envs if is_vec else 1

    def reset(**kwargs):
      obs, _ = env.reset(**kwargs)
      return obs if is_vec else obs[None, ...]

    # Same reason as ReplayBuffer.rollout(): scaling on the device costs a
    # second D2H round-trip per step, a quarter of this loop's wall clock, for
    # an affine map on a tensor that is already back on the host.
    act_scale_np = act_bias_np = None
    if not self.action_discrete:
      act_scale_np = self.act_scale.detach().cpu().numpy()
      act_bias_np  = self.act_bias.detach().cpu().numpy()

    out = []
    while len(out) < n:
      wave = min(n_envs, n - len(out))
      obs  = reset()
      data = [dict(frames=[], rewards=[], terminated=False) for _ in range(n_envs)]
      live = np.zeros(n_envs, dtype=bool)
      live[:wave] = True

      while live.any():
        action, _ = self.inference_step(torch.from_numpy(obs).float(), **inference_kwargs)
        action_np  = action.cpu().numpy()
        env_action = action_np if act_scale_np is None else action_np * act_scale_np + act_bias_np
        obs, reward, term, trunc, _ = env.step(env_action if is_vec else env_action[0])

        if not is_vec:
          obs, reward, term, trunc = obs[None, ...], np.array([reward]), np.array([term]), np.array([trunc])

        frames = (env.call("render") if is_vec else [env.render()]) if render else None

        done = term | trunc
        for i in np.flatnonzero(live):
          data[i]["rewards"].append(reward[i])
          if render:
            data[i]["frames"].append(frames[i])
          if done[i]:
            data[i]["terminated"] = bool(term[i])
            live[i] = False

        if live.any() and done.any():
          obs = reset(options={"reset_mask": done}) if is_vec else reset()

      out += data[:wave]

    return out


def register_policy(cls: type):
  name = cls.__name__
  if name in Policy._classes:
    raise ValueError(f"Policy {name} already registered")
  Policy._classes[name] = cls
  return cls



