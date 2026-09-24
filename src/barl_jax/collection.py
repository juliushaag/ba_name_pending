import math
from typing import Callable, Protocol

import gymnasium as gym
import jax
import numpy as np

from barl_jax.action_chunking import ActionChunkExecutor, normalize_action_chunk, step_env
from barl_jax.components import action_bounds_numpy


class ActionProvider(Protocol):
  action_dim: int
  chunk_size: int

  def inference_fn(self) -> Callable:
    ...


class TransitionSink(Protocol):
  def add(self, **transition) -> None:
    ...


class RandomPolicy:
  """Duck-types just enough of Policy's rollout contract
  (inference_step()) to drive Collector.collect() with
  actions sampled uniformly from the env's actual action space, instead of a
  trained policy -- used only by Collector.random_fill() to seed the
  buffer with broad, uncorrelated coverage before training starts (SAC/TD3-
  style `random_steps`); never registered/selectable like a real Policy."""

  def __init__(self, action_space: gym.Space, seed: int = None, chunk_size: int = 1):
    self.action_space = action_space
    self.action_discrete = isinstance(action_space, gym.spaces.Discrete)
    self.chunk_size = int(chunk_size)
    self.action_dim = int(action_space.n) if self.action_discrete else int(np.prod(action_space.shape))
    self._rng = np.random.RandomState(seed)
    if not self.action_discrete:
      low  = np.asarray(action_space.low,  dtype=np.float32)
      high = np.asarray(action_space.high, dtype=np.float32)
      self.act_scale = (high - low) / 2
      self.act_bias  = (high + low) / 2

  def inference_fn(self):
    return lambda observation, key: self._sample(observation)

  def _sample(self, observation):
    batch = observation.shape[0]
    if self.action_discrete:
      plan = self._rng.randint(0, self.action_space.n,
                               (batch, self.chunk_size, 1)).astype(np.float32)
    else:
      # canonical [-1, 1] per dimension -- same convention every real actor outputs.
      plan = self._rng.uniform(-1.0, 1.0,
                               (batch, self.chunk_size, *self.action_space.shape)).astype(np.float32)
    return normalize_action_chunk(
      plan, chunk_size=self.chunk_size, act_dim=self.action_dim, xp=np), {}

def summarize_rollout(acc: dict, slots: slice = slice(None)) -> dict:
  """Per-sub-env-slot sums -> the reported rollout statistics, over `slots`.

  The same function serves the whole rollout and one seed's block of slots, so
  a per-seed number and the blended one can never be computed differently."""
  def mean_std(key):
    n = float(acc[f"n_{key}"][slots].sum())
    if n == 0:
      return float("nan"), float("nan")
    mean = float(acc[f"{key}_sum"][slots].sum()) / n
    var  = float(acc[f"{key}_sq_sum"][slots].sum()) / n - mean ** 2
    return mean, math.sqrt(max(var, 0.0))

  rew_mean, rew_std = mean_std("rew")
  obs_mean, obs_std = mean_std("obs")
  act_mean, act_std = mean_std("act")
  n_act = float(acc["n_act"][slots].sum())
  seen  = n_act > 0
  out = dict(
    avg_reward=rew_mean, reward_std=rew_std,
    obs_mean=obs_mean,   obs_std=obs_std,
    act_mean=act_mean,   act_std=act_std,
    act_min=float(acc["act_min"][slots].min()) if seen else float("nan"),
    act_max=float(acc["act_max"][slots].max()) if seen else float("nan"),
    # Fractions of action components the failsafe had to repair. Both are 0.0 in
    # any healthy run; act_nonfinite > 0 means the actor produced NaN/inf and is
    # a bug, not a tuning problem.
    act_nonfinite=float(acc["act_nonfinite"][slots].sum()) / n_act if seen else 0.0,
    act_clipped=float(acc["act_clipped"][slots].sum()) / n_act if seen else 0.0,
  )
  if "chunk_refills" in acc:
    refills = float(acc["chunk_refills"][slots].sum())
    wasted = float(acc["chunk_wasted"][slots].sum())
    chunk_size = float(acc["chunk_size"][slots].max()) if refills else float("nan")
    out.update(
      chunk_size=chunk_size,
      chunk_refills=refills,
      chunk_refill_rate=refills / max(1.0, float(acc["n_rew"][slots].sum())),
      chunk_waste_frac=wasted / max(1.0, refills * max(chunk_size, 1.0)),
    )
  return out


class PendingChunks:
  """Per slot, the chunk being executed: its start observation, plan, discounted
  return so far, and horizon -- held until the chunk finishes and is emitted to
  replay as one transition."""

  ARRAYS = ("active", "obs", "act", "chunk_return", "next_obs", "term", "horizon")

  def __init__(self, n_envs: int, obs_dim: int, plan_dim: int):
    self.active = np.zeros(n_envs, dtype=bool)
    self.obs = np.zeros((n_envs, obs_dim), dtype=np.float32)
    self.act = np.zeros((n_envs, plan_dim), dtype=np.float32)
    self.chunk_return = np.zeros(n_envs, dtype=np.float32)
    self.next_obs = np.zeros((n_envs, obs_dim), dtype=np.float32)
    self.term = np.zeros(n_envs, dtype=np.float32)
    self.horizon = np.zeros(n_envs, dtype=np.int32)
    self.extras: dict = {}

  def copy(self) -> "PendingChunks":
    other = object.__new__(PendingChunks)
    for name in self.ARRAYS:
      setattr(other, name, np.array(getattr(self, name), copy=True))
    other.extras = {k: np.array(v, copy=True) for k, v in self.extras.items()}
    return other

  def start(self, slots, obs, plan, step_extras: dict) -> None:
    """Begin a new chunk on the boolean `slots` that just refilled their plan."""
    self.active[slots] = True
    self.obs[slots] = obs[slots]
    self.act[slots] = plan[slots]
    self.chunk_return[slots] = 0.0
    self.term[slots] = 0.0
    self.horizon[slots] = 0
    for name in set(self.extras) - set(step_extras):
      self.extras[name][slots] = 0
    for name, value in step_extras.items():
      value = np.asarray(value)
      if name not in self.extras:
        self.extras[name] = np.zeros((len(self.active), *value.shape[1:]), dtype=value.dtype)
      self.extras[name][slots] = value[slots]

  def observe(self, reward, next_obs, term, discount: float) -> None:
    """Fold one env step into every slot's running chunk."""
    self.chunk_return += (discount ** self.horizon) * reward
    self.next_obs = next_obs
    self.term = np.maximum(self.term, term)
    self.horizon += self.active.astype(np.int32)

  def emit(self, finished, sink: TransitionSink) -> None:
    """Hand the `finished` slots' chunks to `sink` and clear them."""
    emit = np.flatnonzero(finished)
    if not len(emit):
      return
    sink.add(
      slot_ids=emit, obs=self.obs[emit], next_obs=self.next_obs[emit], act=self.act[emit],
      chunk_return=self.chunk_return[emit], term=self.term[emit], horizon=self.horizon[emit],
      **{name: value[emit] for name, value in self.extras.items()})
    self.active[emit] = False
    self.chunk_return[emit] = 0.0
    self.term[emit] = 0.0
    self.horizon[emit] = 0


class Collector:
  """Owns environment/chunk state and emits transitions into an injected sink."""

  def __init__(self, seed=None):
    self.state = None
    self.statistics = None
    self.rng = np.random.default_rng(seed)

  def random_fill(self, env : gym.Env | gym.vector.VectorEnv, n_steps : int, seed : int = None,
                  chunk_size: int = 1, discount: float = 1.0, *, sink: TransitionSink):
    """Seeds the buffer with `n_steps` of uniformly-random actions before
    training starts: broader, less-correlated coverage than an untrained policy's
    own actions. Just `collect()` under a `RandomPolicy`, so every truncation
    rule there applies unchanged."""
    action_space = env.single_action_space if isinstance(env, gym.vector.VectorEnv) else env.action_space
    return self.collect(RandomPolicy(action_space, seed=seed, chunk_size=chunk_size),
                        env, n_steps=n_steps, seed=seed, discount=discount, sink=sink)

  def per_seed(self, num_seeds, num_envs):
    return [summarize_rollout(self.statistics, slice(i * num_envs, (i + 1) * num_envs))
            for i in range(num_seeds)]

  def collect(self, policy: ActionProvider, env: gym.Env | gym.vector.VectorEnv, n_steps=10_000,
              seed : int = None, discount: float = 1.0, *, sink: TransitionSink):
    """A VectorEnv must use autoreset_mode=DISABLED -- finished sub-envs are
    reset explicitly here via `reset_mask`. `seed` only seeds the first reset;
    later resets deliberately stay unseeded (gymnasium envs carry their own RNG forward).

    Episodes persist *across* calls: only the first call (or one passing an
    explicit `seed`) resets. Resetting per call would cap the visited horizon at
    `n_steps`, so a run with many envs would only ever train on the first
    `rollout_steps // num_envs` steps of an episode while eval runs the full
    `max_episode_steps`."""
    is_vec = isinstance(env, gym.vector.VectorEnv)
    n_envs = env.num_envs if is_vec else 1

    state = self.state
    if state is None or state["env"] is not env or seed is not None:
      obs, _ = env.reset(seed=seed)
      if not is_vec:
        obs = obs[None, ...]
      chunk_state, pending = None, None
    else:
      obs, chunk_state, pending = state["obs"], state.get("chunk_state"), state.get("pending")

    act = policy.inference_fn()
    key = jax.random.PRNGKey(int(self.rng.integers(0, 2 ** 31 - 1)))
    chunk_size = int(policy.chunk_size)
    discount = float(discount)
    chunks = ActionChunkExecutor(act, chunk_size=chunk_size,
                                 act_dim=policy.action_dim, n_envs=n_envs)
    chunks.restore(chunk_state)

    action_space = env.single_action_space if is_vec else env.action_space
    act_scale_np, act_bias_np = action_bounds_numpy(action_space)

    pending = (PendingChunks(n_envs, obs.reshape(n_envs, -1).shape[1], chunk_size * policy.action_dim)
               if pending is None else pending.copy())
    zeros = lambda: np.zeros(n_envs, dtype=np.float64)
    acc = dict(rew_sum=zeros(), rew_sq_sum=zeros(), n_rew=zeros(),
               obs_sum=zeros(), obs_sq_sum=zeros(), n_obs=zeros(),
               act_sum=zeros(), act_sq_sum=zeros(), n_act=zeros(),
               act_min=np.full(n_envs, np.inf), act_max=np.full(n_envs, -np.inf),
               act_nonfinite=zeros(), act_clipped=zeros())

    for _ in range(n_steps):
      obs64 = obs.astype(np.float64).reshape(n_envs, -1)
      acc["obs_sum"]    += obs64.sum(axis=1)
      acc["obs_sq_sum"] += (obs64 ** 2).sum(axis=1)
      acc["n_obs"]      += obs64.shape[1]

      key, subkey = jax.random.split(key)
      action_np, step_extras = chunks.current(obs, subkey)
      refilled = chunks.last_refilled()
      if refilled.any():
        pending.start(refilled, obs.reshape(n_envs, -1).astype(np.float32),
                      chunks.plan().reshape(n_envs, -1).astype(np.float32), step_extras)

      act64 = action_np.astype(np.float64).reshape(n_envs, -1)
      acc["act_sum"]    += act64.sum(axis=1)
      acc["act_sq_sum"] += (act64 ** 2).sum(axis=1)
      acc["act_min"]     = np.minimum(acc["act_min"], act64.min(axis=1))
      acc["act_max"]     = np.maximum(acc["act_max"], act64.max(axis=1))
      acc["n_act"]      += act64.shape[1]
      acc["act_nonfinite"] += (~np.isfinite(act64)).sum(axis=1)
      acc["act_clipped"]   += (np.abs(act64) > 1.0).sum(axis=1)

      next_obs, reward, term, trunc, _ = step_env(env, action_np, act_scale_np, act_bias_np,
                                                  is_vec=is_vec)
      rew64 = reward.astype(np.float64).reshape(n_envs)
      acc["rew_sum"]    += rew64
      acc["rew_sq_sum"] += rew64 ** 2
      acc["n_rew"]      += 1

      done = np.asarray(term | trunc, dtype=bool)
      pending.observe(reward.astype(np.float32).reshape(n_envs),
                      next_obs.reshape(n_envs, -1).astype(np.float32),
                      term.astype(np.float32).reshape(n_envs), discount)
      pending.emit(chunks.advance(done), sink)

      obs = next_obs
      if done.any():
        if is_vec:
          obs, _ = env.reset(options={"reset_mask": done})
        else:
          obs[0], _ = env.reset()

    self.state = dict(env=env, obs=obs, chunk_state=chunks.snapshot(), pending=pending.copy())

    acc["chunk_refills"] = chunks.slot_refills().astype(np.float64)
    acc["chunk_wasted"] = chunks.slot_wasted().astype(np.float64)
    acc["chunk_size"] = np.full(n_envs, float(policy.chunk_size), np.float64)
    self.statistics = acc
    return summarize_rollout(acc)
