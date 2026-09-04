import math
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from barl_jax.action_chunking import ActionChunkExecutor
from barl_jax.policy import action_bounds_numpy, sanitize_action

import gymnasium as gym

if TYPE_CHECKING:
  from barl_jax.policy import Policy


class RandomPolicy:
  """Duck-types just enough of Policy's rollout contract
  (inference_step()) to drive ReplayBuffer.rollout() with
  actions sampled uniformly from the env's actual action space, instead of a
  trained policy -- used only by ReplayBuffer.random_fill() to seed the
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

  def inference_step(self, observation, step_info=None):
    batch = observation.shape[0]
    if self.action_discrete:
      plan = self._rng.randint(0, self.action_space.n,
                               (batch, self.chunk_size, 1)).astype(np.float32)
    else:
      # canonical [-1, 1] per dimension -- same convention every real actor outputs.
      plan = self._rng.uniform(-1.0, 1.0,
                               (batch, self.chunk_size, *self.action_space.shape)).astype(np.float32)
    from barl_jax.action_chunking import normalize_action_chunk
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


class ReplayBuffer():
  """A flat ring buffer of single transitions: one preallocated array per field,
  written at a rolling index, evicted oldest-first.

  Replaces the list-of-chunks this used to be, whose `sample()` read from a
  full re-concatenation of every chunk rebuilt whenever the buffer changed.
  That is O(buffer) per rollout and superlinear in the *chunk count*, and the
  chunk length was `min(traj_len, n_steps)` -- so shortening a rollout shortened
  the chunks too and the rebuild went 120 ms (1M transitions, 32-step chunks) to
  3.8 s (1M one-transition chunks). Appending here is O(rows added), which is
  what lets `rollout_steps` drop to `num_envs` without the rebuild dominating.

  `next_obs` is stored per transition rather than reconstructed from the next
  row. It costs a second observation array -- the same memory the chunk list and
  its flattened copy already used between them -- and in exchange there are no
  seams: the observation the env returned *before* its reset is the one stored,
  so a truncation boundary needs no special handling, and oldest-first eviction
  can cut anywhere.

  `term` is termination only, never truncation: a time-limit cut is not a
  terminal state and must keep its bootstrap."""

  def __init__(self, max_transitions : int):
    self.max_transitions = int(max_transitions)
    self._fields = None
    self._write  = 0
    self._size   = 0
    # Live env/episode state carried across rollout() calls -- see rollout().
    self._rollout_state = None
    # Per-sub-env-slot sums from the last rollout(), for a caller that owns a
    # subset of the slots (MultiSeedReplay.last_rollout_per_seed).
    self._rollout_per_env = None

  @property
  def n_transitions(self) -> int:
    return self._size

  def _allocate(self, row: dict) -> None:
    self._fields = {
      name: np.empty((self.max_transitions, *value.shape[1:]), dtype=value.dtype)
      for name, value in row.items()
    }

  def add(self, **row) -> None:
    """Append a batch of transitions, one row per sub-env slot. Every array must
    have the same leading length; each is stored under its keyword name and
    comes back out of `sample_numpy()` under that same name."""
    slot_ids = row.pop("slot_ids", None)
    if self._fields is None:
      self._allocate(row)
    elif row.keys() != self._fields.keys():
      raise KeyError(
        f"transition fields changed mid-run: buffer holds {sorted(self._fields)}, "
        f"got {sorted(row)} -- a ring buffer cannot backfill the rows already written")

    n = len(next(iter(row.values())))
    if n > self.max_transitions:
      raise ValueError(f"{n} rows at once exceeds the buffer's {self.max_transitions}")

    # One wrapped index array rather than two slice assignments: n is num_envs
    # per step, so this is a handful of rows and the fancy index costs less than
    # the branch would.
    idx = (self._write + np.arange(n)) % self.max_transitions
    for name, value in row.items():
      self._fields[name][idx] = value

    self._write = int((self._write + n) % self.max_transitions)
    self._size  = int(min(self._size + n, self.max_transitions))

  def sample(self, batch_size : int) -> dict:
    """Draws `batch_size` i.i.d. random transition indices across the whole
    buffer, with replacement (standard SAC-style minibatch). Returns
    the same keys as the trainer expects, already on the device."""
    return jax.device_put(self.sample_numpy(batch_size))

  def sample_numpy(self, batch_size: int) -> dict:
    """`sample()` without the device transfer. A caller drawing several batches
    at once (the multi-seed trainer, one per seed) stacks them on the host and
    crosses once, rather than once per seed per field."""
    if self._size == 0:
      raise ValueError("sample() from an empty buffer")
    idx = np.random.randint(0, self._size, size=batch_size)
    return {name: value[idx] for name, value in self._fields.items()}

  def random_fill(self, env : gym.Env | gym.vector.VectorEnv, n_steps : int, seed : int = None,
                  chunk_size: int = 1, discount: float = 1.0):
    """Seeds the buffer with `n_steps` of uniformly-random actions before
    training starts -- broader, less-correlated coverage than an untrained
    (but already-biased) policy's own actions would give. Just rollout()
    under a RandomPolicy stand-in, so every truncation rule there applies
    unchanged."""
    action_space = env.single_action_space if isinstance(env, gym.vector.VectorEnv) else env.action_space
    return self.rollout(RandomPolicy(action_space, seed=seed, chunk_size=chunk_size),
                        env, n_steps=n_steps, seed=seed, discount=discount)

  def rollout(self, policy: "Policy", env: gym.Env | gym.vector.VectorEnv, n_steps=10_000,
              seed : int = None, discount: float = None):
    """A VectorEnv must use autoreset_mode=DISABLED -- finished sub-envs are
    reset explicitly here via `reset_mask`. `seed` only seeds the first reset;
    later resets deliberately stay unseeded (gymnasium envs carry their own RNG forward).

    Episodes persist *across* calls: only the first call (or one passing an
    explicit `seed`) resets. Resetting per call would cap the visited horizon at
    `n_steps`, so a run with many envs would only ever train on the first
    `rollout_steps // num_envs` steps of an episode while eval runs the full
    `max_episode_steps` -- num_envs would trade against horizon coverage."""

    is_vec = isinstance(env, gym.vector.VectorEnv)
    n_envs = env.num_envs if is_vec else 1

    state = self._rollout_state
    if state is None or state["env"] is not env or seed is not None:
      obs, _ = env.reset(seed=seed)
      if not is_vec:
        obs = obs[None, ...]
      chunk_state = None
      pending = None
    else:
      obs = state["obs"]
      chunk_state = state.get("chunk_state")
      pending = state.get("pending")

    def to_device(o):
      return jnp.asarray(o, jnp.float32)

    inference_fn = getattr(policy, "inference_fn", None)
    act = inference_fn() if inference_fn else (lambda o, _k: policy.inference_step(o))
    key = jax.random.PRNGKey(int(np.random.randint(0, 2 ** 31 - 1)))
    chunk_size = int(getattr(policy, "chunk_size", 1) or 1)
    discount = float(getattr(getattr(policy, "critic", None), "gamma", 1.0)
                     if discount is None else discount)
    chunks = ActionChunkExecutor(act, chunk_size=chunk_size,
                                 act_dim=policy.action_dim, n_envs=n_envs)
    chunks.restore(chunk_state)

    action_space = env.single_action_space if is_vec else env.action_space
    act_scale_np, act_bias_np = action_bounds_numpy(action_space)

   
    zeros = lambda: np.zeros(n_envs, dtype=np.float64)
    acc = dict(rew_sum=zeros(), rew_sq_sum=zeros(), n_rew=zeros(),
               obs_sum=zeros(), obs_sq_sum=zeros(), n_obs=zeros(),
               act_sum=zeros(), act_sq_sum=zeros(), n_act=zeros(),
               act_min=np.full(n_envs, np.inf), act_max=np.full(n_envs, -np.inf),
               act_nonfinite=zeros(), act_clipped=zeros())

    if pending is None:
      pending = dict(
        active=np.zeros(n_envs, dtype=bool),
        obs=np.zeros((n_envs, obs.reshape(n_envs, -1).shape[1]), dtype=np.float32),
        act=np.zeros((n_envs, chunk_size * policy.action_dim), dtype=np.float32),
        rew=np.zeros(n_envs, dtype=np.float32),
        next_obs=np.zeros((n_envs, obs.reshape(n_envs, -1).shape[1]), dtype=np.float32),
        term=np.zeros(n_envs, dtype=np.float32),
        horizon=np.zeros(n_envs, dtype=np.int32),
        extras={},
      )
    else:
      pending = dict(
        active=np.array(pending["active"], copy=True),
        obs=np.array(pending["obs"], copy=True),
        act=np.array(pending["act"], copy=True),
        rew=np.array(pending["rew"], copy=True),
        next_obs=np.array(pending["next_obs"], copy=True),
        term=np.array(pending["term"], copy=True),
        horizon=np.array(pending["horizon"], copy=True),
        extras={k: np.array(v, copy=True) for k, v in pending.get("extras", {}).items()},
      )

    for _ in range(n_steps):

      obs64 = obs.astype(np.float64).reshape(n_envs, -1)
      acc["obs_sum"]    += obs64.sum(axis=1)
      acc["obs_sq_sum"] += (obs64 ** 2).sum(axis=1)
      acc["n_obs"]      += obs64.shape[1]

      key, subkey = jax.random.split(key)
      action_np, step_extras = chunks.current(obs, subkey, to_device=to_device)
      plan_np = chunks.plan().reshape(n_envs, -1).astype(np.float32)
      refilled = chunks.last_refilled()
      if refilled.any():
        obs_flat = obs.reshape(n_envs, -1).astype(np.float32)
        pending["active"][refilled] = True
        pending["obs"][refilled] = obs_flat[refilled]
        pending["act"][refilled] = plan_np[refilled]
        pending["rew"][refilled] = 0.0
        pending["term"][refilled] = 0.0
        pending["horizon"][refilled] = 0
        for name in set(pending["extras"]) - set(step_extras):
          pending["extras"][name][refilled] = 0
        for name, value in step_extras.items():
          value = np.asarray(value)
          if name not in pending["extras"]:
            pending["extras"][name] = np.zeros((n_envs, *value.shape[1:]), dtype=value.dtype)
          pending["extras"][name][refilled] = value[refilled]

      act64 = action_np.astype(np.float64).reshape(n_envs, -1)
      acc["act_sum"]    += act64.sum(axis=1)
      acc["act_sq_sum"] += (act64 ** 2).sum(axis=1)
      acc["act_min"]     = np.minimum(acc["act_min"], act64.min(axis=1))
      acc["act_max"]     = np.maximum(acc["act_max"], act64.max(axis=1))
      acc["n_act"]      += act64.shape[1]

      acc["act_nonfinite"] += (~np.isfinite(act64)).sum(axis=1)
      acc["act_clipped"]   += (np.abs(act64) > 1.0).sum(axis=1)
      action_np = sanitize_action(action_np, xp=np)

      env_action_np = action_np if act_scale_np is None else action_np * act_scale_np + act_bias_np

      env_action_np = np.ascontiguousarray(env_action_np, dtype=np.float32).copy()
      env_action = env_action_np[0] if not is_vec else env_action_np
      next_obs, reward, term, trunc, _ = env.step(env_action)

      if not is_vec:
        next_obs, reward, term, trunc = next_obs[None, ...], np.array([reward]), np.array([term]), np.array([trunc])

      rew64 = reward.astype(np.float64).reshape(n_envs)
      acc["rew_sum"]    += rew64
      acc["rew_sq_sum"] += rew64 ** 2
      acc["n_rew"]      += 1

      done = term | trunc
      pending["rew"] += (discount ** pending["horizon"]) * reward.astype(np.float32).reshape(n_envs)
      pending["next_obs"] = next_obs.reshape(n_envs, -1).astype(np.float32)
      pending["term"] = np.maximum(pending["term"], term.astype(np.float32).reshape(n_envs))
      pending["horizon"] += pending["active"].astype(np.int32)
      finished = chunks.advance(done)

      emit = np.flatnonzero(finished)
      if len(emit):
        self.add(
          slot_ids=emit,
          obs=pending["obs"][emit],
          next_obs=pending["next_obs"][emit],
          act=pending["act"][emit],
          rew=pending["rew"][emit],
          term=pending["term"][emit],
          horizon=pending["horizon"][emit],
          **{name: value[emit] for name, value in pending["extras"].items()},
        )
        pending["active"][emit] = False
        pending["rew"][emit] = 0.0
        pending["term"][emit] = 0.0
        pending["horizon"][emit] = 0

      finished_idx = np.flatnonzero(done).tolist()

      obs = next_obs

      if finished_idx:
        if is_vec:
          mask = np.zeros(n_envs, dtype=bool)
          mask[finished_idx] = True
          obs, _ = env.reset(options={"reset_mask": mask})
        else:
          obs[0], _ = env.reset()

    self._rollout_state = dict(
      env=env, obs=obs, chunk_state=chunks.snapshot(),
      pending=dict(
        active=np.array(pending["active"], copy=True),
        obs=np.array(pending["obs"], copy=True),
        act=np.array(pending["act"], copy=True),
        rew=np.array(pending["rew"], copy=True),
        next_obs=np.array(pending["next_obs"], copy=True),
        term=np.array(pending["term"], copy=True),
        horizon=np.array(pending["horizon"], copy=True),
        extras={k: np.array(v, copy=True) for k, v in pending["extras"].items()},
      ))

    # The per-slot sums stay off the return value -- a caller owning a subset of
    # the slots reads them through last_rollout_per_seed() instead.
    acc["chunk_refills"] = chunks.slot_refills().astype(np.float64)
    acc["chunk_wasted"] = chunks.slot_wasted().astype(np.float64)
    acc["chunk_size"] = np.full(n_envs, float(policy.chunk_size), np.float64)
    self._rollout_per_env = acc
    return summarize_rollout(acc)


class MultiSeedReplay(ReplayBuffer):
  """One ReplayBuffer per seed behind a single rollout loop.

  `num_seeds: 1` is not a separate path; it is the size-1 instance of the same
  routing and sampling behavior the trainer uses at larger seed counts."""

  def __init__(self, num_seeds: int, num_envs: int, max_transitions: int):
    super().__init__(max_transitions)
    self.num_seeds = num_seeds
    self.num_envs  = num_envs
    self.seeds = [ReplayBuffer(max_transitions) for _ in range(num_seeds)]

  def add(self, **row) -> None:
    slot_ids = row.pop("slot_ids", None)
    if slot_ids is None:
      for i, buffer in enumerate(self.seeds):
        block = slice(i * self.num_envs, (i + 1) * self.num_envs)
        buffer.add(**{name: value[block] for name, value in row.items()})
      return

    slot_ids = np.asarray(slot_ids)
    for i, buffer in enumerate(self.seeds):
      lo, hi = i * self.num_envs, (i + 1) * self.num_envs
      mask = (slot_ids >= lo) & (slot_ids < hi)
      if not np.any(mask):
        continue
      buffer.add(**{name: value[mask] for name, value in row.items()})

  def last_rollout_per_seed(self) -> list[dict]:
    acc = self._rollout_per_env
    return [summarize_rollout(acc, slice(i * self.num_envs, (i + 1) * self.num_envs))
            for i in range(self.num_seeds)]

  @property
  def n_transitions(self) -> int:
    return sum(buffer.n_transitions for buffer in self.seeds)

  def min_transitions(self) -> int:
    return min(buffer.n_transitions for buffer in self.seeds)

  def transitions_per_seed(self) -> list[int]:
    return [buffer.n_transitions for buffer in self.seeds]

  def sample(self, batch_size: int) -> dict:
    return self.sample_chunk(1, batch_size, leading=False)

  def sample_chunk(self, k: int, batch_size: int, *, leading: bool = True) -> dict:
    per_step = [[buffer.sample_numpy(batch_size) for buffer in self.seeds]
                for _ in range(k)]
    stack_seeds = lambda step, key: np.stack([seed[key] for seed in step])
    out = {key: np.stack([stack_seeds(step, key) for step in per_step])
           for key in per_step[0][0]}
    if not leading:
      out = {key: value[0] for key, value in out.items()}
    return jax.device_put(out)
