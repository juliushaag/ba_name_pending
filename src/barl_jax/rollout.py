from dataclasses import dataclass, field
import math

import jax
import jax.numpy as jnp
import numpy as np

from barl_jax.policy import Policy

import gymnasium as gym


class RandomPolicy:
  """Duck-types just enough of Policy's rollout contract
  (inference_step()/action_bounds_numpy()) to drive ReplayBuffer.rollout() with
  actions sampled uniformly from the env's actual action space, instead of a
  trained policy -- used only by ReplayBuffer.random_fill() to seed the
  buffer with broad, uncorrelated coverage before training starts (SAC/TD3-
  style `random_steps`); never registered/selectable like a real Policy."""

  def __init__(self, action_space: gym.Space):
    self.action_space = action_space
    self.action_discrete = isinstance(action_space, gym.spaces.Discrete)
    if not self.action_discrete:
      low  = np.asarray(action_space.low,  dtype=np.float32)
      high = np.asarray(action_space.high, dtype=np.float32)
      self.act_scale = (high - low) / 2
      self.act_bias  = (high + low) / 2

  def inference_step(self, observation, step_info=None):
    batch = observation.shape[0]
    if self.action_discrete:
      action = np.random.randint(0, self.action_space.n, (batch, 1))
    else:
      # canonical [-1, 1] per dimension -- same convention every real actor outputs.
      action = np.random.uniform(-1.0, 1.0, (batch, *self.action_space.shape)).astype(np.float32)
    return action, {}

  def action_bounds_numpy(self):
    if self.action_discrete:
      return None, None
    return self.act_scale, self.act_bias


@dataclass
class Trajectory:
  id: int
  observations : np.ndarray
  actions : np.ndarray
  rewards : np.ndarray
  terminated : bool
  truncated : bool
  extras : dict[str, np.ndarray] = field(default_factory=dict)

  def __len__(self):
    return len(self.actions)


class ReplayBuffer():

  def __init__(self, max_transitions : int, traj_len : int):

    self.max_transitions = max_transitions
    self.traj_len = traj_len
    self._buffer = list()
    self._n_transitions = 0
    self._length_cumsum_cache = None
    self._flat_cache = None
    self._dest_cache = dict()
    # Live env/episode state carried across rollout() calls -- see rollout().
    self._rollout_state = None

  @property
  def buffer(self) -> list[Trajectory]:
    return self._buffer

  @buffer.setter
  def buffer(self, trajectories: list[Trajectory]):
    # A property purely so the on-policy `buffer.buffer = []` reset invalidates
    # the derived caches below instead of silently desyncing them.
    self._buffer = trajectories
    self._n_transitions = sum(len(t) for t in trajectories)
    self._invalidate()

  @property
  def n_transitions(self) -> int:
    return self._n_transitions

  def _invalidate(self):
    self._length_cumsum_cache = None
    self._flat_cache = None

  def save_trajectory(self, traj : Trajectory):
    """Evicts random whole chunks (never splits one) until `traj` fits under
    the transition budget. The running `_n_transitions` is what keeps this O(1)
    in the common case -- re-summing the whole buffer here cost ~5 ms per call
    once it holds a 1M-transition rollout's worth of chunks, and a rollout ends
    with one call per env."""
    while self._buffer and self._n_transitions + len(traj) > self.max_transitions:
      idx = np.random.randint(0, len(self._buffer))
      self._n_transitions -= len(self._buffer[idx])
      del self._buffer[idx]

    self._buffer.append(traj)
    self._n_transitions += len(traj)
    self._invalidate()

  def _length_cumsum(self):
    """Cached across sample() calls within the same outer iteration --
    invalidated whenever the buffer gains, loses or is reassigned a chunk."""
    if self._length_cumsum_cache is None:
      self._length_cumsum_cache = np.cumsum([len(t) for t in self._buffer])
    return self._length_cumsum_cache

  def _cat(self, key: str, arrays: list[np.ndarray], dtype) -> np.ndarray:
    """np.concatenate into a destination reused across rebuilds. Letting it
    allocate a fresh one each time cost 4x the copy itself at a full buffer
    (438 ms vs 108 ms for a 1M-transition DogRun observation array) -- faulting
    in ~900 MB of new pages, then freeing them again one rollout later."""
    rows = sum(len(a) for a in arrays)
    tail = arrays[0].shape[1:]

    buf = self._dest_cache.get(key)
    if buf is None or buf.shape[1:] != tail or buf.dtype != dtype or len(buf) < rows:
      buf = np.empty((rows, *tail), dtype=dtype)
      self._dest_cache[key] = buf

    return np.concatenate(arrays, out=buf[:rows])

  def _flat(self) -> dict:
    """The whole buffer flattened into contiguous per-field arrays, so sample()
    is one fancy-index per field instead of a Python loop building `batch_size`
    dicts. Rebuilt once per rollout (whenever a chunk is added/evicted), which
    amortizes over the thousands of grad steps that follow.

    Observations are stored once, not twice: `obs_row` indexes a transition's
    observation in the concatenated (traj_len + 1)-row array, and that row + 1
    is its next_obs. Floats are narrowed to float32 here -- MuJoCo hands back
    float64, which would otherwise double this array and be cast per batch
    anyway."""
    if self._flat_cache is None:
      trajs = self._buffer
      lengths = np.array([len(t) for t in trajs])

      observations = self._cat("obs", [t.observations for t in trajs], np.float32)
      # Each chunk contributes len(t) + 1 rows; a transition's row is its chunk's
      # start offset plus its within-chunk index.
      starts  = np.concatenate([[0], np.cumsum(lengths + 1)[:-1]])
      obs_row = np.concatenate([s + np.arange(n) for s, n in zip(starts, lengths)])

      term = np.zeros(int(lengths.sum()), dtype=bool)
      term[np.cumsum(lengths) - 1] = [t.terminated for t in trajs]

      act_chunks = [t.actions for t in trajs]
      act_dtype  = act_chunks[0].dtype if np.issubdtype(act_chunks[0].dtype, np.integer) else np.float32

      self._flat_cache = dict(
        observations=observations,
        obs_row=obs_row,
        act=self._cat("act", act_chunks, act_dtype),
        rew=self._cat("rew", [t.rewards for t in trajs], np.float32),
        term=term,
        # Intersected, not just the first chunk's: a random_fill() prefix stores
        # no extras at all, and a field only some chunks carry can't be batched.
        extras={k: self._cat(f"extras/{k}", [t.extras[k] for t in trajs], np.float32)
                for k in set.intersection(*(set(t.extras) for t in trajs))},
      )
    return self._flat_cache

  def sample(self, batch_size : int) -> dict:
    """Draws `batch_size` i.i.d. random transition indices across the whole
    buffer, with replacement (standard SAC-style minibatch). Returns
    the same keys as the trainer expects, already on the device."""
    return {k: jnp.asarray(v) for k, v in self.sample_numpy(batch_size).items()}

  def sample_numpy(self, batch_size: int) -> dict:
    """`sample()` without the device transfer. A caller drawing several batches
    at once (the multi-seed trainer, one per seed) stacks them on the host and
    crosses once, rather than once per seed per field."""
    flat = self._flat()
    idx  = np.random.randint(0, len(flat["obs_row"]), size=batch_size)
    rows = flat["obs_row"][idx]

    observations = flat["observations"]
    batch = dict(
      obs=      observations[rows],
      next_obs= observations[rows + 1],
      act=      flat["act"][idx],
      rew=      flat["rew"][idx],
      term=     flat["term"][idx].astype(np.float32),
    )
    for key, values in flat["extras"].items():
      batch[key] = values[idx]

    return batch

  def random_fill(self, env : gym.Env | gym.vector.VectorEnv, n_steps : int, seed : int = None):
    """Seeds the buffer with `n_steps` of uniformly-random actions before
    training starts -- broader, less-correlated coverage than an untrained
    (but already-biased) policy's own actions would give. Just rollout()
    under a RandomPolicy stand-in, so every chunking/truncation rule there
    applies unchanged; the resulting Trajectory extras are empty, which is
    fine for off-policy algorithms (they read obs/act/rew/next_obs only) but
    not for on-policy ones (PPO's compute_gae needs extras["value"]) -- callers
    should only use this for off-policy Policies (see Policy.on_policy)."""
    action_space = env.single_action_space if isinstance(env, gym.vector.VectorEnv) else env.action_space
    return self.rollout(RandomPolicy(action_space), env, n_steps=n_steps, seed=seed)

  def rollout(self, policy : Policy, env : gym.Env | gym.vector.VectorEnv, n_steps = 10_000, seed : int = None):
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

    # A slot can't outlive this call -- the tail flush below cuts every open
    # chunk -- so sizing to n_steps avoids handing the buffer `traj_len`-row
    # allocations that a short rollout only fills a fraction of (the saved
    # Trajectory slices are views, and would keep the whole thing alive).
    cap = min(self.traj_len, n_steps)

    def fresh_slot(o):
      slot = dict(
        observations=np.empty((cap + 1, *o.shape), dtype=o.dtype),
        actions=None,
        rewards=np.empty(cap, dtype=np.float32),
        extras=dict(),
        t=0,
      )
      slot["observations"][0] = o
      return slot

    state = self._rollout_state
    if state is None or state["env"] is not env or seed is not None:
      obs, _ = env.reset(seed=seed)
      if not is_vec:
        obs = obs[None, ...]
    else:
      obs = state["obs"]

    # Only `obs` is carried between calls -- every open chunk was cut and
    # flushed at the end of the last one, so the slots start empty regardless.
    slots = [fresh_slot(obs[i]) for i in range(n_envs)]

    # JAX transfers are already async and it has no pinned-host-buffer API, so
    # the torch version's staging buffer has no counterpart -- jnp.asarray is
    # the whole H2D path.
    def to_device(o):
      return jnp.asarray(o, jnp.float32)

    # RandomPolicy has no jitted path (and needs none); a real Policy must not
    # run inference eagerly here -- this is the hot loop. Built once per
    # rollout, so it is pinned to the weights this rollout collects under.
    inference_fn = getattr(policy, "inference_fn", None)
    act = inference_fn() if inference_fn else (lambda o, _k: policy.inference_step(o))
    key = jax.random.PRNGKey(int(np.random.randint(0, 2 ** 31 - 1)))

    # Host-side copies of the policy's action bounds. Scaling on the device
    # instead costs a second round-trip per step for an affine map on an array
    # that came back to the host one line earlier.
    act_scale_np, act_bias_np = policy.action_bounds_numpy()

    # Running scalar (not per-dimension) sums -- cheap "is anything blowing up" signal.
    rew_sum = rew_sq_sum = 0.0
    obs_sum = obs_sq_sum = 0.0
    act_sum = act_sq_sum = 0.0
    act_min, act_max = float("inf"), float("-inf")
    n_rew = n_obs = n_act = 0

    for _ in range(n_steps):

      obs64 = obs.astype(np.float64)
      obs_sum    += obs64.sum()
      obs_sq_sum += (obs64 ** 2).sum()
      n_obs      += obs64.size

      key, subkey = jax.random.split(key)
      action, step_extras = act(to_device(obs), subkey)
      action_np = np.asarray(action)  # canonical [-1, 1] -- stored in the buffer as-is
      step_extras = {k: np.asarray(v) for k, v in step_extras.items()}

      act64 = action_np.astype(np.float64)
      act_sum    += act64.sum()
      act_sq_sum += (act64 ** 2).sum()
      act_min     = min(act_min, float(act64.min()))
      act_max     = max(act_max, float(act64.max()))
      n_act      += act64.size

      # Only the env needs real-scale actions -- everything stored/returned here is canonical.
      env_action_np = action_np if act_scale_np is None else action_np * act_scale_np + act_bias_np
      # .copy(): np.asarray() on a JAX array can alias the device buffer, and
      # MuJoCo writes through the action array it is handed.
      env_action_np = np.ascontiguousarray(env_action_np, dtype=np.float32).copy()
      env_action = env_action_np[0] if not is_vec else env_action_np
      next_obs, reward, term, trunc, _ = env.step(env_action)

      if not is_vec:
        next_obs, reward, term, trunc = next_obs[None, ...], np.array([reward]), np.array([term]), np.array([trunc])

      rew64 = reward.astype(np.float64)
      rew_sum    += rew64.sum()
      rew_sq_sum += (rew64 ** 2).sum()
      n_rew      += rew64.size

      done = term | trunc
      finished_idx = []

      for i in range(n_envs):
        slot = slots[i]

        if slot["actions"] is None:
          slot["actions"] = np.empty((cap, *action_np[i].shape), dtype=action_np.dtype)

        t = slot["t"]
        slot["observations"][t + 1] = next_obs[i]
        slot["actions"][t] = action_np[i]
        slot["rewards"][t] = reward[i]

        for key, value in step_extras.items():
          value_i = value[i]
          if key not in slot["extras"]:
            slot["extras"][key] = np.empty((cap, *np.shape(value_i)), dtype=np.asarray(value_i).dtype)
          slot["extras"][key][t] = value_i

        slot["t"] = t + 1

        if done[i] or slot["t"] >= cap:
          self.save_trajectory(Trajectory(
            id=i,  # originating sub-env slot index
            observations=slot["observations"][:slot["t"] + 1],
            actions=slot["actions"][:slot["t"]],
            rewards=slot["rewards"][:slot["t"]],
            terminated=bool(term[i]),
            # A chunk-size cut mid-episode (done[i] False here) is truncated too,
            # same as a real env-reported one -- either way this chunk doesn't
            # represent an actual episode end.
            truncated=bool(trunc[i]) if done[i] else True,
            extras={k: v[:slot["t"]] for k, v in slot["extras"].items()},
          ))
          if done[i]:
            finished_idx.append(i)
          else:
            # chunk boundary mid-episode: continue the same episode in a fresh chunk
            slots[i] = fresh_slot(next_obs[i])

      obs = next_obs

      if finished_idx:
        if is_vec:
          mask = np.zeros(n_envs, dtype=bool)
          mask[finished_idx] = True
          obs, _ = env.reset(options={"reset_mask": mask})
        else:
          obs[0], _ = env.reset()

        for i in finished_idx:
          slots[i] = fresh_slot(obs[i])

    # n_steps ran out mid-chunk -- flush whatever each slot collected instead of
    # silently discarding it when `slots` goes out of scope; the episode itself
    # isn't over, so terminated=False, truncated=True, same as a chunk-size cut.
    # The episode continues in the next call, from `obs` below.
    for i in range(n_envs):
      slot = slots[i]
      if slot["t"] > 0:
        self.save_trajectory(Trajectory(
          id=i,
          observations=slot["observations"][:slot["t"] + 1],
          actions=slot["actions"][:slot["t"]],
          rewards=slot["rewards"][:slot["t"]],
          terminated=False,
          truncated=True,
          extras={k: v[:slot["t"]] for k, v in slot["extras"].items()},
        ))

    self._rollout_state = dict(env=env, obs=obs)

    def _mean_std(s, sq, n):
      if n == 0:
        return float("nan"), float("nan")
      mean = s / n
      return mean, math.sqrt(max(sq / n - mean ** 2, 0.0))

    rew_mean, rew_std = _mean_std(rew_sum, rew_sq_sum, n_rew)
    obs_mean, obs_std = _mean_std(obs_sum, obs_sq_sum, n_obs)
    act_mean, act_std = _mean_std(act_sum, act_sq_sum, n_act)

    return dict(
      avg_reward=rew_mean, reward_std=rew_std,
      obs_mean=obs_mean,   obs_std=obs_std,
      act_mean=act_mean,   act_std=act_std,
      act_min=act_min if n_act else float("nan"),
      act_max=act_max if n_act else float("nan"),
    )
