from dataclasses import dataclass, field
import math
import numpy as np
import torch

from barl.policy import Policy

import gymnasium as gym


class RandomPolicy:
  """Duck-types just enough of Policy's rollout contract (eval()/
  inference_step()/scale_action()) to drive ReplayBuffer.rollout() with
  actions sampled uniformly from the env's actual action space, instead of a
  trained policy -- used only by ReplayBuffer.random_fill() to seed the
  buffer with broad, uncorrelated coverage before training starts (SAC/TD3-
  style `random_steps`); never registered/selectable like a real Policy."""

  def __init__(self, action_space: gym.Space):
    self.action_space = action_space
    self.discrete = isinstance(action_space, gym.spaces.Discrete)
    if not self.discrete:
      low  = torch.as_tensor(action_space.low,  dtype=torch.float32)
      high = torch.as_tensor(action_space.high, dtype=torch.float32)
      self.act_scale = (high - low) / 2
      self.act_bias  = (high + low) / 2

  def eval(self):
    ...

  def inference_step(self, observation: torch.Tensor, step_info=None):
    batch = observation.shape[0]
    if self.discrete:
      action = torch.randint(0, self.action_space.n, (batch, 1))
    else:
      # canonical [-1, 1] per dimension -- same convention every real actor outputs.
      action = torch.empty(batch, *self.action_space.shape).uniform_(-1.0, 1.0)
    return action, {}

  def scale_action(self, raw_action: torch.Tensor) -> torch.Tensor:
    if self.discrete:
      return raw_action
    return raw_action * self.act_scale + self.act_bias


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

      observations = np.concatenate([t.observations for t in trajs]).astype(np.float32, copy=False)
      # Each chunk contributes len(t) + 1 rows; a transition's row is its chunk's
      # start offset plus its within-chunk index.
      starts  = np.concatenate([[0], np.cumsum(lengths + 1)[:-1]])
      obs_row = np.concatenate([s + np.arange(n) for s, n in zip(starts, lengths)])

      term = np.zeros(int(lengths.sum()), dtype=bool)
      term[np.cumsum(lengths) - 1] = [t.terminated for t in trajs]

      actions = np.concatenate([t.actions for t in trajs])
      if not np.issubdtype(actions.dtype, np.integer):
        actions = actions.astype(np.float32, copy=False)

      self._flat_cache = dict(
        observations=observations,
        obs_row=obs_row,
        act=actions,
        rew=np.concatenate([t.rewards for t in trajs]).astype(np.float32, copy=False),
        term=term,
        # Intersected, not just the first chunk's: a random_fill() prefix stores
        # no extras at all, and a field only some chunks carry can't be batched.
        extras={k: np.concatenate([t.extras[k] for t in trajs]).astype(np.float32, copy=False)
                for k in set.intersection(*(set(t.extras) for t in trajs))},
      )
    return self._flat_cache

  def sample(self, batch_size : int) -> dict:
    """Draws `batch_size` i.i.d. random transition indices across the whole
    buffer, with replacement (standard SAC-style minibatch) -- unlike
    SingleStepDataset, which is an ordered, non-repeating epoch pass. Returns
    the same keys/dtypes as SingleStepDataset.__getitem__, stacked into a
    batch."""
    flat = self._flat()
    idx  = np.random.randint(0, len(flat["obs_row"]), size=batch_size)
    rows = flat["obs_row"][idx]

    observations = flat["observations"]
    batch = dict(
      obs=      torch.from_numpy(observations[rows]),
      next_obs= torch.from_numpy(observations[rows + 1]),
      act=      torch.from_numpy(flat["act"][idx]),
      rew=      torch.from_numpy(flat["rew"][idx]),
      term=     torch.from_numpy(flat["term"][idx]),
    )
    for key, values in flat["extras"].items():
      batch[key] = torch.from_numpy(values[idx])

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

  @torch.no_grad()
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

    policy.eval()

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

    # Staged through pinned memory so the per-step upload can be async: a
    # pageable .to(device) blocks the host until the copy retires, which
    # measured ~3 ms/step against ~0.01 ms here once a few runs share the GPU.
    # Reusing one staging buffer is safe only because the loop below pulls the
    # action back to the host every step, which drains the copy first.
    device = getattr(policy, "device", None)
    staging = torch.empty(obs.shape, dtype=torch.float32,
                          pin_memory=device is not None and device.type == "cuda")

    def to_device(o):
      staging.copy_(torch.from_numpy(o))
      return staging if device is None else staging.to(device, non_blocking=True)

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

      action, step_extras = policy.inference_step(to_device(obs))
      action_np = action.cpu().numpy()  # canonical [-1, 1] -- stored in the buffer as-is
      step_extras = {
        k: (v.cpu().numpy() if torch.is_tensor(v) else np.asarray(v))
        for k, v in step_extras.items()
      }

      act64 = action_np.astype(np.float64)
      act_sum    += act64.sum()
      act_sq_sum += (act64 ** 2).sum()
      act_min     = min(act_min, float(act64.min()))
      act_max     = max(act_max, float(act64.max()))
      n_act      += act64.size

      # Only the env needs real-scale actions -- everything stored/returned here is canonical.
      env_action_np = policy.scale_action(action).cpu().numpy()
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
