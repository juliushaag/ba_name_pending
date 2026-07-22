from dataclasses import dataclass, field
import math
import numpy as np
import torch

from barl.policy import Policy

import gymnasium as gym


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
    self.buffer = list()

  def save_trajectory(self, traj : Trajectory):
    """Evicts random whole chunks (never splits one) until `traj` fits under
    the transition budget. Recomputes the running total each call rather
    than an incremental counter, since on-policy callers reset the buffer by
    reassigning `buffer.buffer = []` directly, which would desync a cache."""
    total = sum(len(t) for t in self.buffer)
    while self.buffer and total + len(traj) > self.max_transitions:
      idx = np.random.randint(0, len(self.buffer))
      total -= len(self.buffer[idx])
      del self.buffer[idx]

    self.buffer.append(traj)

  @torch.inference_mode()
  def rollout(self, policy : Policy, env : gym.Env | gym.vector.VectorEnv, n_steps = 10_000, seed : int = None):
    """A VectorEnv must use autoreset_mode=DISABLED -- finished sub-envs are
    reset explicitly here via `reset_mask`. `seed` only seeds the first reset;
    later resets deliberately stay unseeded (gymnasium envs carry their own RNG forward)."""

    is_vec = isinstance(env, gym.vector.VectorEnv)
    n_envs = env.num_envs if is_vec else 1

    obs, _ = env.reset(seed=seed)
    if not is_vec:
      obs = obs[None, ...]

    policy.eval()

    def fresh_slot(o):
      slot = dict(
        observations=np.empty((self.traj_len + 1, *o.shape), dtype=o.dtype),
        actions=None,
        rewards=np.empty(self.traj_len, dtype=np.float32),
        extras=dict(),
        t=0,
      )
      slot["observations"][0] = o
      return slot

    slots = [fresh_slot(obs[i]) for i in range(n_envs)]

    # Running scalar (not per-dimension) sums -- cheap "is anything blowing up" signal.
    rew_sum = rew_sq_sum = 0.0
    obs_sum = obs_sq_sum = 0.0
    act_sum = act_sq_sum = 0.0
    n_rew = n_obs = n_act = 0

    for _ in range(n_steps):

      obs64 = obs.astype(np.float64)
      obs_sum    += obs64.sum()
      obs_sq_sum += (obs64 ** 2).sum()
      n_obs      += obs64.size

      action, step_extras = policy.inference_step(torch.from_numpy(obs).float())
      action_np = action.cpu().numpy()  # canonical [-1, 1] -- stored in the buffer as-is
      step_extras = {
        k: (v.cpu().numpy() if torch.is_tensor(v) else np.asarray(v))
        for k, v in step_extras.items()
      }

      act64 = action_np.astype(np.float64)
      act_sum    += act64.sum()
      act_sq_sum += (act64 ** 2).sum()
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
          slot["actions"] = np.empty((self.traj_len, *action_np[i].shape), dtype=action_np.dtype)

        t = slot["t"]
        slot["observations"][t + 1] = next_obs[i]
        slot["actions"][t] = action_np[i]
        slot["rewards"][t] = reward[i]

        for key, value in step_extras.items():
          value_i = value[i]
          if key not in slot["extras"]:
            slot["extras"][key] = np.empty((self.traj_len, *np.shape(value_i)), dtype=np.asarray(value_i).dtype)
          slot["extras"][key][t] = value_i

        slot["t"] = t + 1

        if done[i] or slot["t"] >= self.traj_len:
          self.save_trajectory(Trajectory(
            id=i,  # originating sub-env slot index
            observations=slot["observations"][:slot["t"] + 1],
            actions=slot["actions"][:slot["t"]],
            rewards=slot["rewards"][:slot["t"]],
            terminated=bool(term[i]),
            truncated=bool(trunc[i]),
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
    )
