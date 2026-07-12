from dataclasses import dataclass, field
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

  def __init__(self, max_trajectories : int, traj_len : int):

    self.max_t = max_trajectories
    self.traj_len = traj_len
    self.buffer = list()

  def save_trajectory(self, traj : Trajectory):

    if len(self.buffer) >= self.max_t:
      self.buffer = self.buffer[:self.max_t]
      idx = np.random.randint(0, len(self.buffer))
      self.buffer[idx] = traj
    else:
      self.buffer.append(traj)

  @torch.inference_mode()
  def rollout(self, policy : Policy, env : gym.Env | gym.vector.VectorEnv, n_steps = 10_000):
    """
    Drives either a plain gym.Env or a gymnasium VectorEnv (num_envs = N). If env is a
    VectorEnv, it must use autoreset_mode=AutoresetMode.DISABLED -- finished sub-envs are
    reset explicitly here via `reset_mask` so a fresh Trajectory chunk always starts from a
    real initial observation. policy.inference_step is called once per step on the full
    batch of N observations, and one Trajectory chunk is accumulated independently per slot.
    """

    is_vec = isinstance(env, gym.vector.VectorEnv)
    n_envs = env.num_envs if is_vec else 1

    obs, _ = env.reset()
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

    for _ in range(n_steps):

      action, step_extras = policy.inference_step(torch.from_numpy(obs).float())
      action_np = action.cpu().numpy()
      step_extras = {
        k: (v.cpu().numpy() if torch.is_tensor(v) else np.asarray(v))
        for k, v in step_extras.items()
      }

      env_action = action_np[0] if not is_vec else action_np
      next_obs, reward, term, trunc, _ = env.step(env_action)

      if not is_vec:
        next_obs, reward, term, trunc = next_obs[None, ...], np.array([reward]), np.array([term]), np.array([trunc])

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
            id=0,
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
