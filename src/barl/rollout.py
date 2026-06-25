from dataclasses import dataclass
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

  def __len__(self):
    return len(self.actions)
  

class ReplayBuffer():

  def __init__(self, max_trajectories : int):
    
    self.max_t = max_trajectories
    self.buffer = list()

  def save_trajectory(self, traj : Trajectory):

    if len(self.buffer) >= self.max_t:
      self.buffer = self.buffer[:self.max_t]
      idx = np.random.randint(0, len(self.buffer))
      self.buffer[idx] = traj
    else:
      self.buffer.append(traj)

  @torch.inference_mode()
  def rollout(self, policy : Policy, env : gym.Env, n_steps = 10_000):

    obs, _ = env.reset()

    policy.eval()

    observations = [obs]
    actions = list()
    rewards = list()

    for _ in range(n_steps):

      action = policy.inference_step(torch.from_numpy(obs).unsqueeze(0).float()).cpu().squeeze(0).numpy()

      next_obs, reward, term, trunc, _ = env.step(action)
      obs = next_obs

      observations.append(next_obs)
      actions.append(action)
      rewards.append(reward)


      if term or trunc:
        observations=np.array(observations)
        actions=np.array(actions)
        rewards=np.array(rewards)
        
        self.save_trajectory(Trajectory(
          id=0,
          observations=observations,
          actions=actions,
          rewards=rewards,
          terminated=term,
          truncated=trunc
        ))


        obs, _ = env.reset()  

        observations = [obs]
        actions = list()
        rewards = list()