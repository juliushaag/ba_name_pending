import copy
from dataclasses import dataclass
import os

import gymnasium as gym
import numpy as np

import ale_py
from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F

from src.barl.policies.dqn import QPolicy
from src.barl.rollout import ReplayBuffer
from src.barl.dataset import SingleStepDataset
import wandb

from src.barl.logger import Logger

ENV = "LunarLander-v3"

torch.set_float32_matmul_precision("high")

"""
Q-function


Q^pi(s,a) = E[r(s,a) + y * E_{a'~pi}[Q(s',a')]]

So the optimality is stated by:

Q*(s, a) = E[r(s,a) + y * max E[Q*(s', a')]]

"""

@dataclass
class TrainingState:

  model : nn.Module
  opt : torch.optim.Optimizer

  grad_steps : int = 0
  env_steps : int = 0

def train_q_function(state : TrainingState, target : QPolicy, buffer : ReplayBuffer, log : Logger, epochs = 10):
  ds = SingleStepDataset.from_replay(buffer)

  Q = state.model
  optim = state.opt

  Q.train()
  train_loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)
  
  # Syncing target model
  target.load_state_dict(Q.state_dict())
  target.eval()
  
  losses = list()
  for i in range(epochs):
    for b, batch in enumerate(train_loader):
      loss = Q.train_step(batch, target)
      losses.append(loss)

      optim.zero_grad(set_to_none=True)
      loss.backward()
      optim.step()

      if b % 10 == 0:
        log.log({
          "train/loss" : loss.item()
        }, step_key="grad_steps")

        log.increment({"grad_steps" : 10})

    avg_loss = torch.stack(losses).mean()
    print(f"Epoch {i} with avg_loss {avg_loss}")
    losses = list()

@torch.inference_mode()
def evaluate_q_function(state : TrainingState, log : Logger, runs = 3):
  Q = state.model
  Q.eval()

  env = gym.make(ENV, render_mode="rgb_array")

  os.makedirs("output", exist_ok=True)

  for i in range(runs):
    obs, _ = env.reset()
    done = False
    frames = list()
    rewards = list()


    while not done:

      action = Q.inference_step(torch.from_numpy(obs).unsqueeze(0), eps_greedy=False)

      obs, reward, term, trunc, _ = env.step(action[0].item())    
      done = term or trunc
      frame = env.render()

      rewards.append(reward)

      frame = env.render()
      frames.append(frame)

    log.log({"cum_reward" : np.mean(rewards)}, step_key="env_steps")
    log.log_video("eval/video", frames, step_key="env_steps")
  
  env.close()
    

    
if __name__ == "__main__":

  log = Logger("BA-RL-QLearning", {"env_steps", "grad_steps"})


  device = "cuda"

  env = gym.make(ENV)

  model = QPolicy(env.observation_space, env.action_space, 0.99).to(device)

  optim = model.create_optimizer()
  
  buffer = ReplayBuffer(500) 

  max_iters = 100

  target = copy.deepcopy(model)
  target.eval()

  for p in target.parameters():
      p.requires_grad_(False)  

  state = TrainingState(model, optim)
      
  for i in range(max_iters):
    eps = 0.1 + 0.6 * (1 - ((i + 1) / max_iters))
    model.set_eps(eps)
    
    print("Setting eps: ", eps)

    steps = 10_000
    buffer.rollout(model, env, n_steps=steps)
    log.increment({"env_steps" : steps})


    train_q_function(state, target, buffer, log, epochs=5)  

    model.set_eps(0)
    
    evaluate_q_function(state, log, runs=1)

    print("Replay buffer episode count: ", len(buffer.buffer))

    torch.save(model.state_dict(), f"output/chkpt_q_{i}.pt")