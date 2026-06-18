import os
os.environ.setdefault("MUJOCO_GL", "egl")  

import copy
from dataclasses import dataclass

import gymnasium as gym
import minari
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

from src.barl.policies.iql import IQLPolicy
from src.barl.policies.flow import FlowPolicy
from src.barl.policies.direct import DirectPolicy

from src.barl.policies.policy import Policy


ENV = "D4RL/pen/expert-v2"

MODEL = "FLOW"

MODELS = {
  "DIRECT" : lambda env: DirectPolicy(env),
  "FLOW" : lambda env: FlowPolicy(env),
  "IQL" : lambda env: IQLPolicy(
    env,
    gamma=0.99,
    tau=0.7,
    beta=3.0,
    polyak_tau=0.005,
  ),
}

torch.set_float32_matmul_precision("high")

def train_epoch(policy : Policy, opts : dict[str, torch.optim.Optimizer], train_loader, log : Logger, log_every = 10):

  policy.train()
  for b, batch in enumerate(train_loader):

    losses, stats = policy.train_step(batch)
    
    for k in losses.keys():
      opts[k].zero_grad(set_to_none=True)

    for k in losses.keys():
      losses[k].backward()

    for k in losses.keys():
      opts[k].step()

    if b % log_every == 0:
      log.log(stats)
    
    log.increment({"grad_steps" : 1})

   


    
if __name__ == "__main__":


  env_ds = minari.load_dataset(ENV, download=True)
  env = env_ds.recover_environment(render_mode="rgb_array")
  
  ds = SingleStepDataset.from_minari(env_ds)
  reward_scale = ds.normalize_rewards()
  stats = ds.normalize_observations()
  

  train_loader = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=True, prefetch_factor=4, num_workers=2, persistent_workers=True)

  device = "cuda"

  model : Policy = MODELS[MODEL](env).to(device)

  model.set_obs_normalization(stats["obs_mean"], stats["obs_std"])

  opts = model.create_optimizer()
  
  log = Logger("BA-RL-IQLearning", {"env_steps", "grad_steps"}, 
    { "env" : ENV, "model" : MODEL }
  )

  log.register("eval/*", "env_steps")
  log.register("train/*", "grad_steps")
  log.register("stats/*", "grad_steps")
  
  epochs = 100
  for i in range(epochs):
    
    train_epoch(model, opts, train_loader, log)

    data_runs = model.evaluate(env, runs=1)
    
    for run in data_runs:
      log.log(
        {
          "eval/cum_reward" : np.sum(run["rewards"]),
          "eval/length" : len(run["rewards"])
          }
      )

      log.log_video(f"eval/video", run["frames"])
      log.increment({"env_steps" : 1})

  env.close()

  torch.save(model.state_dict(), f"{MODEL.lower()}_{ENV.lower()}.pt")
