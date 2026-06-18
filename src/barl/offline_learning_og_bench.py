import datetime
import os
import time


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

from src.barl.rollout import ReplayBuffer
from src.barl.dataset import SingleStepDataset
import wandb

from src.barl.logger import Logger

from src.barl.policies.dqn import QPolicy
from src.barl.policies.flowq import FlowQLPolicy
from src.barl.policies.iql import IQLPolicy
from src.barl.policies.flow import FlowPolicy
from src.barl.policies.direct import DirectPolicy
from src.barl.policies.fam import FlowAdjointMatching

from src.barl.policies.policy import Policy
import ogbench

torch.set_float32_matmul_precision("high")

MODELS = {
  "DIRECT" : lambda env: DirectPolicy(env),
  "FQL" : lambda env : FlowQLPolicy(env, gamma=0.99, alpha=5), # Alpha needs to be tuned for every environment  
  "FLOW" : lambda env: FlowPolicy(env),
  "IQL" : lambda env: IQLPolicy(
    env,
    gamma=0.99,
    tau=0.7,
    beta=3.0,
    polyak_tau=0.005,
  ),
  "FAM" : lambda env: FlowAdjointMatching(
    env,
    gamma=0.99,
    tau=0.7,
    beta=3.0,
    polyak_tau=0.005,
  )
}

torch.set_float32_matmul_precision("high")

def train_epoch(policy : Policy, opts : dict[str, torch.optim.Optimizer], train_loader, log : Logger, env : gym.Env, log_every = 10):

  policy.train()
  for b, batch in enumerate(train_loader):

    losses, stats = policy.train_step(batch)

    for k in losses.keys():
      opts[k].zero_grad(set_to_none=True)

    for k in losses.keys():
      losses[k].backward()

    for k in losses.keys():
      torch.nn.utils.clip_grad_norm_(
          [p for group in opts[k].param_groups for p in group["params"]],
          max_norm=1.0,
      )
      opts[k].step()

    if b % log_every == 0:
      log.log(stats)
    

    if log.get_step("grad_steps") % 50_000 == 0:

      RUNS = 20
      
      data_runs = model.evaluate(env, runs=RUNS)

      log.increment({"env_steps" : RUNS})

      wrst_reward = min(np.mean(run["rewards"]) for run in data_runs)
      best_reward = max(np.mean(run["rewards"]) for run in data_runs)
      
      log.log(
        {
          "eval/avg_cum_reward" : np.mean(list(np.sum(run["rewards"]) for run in data_runs)),
          "eval/avg_length" : np.mean(list(len(run["rewards"]) for run in data_runs)),
          "eval/success_rate" : np.mean(list(run["terminated"] for run in data_runs))
        }
      )


      best_idx = next(i for i, run in enumerate(data_runs) if np.mean(run["rewards"]) >= best_reward)
      wrst_idx = next(i for i, run in enumerate(data_runs) if np.mean(run["rewards"]) <= wrst_reward)


      log.log_video(f"eval/video_best", data_runs[best_idx]["frames"])
      log.log_video(f"eval/worst_best", data_runs[wrst_idx]["frames"])

    log.increment({"grad_steps" : 1})

    
if __name__ == "__main__":

  MODEL = "FAM"
  DATASET = 'cube-double-play-singletask-task2-v0'

  now = datetime.datetime.now()
  f = now.strftime("%Y-%m-%d %H:%M:%S")
  name = f"{MODEL.lower()}:{DATASET.lower()} {f}"
  
  env, train_dataset, val_dataset = ogbench.make_env_and_datasets(DATASET, compact_dataset=True)
  

  ds = SingleStepDataset.from_ogbench(train_dataset)
  # reward_scale = ds.normalize_rewards()
  # stats = ds.normalize_observations()
  

  train_loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True, prefetch_factor=4, num_workers=4, persistent_workers=True)

  device = "cuda"

  model : Policy = MODELS[MODEL](env).to(device)

  # model.set_obs_normalization(stats["obs_mean"], stats["obs_std"])

  opts = model.create_optimizer()
  
  log = Logger("BA-RL", run_name=name, step_keys={"env_steps", "grad_steps"},
    config={ "env" : DATASET, "model" : MODEL, "env_obs" : str(env.observation_space), "env_act" : str(env.action_space) }
  )

  log.register("eval/*", "env_steps")
  log.register("train/*", "grad_steps")
  log.register("stats/*", "grad_steps")
  
  epochs = 500
  for i in range(epochs):
    train_epoch(model, opts, train_loader, log, env)

  env.close()

  # torch.save(model.state_dict(), f"{MODEL.lower()}_{ENV.lower()}.pt")
