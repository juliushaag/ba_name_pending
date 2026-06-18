import os
import sys

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import minari

from src.barl.models.mlp import MLPPolicy
from source.policies.flow import FlowPolicy
from src.barl.policies.policy import Policy

torch.manual_seed(42)
torch.set_float32_matmul_precision("high")

from src.barl.dataset import collate_fn

minari_dataset = minari.load_dataset("D4RL/pen/expert-v2", download=True)
dataloader = DataLoader(minari_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)

env = minari_dataset.recover_environment()
observation_space = env.observation_space
action_space = env.action_space

all_actions = np.concatenate([ep.actions for ep in minari_dataset], axis=0)
all_obs = np.concatenate([ep.observations for ep in minari_dataset], axis=0)

print("#" * 20)
print('Shape:', all_actions.shape)
print('Mean: ', all_actions.mean(axis=0))
print('Std:  ', all_actions.std(axis=0))
print('Min:  ', all_actions.min(axis=0))
print('Max:  ', all_actions.max(axis=0))
print()
print("#" * 20)
print('Shape:', all_obs.shape)
print('Mean: ', all_obs.mean(axis=0))
print('Std:  ', all_obs.std(axis=0))
print('Min:  ', all_obs.min(axis=0))
print('Max:  ', all_obs.max(axis=0))

observation_mean = all_obs.mean(axis=0).astype(np.float32)
observation_std = all_obs.std(axis=0).astype(np.float32)

device = "cuda"
num_epochs = 128

print("Observation space:", observation_space)
print("Action space:", action_space)

obs_dim = np.prod(observation_space.shape)
action_dim =  np.prod(action_space.shape)

policy : Policy = FlowPolicy(observation_space, action_space).to(device)
optimizer = policy.create_optimizer()
sched = policy.create_scheduler(optimizer, num_epochs)

print(f"Training Policy with {(policy.num_params() / 1e6):.2f}M parameters")

for epoch in range(num_epochs):
    losses = list()
    for batch in dataloader:
        batch["observations"] = (batch["observations"] - observation_mean) / (observation_std + 1e-8) 

        loss = policy.train_step(batch)

        losses.append(loss.detach())
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

    print(f"Epoch: {epoch}/{num_epochs}, Loss: {torch.stack(losses).mean().item()} LR: {optimizer.param_groups[0]['lr']}")
    losses.clear()
    sched.step()

env = minari_dataset.recover_environment()
for i in range(10):
    obs, _ = env.reset(seed=42)
    done = False
    accumulated_rew = 0
    history = None
    while not done:
        obs = (obs.astype(np.float32) - observation_mean) / (observation_std + 1e-8) 
        action, history = policy.inference_step(obs, history)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        accumulated_rew += reward
    
    print("Accumulated rew: ", accumulated_rew)

env.close()