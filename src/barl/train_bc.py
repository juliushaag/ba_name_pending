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
from minari import DataCollector


torch.manual_seed(42)

from src.barl.dataset import collate_fn


minari_dataset = minari.load_dataset("mujoco/invertedpendulum/expert-v0", download=True)
dataloader = DataLoader(minari_dataset, batch_size=256, shuffle=True, collate_fn=collate_fn)

env = minari_dataset.recover_environment()
observation_space = env.observation_space
action_space = env.action_space

assert isinstance(observation_space, spaces.Box)

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = self.layers(x)
        x = torch.tanh(x) * 3
        return x

device = "cuda"


print(action_space)
policy_net = PolicyNetwork(np.prod(observation_space.shape), np.prod(action_space.shape)).to(device)
optimizer = torch.optim.AdamW(policy_net.parameters())
loss_fn = nn.MSELoss()

num_epochs = 1028

for epoch in range(num_epochs):
    for batch in dataloader:
        a_pred = policy_net(batch['observations'][:, :-1].to(device))
        loss = loss_fn(a_pred, batch["actions"].to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch: {epoch}/{num_epochs}, Loss: {loss.item()}")


env = minari_dataset.recover_environment(eval_env=True)
obs, _ = env.reset(seed=42)
done = False
accumulated_rew = 0
with torch.inference_mode():
  while not done:
      action = policy_net(torch.Tensor(obs).to(device)).cpu().to(torch.float32)
      obs, reward, terminated, truncated, _ = env.step(np.zeros(1)) # action.numpy())
      done = terminated or truncated
      accumulated_rew += reward

env.close()
print("Accumulated rew: ", accumulated_rew)