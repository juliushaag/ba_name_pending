import argparse
import datetime
import os

from barl.config import load_config

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import numpy as np

import torch

from barl.logger import Logger
from barl import Policy
from barl.env import Environment

torch.set_float32_matmul_precision("high")

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "..", "config", "offline_learning.yaml")


def evaluate_and_log(model: Policy, env: gym.Env, log: Logger, ckpt_dir: str, best_reward: float, cfg: dict) -> float:
  data_runs = model.evaluate(env, runs=cfg["eval_runs"])

  log.increment({"env_steps": cfg["eval_runs"]})

  rewards     = [np.sum(run["rewards"]) for run in data_runs]
  avg_reward  = float(np.mean(rewards))
  best_idx    = int(np.argmax(rewards))
  worst_idx   = int(np.argmin(rewards))

  log.log({
    "eval/avg_cum_reward": avg_reward,
    "eval/avg_length":     np.mean([len(run["rewards"]) for run in data_runs]),
    "eval/success_rate":   np.mean([run["terminated"]   for run in data_runs]),
  })

  log.log_video("eval/video_best",  data_runs[best_idx]["frames"])
  log.log_video("eval/video_worst", data_runs[worst_idx]["frames"])

  model.save(os.path.join(ckpt_dir, "latest.pt"))

  if avg_reward > best_reward:
    model.save(os.path.join(ckpt_dir, "best.pt"))
    best_reward = avg_reward
    log.summary("best_reward", best_reward)

  return best_reward


def train_epoch(model: Policy, opts: dict[str, torch.optim.Optimizer], train_loader, log: Logger, env: gym.Env, ckpt_dir: str, best_reward: float, cfg: dict) -> float:

  model.train()
  for b, batch in enumerate(train_loader):

    losses, stats = model.train_step(batch)

    # One update after another, with a gradient reset in between each --
    # but every backward pass completes before any optimizer actually
    # steps, so an actor loss reading the critic's forward pass never has
    # its graph invalidated by the critic's own optimizer stepping mid-loop.
    for k in losses:
      opts[k].zero_grad(set_to_none=True)
      losses[k].backward(retain_graph=True)

    for k in losses:
      torch.nn.utils.clip_grad_norm_(
        [p for group in opts[k].param_groups for p in group["params"]],
        max_norm=1.0,
      )
      opts[k].step()

    if b % cfg["log_every"] == 0:
      log.log(stats)

    log.increment({"grad_steps": 1})

    if log.get_step("grad_steps") % cfg["eval_every"] == 0:
      model.eval()
      best_reward = evaluate_and_log(model, env, log, ckpt_dir, best_reward, cfg)
      model.train()

  return best_reward


def main(cfg: dict):
  seed = cfg.get("seed")
  if seed is not None:
    torch.manual_seed(seed)
    np.random.seed(seed)

  MODEL = cfg["model"]
  ENV   = cfg["env"]

  now  = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  name = f"{MODEL.lower()}/{ENV.lower()}/{now}"

  environment: Environment = Environment.make(ENV)

  env = environment.get_env()
  ds  = environment.get_dataset()

  train_loader = torch.utils.data.DataLoader(
    ds, batch_size=cfg["batch_size"], shuffle=True,
    prefetch_factor=cfg["prefetch_factor"], num_workers=cfg["num_workers"], persistent_workers=True,
  )

  model: Policy = Policy.make(MODEL, env=env).to("cuda")

  opts  = model.create_optimizer()

  log = Logger(
    "BA-RL",
    run_name=name,
    step_keys={"env_steps", "grad_steps"},
    tags=[MODEL, ENV, "offline"],
    config={**cfg, "hparams": model.hparams,
            "env_obs": str(env.observation_space), "env_act": str(env.action_space)}
  )

  log.register("eval/*",  "env_steps")
  log.register("train/*", "grad_steps")
  log.register("train/loss/*", "grad_steps")
  log.register("stats/*", "grad_steps")

  ckpt_dir = os.path.join("checkpoints", name)
  os.makedirs(ckpt_dir, exist_ok=True)

  best_reward = float("-inf")
  for _ in range(cfg["num_epochs"]):
    best_reward = train_epoch(model, opts, train_loader, log, env, ckpt_dir, best_reward, cfg)

  model.save(os.path.join(ckpt_dir, "final.pt"))
  env.close()
  log.finish()


if __name__ == "__main__":

  args = argparse.ArgumentParser()
  args.add_argument("--config", default=DEFAULT_CONFIG,
                     help="Path to a YAML config (see config/offline_learning.yaml). "
                          "Lists one or more {model, env, ...} runs, executed in series.")
  kwargs = args.parse_args()

  runs = load_config(kwargs.config)
  for i, cfg in enumerate(runs):
    print(f"=== Run {i + 1}/{len(runs)}: {cfg['model']} on {cfg['env']} ===")
    main(cfg)
