import argparse
import datetime
import os

from barl.dataset import SingleStepDataset
from barl.config import load_config

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import numpy as np

import torch

from barl.logger import Logger
from barl import Policy
from barl.env import Environment

from barl.rollout import ReplayBuffer

torch.set_float32_matmul_precision("high")

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "..", "config", "offline_to_online.yaml")


def offline_fraction(iteration: int, anneal_iters: int, start_frac: float, end_frac: float) -> float:
  """Linearly anneal the offline mixing fraction from start_frac to end_frac over anneal_iters outer iterations."""
  t = min(1.0, iteration / anneal_iters) if anneal_iters > 0 else 1.0
  return start_frac + (end_frac - start_frac) * t


def make_combined_dataset(offline_ds: SingleStepDataset, online_buffer: ReplayBuffer, frac: float) -> SingleStepDataset:
  """Combine offline and online trajectories so that, in expectation, `frac` of transitions come
  from the offline dataset. The online side always contributes its full current buffer; the
  offline side is subsampled (by trajectory, weighted by length) to hit the target ratio."""

  online_traj = online_buffer.buffer
  online_steps = sum(len(t) for t in online_traj)

  if frac >= 1.0 or online_steps == 0:
    return SingleStepDataset(list(offline_ds.buffer) + list(online_traj))
  if frac <= 0.0:
    return SingleStepDataset(list(online_traj)) if online_traj else offline_ds

  # Solve for target offline step count: frac = offline_steps / (offline_steps + online_steps)
  target_offline_steps = frac * online_steps / (1.0 - frac)

  offline_lengths = np.array([len(t) for t in offline_ds.buffer])
  order = np.random.permutation(len(offline_ds.buffer))

  cum = np.cumsum(offline_lengths[order])
  n_take = int(np.searchsorted(cum, target_offline_steps)) + 1
  n_take = min(n_take, len(offline_ds.buffer))

  offline_sample = [offline_ds.buffer[i] for i in order[:n_take]]

  return SingleStepDataset(offline_sample + list(online_traj))


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

  model.on_epoch_start()
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

  model.on_epoch_end()

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

  env         = environment.get_env()  # single env: used for eval (rendering) and action/obs spaces
  rollout_env = environment.get_vector_env(cfg["num_envs"])

  offline_ds = environment.get_dataset()
  buffer     = ReplayBuffer(cfg["buffer_size"], traj_len=cfg["traj_len"])

  checkpoint = cfg.get("checkpoint")
  if checkpoint is not None and os.path.exists(checkpoint):
    model = Policy.load(MODEL, checkpoint, env).to("cuda")
  else:
    model: Policy = Policy.make(MODEL, env).to("cuda")

  opts  = model.create_optimizer()

  log = Logger(
    "BA-RL",
    run_name=name,
    step_keys={"env_steps", "grad_steps"},
    tags=[MODEL, ENV, "offline_to_online"],
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
  for iteration in range(cfg["num_iterations"]):

    buffer.rollout(model, env=rollout_env, n_steps=cfg["rollout_steps"] // cfg["num_envs"],
                    seed=seed if iteration == 0 else None)

    log.increment({"env_steps": cfg["rollout_steps"]})

    frac = offline_fraction(iteration, cfg["anneal_iterations"], cfg["offline_start_frac"], cfg["offline_end_frac"])
    log.log({"train/offline_fraction": frac})

    combined_ds = make_combined_dataset(offline_ds, buffer, frac)

    log.message(f"Iteration {iteration}: offline_fraction={frac:.3f}, "
                f"online_trajectories={len(buffer.buffer)}, combined_trajectories={len(combined_ds.buffer)}")

    train_loader = torch.utils.data.DataLoader(
      combined_ds,
      batch_size=cfg["batch_size"],
      shuffle=True,
    )

    for i in range(cfg["epochs_per_rollout"]):
      best_reward = train_epoch(model, opts, train_loader, log, env, ckpt_dir, best_reward, cfg)

  model.save(os.path.join(ckpt_dir, "final.pt"))
  env.close()
  rollout_env.close()
  log.finish()


if __name__ == "__main__":

  args = argparse.ArgumentParser()
  args.add_argument("--config", default=DEFAULT_CONFIG,
                     help="Path to a YAML config (see config/offline_to_online.yaml). "
                          "Lists one or more {model, env, ...} runs, executed in series.")
  kwargs = args.parse_args()

  runs = load_config(kwargs.config)
  for i, cfg in enumerate(runs):
    print(f"=== Run {i + 1}/{len(runs)}: {cfg['model']} on {cfg['env']} ===")
    main(cfg)
