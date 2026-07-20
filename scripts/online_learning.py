import argparse
import datetime
import os

from barl.dataset import SingleStepDataset, compute_gae
from barl.config import load_config

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import numpy as np

import torch

from barl.logger import Logger
from barl import Policy
from barl.env import Environment
from barl.vmap_policy import VmapPolicyGroup

from barl.rollout import ReplayBuffer

torch.set_float32_matmul_precision("high")

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "..", "config", "online_learning.yaml")


def evaluate_and_log(model: Policy, env: gym.Env, log: Logger, ckpt_dir: str, best_reward: float, cfg: dict) -> float:
  data_runs = model.evaluate(env, runs=cfg["eval_runs"])

  rewards     = [np.sum(run["rewards"]) for run in data_runs]
  avg_reward  = float(np.mean(rewards))
  best_idx    = int(np.argmax(rewards))
  worst_idx   = int(np.argmin(rewards))

  log.log({
    "eval/avg_cum_reward": avg_reward,
    "eval/avg_length":     np.mean([len(run["rewards"]) for run in data_runs]),
    "eval/success_rate":   np.mean([int(not bool(run["terminated"])) for run in data_runs]),
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


class _VmapRolloutAdapter:
  """
  Makes a VmapPolicyGroup look like a single Policy to ReplayBuffer.rollout()
  (which only ever calls .inference_step() on whatever it's handed): flattens
  the vmapped (num_seeds, num_envs, ...) shape down to the flat
  (num_seeds * num_envs, ...) shape rollout() expects, and back. Sub-env
  slots [seed*num_envs : (seed+1)*num_envs) always belong to that seed, so
  each seed's own slice of environments only ever queries that seed's own
  current weights for action selection.
  """

  def __init__(self, group: VmapPolicyGroup, num_envs: int):
    self.group = group
    self.num_envs = num_envs

  def eval(self): ...
  def train(self): ...

  def inference_step(self, observation: torch.Tensor, **kwargs):
    N, E = self.group.num_seeds, self.num_envs
    obs = observation.view(N, E, *observation.shape[1:])
    action, extras = self.group.inference_step(obs, **kwargs)
    action = action.reshape(N * E, *action.shape[2:])
    extras = {k: v.reshape(N * E, *v.shape[2:]) for k, v in extras.items()}
    return action, extras


def train_epoch_multi_seed(group: VmapPolicyGroup, opts: dict[str, torch.optim.Optimizer], loaders, log: Logger, cfg: dict):

  group.on_epoch_start()
  for b, per_seed_batches in enumerate(zip(*loaders)):

    batch = {k: torch.stack([sb[k] for sb in per_seed_batches]) for k in per_seed_batches[0]}
    losses, stats = group.train_step(batch)

    # Same deferred backward-then-step shape as train_epoch() above, plus
    # .sum() (not .mean()) over the num_seeds axis: each replica's loss is
    # an independent computation over its own parameter slice (no
    # cross-replica coupling), so summing gives replica i's slice exactly
    # its own gradient -- .mean() would silently scale every replica's
    # effective learning rate down by 1/num_seeds.
    for k in losses:
      opts[k].zero_grad(set_to_none=True)
      losses[k].sum().backward(retain_graph=True)

    for k in losses:
      torch.nn.utils.clip_grad_norm_(
        [p for grp in opts[k].param_groups for p in grp["params"]],
        max_norm=1.0,
      )
      opts[k].step()

    if b % cfg["log_every"] == 0:
      log.log({k: v.mean() for k, v in stats.items()})

    log.increment({"grad_steps": 1})

  group.on_epoch_end()


def main_multi_seed(cfg: dict):
  """
  Trains num_seeds independently-initialized replicas of the same
  model/env/hparams at once: N sub-vector-envs' worth of rollout collection
  (still real, CPU-bound env stepping -- not what vmap accelerates) feeding
  N separate per-seed datasets, then one vmapped train_step per minibatch
  across all N replicas together instead of N sequential Python-level
  train_step calls -- see barl/vmap_policy.py.

  Known limitations vs. the single-seed main(): no periodic eval/video
  logging (checkpoints are still written, just less often and without
  eval/success-rate stats), and on-policy algorithms (PPO, on_policy=True)
  aren't supported yet -- compute_gae needs a single Policy's critic, not
  a stacked group.
  """
  num_seeds = cfg["num_seeds"]
  base_seed = cfg.get("seed") or 0

  MODEL = cfg["model"]
  ENV   = cfg["env"]

  if Policy._classes[MODEL].on_policy:
    raise NotImplementedError(f"{MODEL} is on-policy (needs compute_gae against a single Policy's "
                               f"critic) -- multi-seed vmapped training doesn't support that yet.")

  now  = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  name = f"{MODEL.lower()}/{ENV.lower()}/{now}_x{num_seeds}seeds"

  environment: Environment = Environment.make(ENV)
  env = environment.get_env()  # obs/action spaces only in this path -- see docstring on eval

  group       = VmapPolicyGroup(MODEL, env, num_seeds=num_seeds, base_seed=base_seed, device="cuda")
  adapter     = _VmapRolloutAdapter(group, cfg["num_envs"])
  rollout_env = environment.get_vector_env(num_seeds * cfg["num_envs"])

  buffer = ReplayBuffer(cfg["buffer_size"] * num_seeds, traj_len=env.spec.max_episode_steps)
  opts   = group.create_optimizer()

  tags = [MODEL, ENV, "online", f"seeds={num_seeds}"]
  if cfg.get("sweep_tag"):
    tags.append(cfg["sweep_tag"])

  log = Logger(
    "BA-RL",
    run_name=name,
    step_keys={"env_steps", "grad_steps"},
    tags=tags,
    config={**cfg, "env_obs": str(env.observation_space), "env_act": str(env.action_space)},
  )
  log.register("train/*", "grad_steps")
  log.register("train/loss/*", "grad_steps")
  log.register("stats/*", "grad_steps")

  ckpt_dir = os.path.join("checkpoints", name)
  os.makedirs(ckpt_dir, exist_ok=True)

  for it in range(cfg["num_iterations"]):

    buffer.rollout(adapter, env=rollout_env,
                    n_steps=cfg["rollout_steps"] // (num_seeds * cfg["num_envs"]),
                    seed=base_seed if it == 0 else None)
    log.increment({"env_steps": cfg["rollout_steps"]})

    per_seed_trajs = [[] for _ in range(num_seeds)]
    for traj in buffer.buffer:
      per_seed_trajs[traj.id // cfg["num_envs"]].append(traj)

    log.message(f"Iteration {it}: " + ", ".join(f"seed{i}={len(t)}" for i, t in enumerate(per_seed_trajs)) + " trajectory chunks")

    if any(len(t) == 0 for t in per_seed_trajs):
      log.message("  skipping training this iteration -- at least one seed hasn't collected a full trajectory chunk yet")
      continue

    loaders = [
      torch.utils.data.DataLoader(SingleStepDataset(trajs), batch_size=max(1, cfg["batch_size"] // num_seeds), shuffle=True)
      for trajs in per_seed_trajs
    ]

    for _ in range(cfg["epochs_per_rollout"]):
      train_epoch_multi_seed(group, opts, loaders, log, cfg)

    # strictly on-policy across iterations, same reasoning as the
    # single-seed path's `if model.on_policy: buffer.buffer = []` --
    # except here it's unconditional: multi-seed collection always uses
    # the freshest per-iteration weights, never a stale replay buffer.
    buffer.buffer = []

  group.sync_replicas()
  for i, policy in enumerate(group.policies):
    policy.save(os.path.join(ckpt_dir, f"seed{i}_final.pt"))

  env.close()
  rollout_env.close()
  log.finish()


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

  buffer = ReplayBuffer(cfg["buffer_size"], traj_len=env.spec.max_episode_steps)

  checkpoint = cfg.get("checkpoint")
  if checkpoint is not None and os.path.exists(checkpoint):
    model = Policy.load(MODEL, checkpoint, env).to("cuda")
  else:
    model: Policy = Policy.make(MODEL, env).to("cuda")

  opts  = model.create_optimizer()

  tags = [MODEL, ENV, "online"]
  if cfg.get("sweep_tag"):
    tags.append(cfg["sweep_tag"])

  log = Logger(
    "BA-RL",
    run_name=name,
    step_keys={"env_steps", "grad_steps"},
    tags=tags,
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
  for it in range(cfg["num_iterations"]):

    buffer.rollout(model, env=rollout_env, n_steps=cfg["rollout_steps"] // cfg["num_envs"],
                    seed=seed if it == 0 else None)
    log.increment({"env_steps": cfg["rollout_steps"]})

    log.message(f"Finished Rollout buffer contains {len(buffer.buffer)} episodes")

    if model.on_policy:
      # GAE needs the trajectories in-order (see dataset.compute_gae) --
      # must run before SingleStepDataset flattens/shuffles them below.
      compute_gae(buffer.buffer, model.critic, gamma=model.gamma, lam=model.lam, device=model.device)

    train_loader = torch.utils.data.DataLoader(
      SingleStepDataset.from_replay(buffer),
      batch_size=cfg["batch_size"],
      shuffle=True,
    )

    for i in range(cfg["epochs_per_rollout"]):
      best_reward = train_epoch(model, opts, train_loader, log, env, ckpt_dir, best_reward, cfg)

    if model.on_policy:
      # strictly on-policy: never replay trajectories from a stale policy
      # version in a later rollout's training pass.
      buffer.buffer = []

  model.save(os.path.join(ckpt_dir, "final.pt"))
  env.close()
  rollout_env.close()
  log.finish()


if __name__ == "__main__":

  args = argparse.ArgumentParser()
  args.add_argument("--config", default=DEFAULT_CONFIG,
                     help="Path to a YAML config (see config/online_learning.yaml). "
                          "Lists one or more {model, env, ...} runs, executed in series.")
  kwargs = args.parse_args()

  runs = load_config(kwargs.config)
  for i, cfg in enumerate(runs):
    num_seeds = cfg.get("num_seeds", 1)
    if num_seeds > 1:
      print(f"=== Run {i + 1}/{len(runs)}: {cfg['model']} on {cfg['env']} ({num_seeds} seeds, vmapped) ===")
      main_multi_seed(cfg)
    else:
      print(f"=== Run {i + 1}/{len(runs)}: {cfg['model']} on {cfg['env']} ===")
      main(cfg)
