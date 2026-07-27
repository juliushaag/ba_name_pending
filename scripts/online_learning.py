import argparse
import datetime
import math
import os
import warnings

# Must be set before the first `import mujoco` (transitively, via `barl` below) --
# mujoco picks its GL backend at import time, so setting this later leaves it
# locked onto GLFW, which hard-crashes env.render() on a headless (no X11) box.
os.environ.setdefault("MUJOCO_GL", "egl")

warnings.filterwarnings("ignore", module="glfw")  # dm_control still probes GLFW at import regardless

from barl.dataset import SingleStepDataset, compute_gae
from barl.config import load_config

import gymnasium as gym
import numpy as np

import torch

from barl.logger import Logger
from barl import Policy
import barl.model_config as model_config
from barl.env import Environment

from barl.rollout import ReplayBuffer

torch.set_float32_matmul_precision("high")

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "..", "config", "online_learning.yaml")


def lr_schedule(step: int, warmup_steps: int, total_steps: int) -> float:
  """Multiplicative LR factor: linear 0 -> 1 ramp over `warmup_steps`, then a
  quarter-sine decay 1 -> 0 from `warmup_steps` to `total_steps` (flat at 0
  past that, in case `total_steps` -- an estimate, see `total_grad_steps` in
  main() -- undershoots the run's actual grad-step count)."""
  if warmup_steps <= 0 or step >= warmup_steps:
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return math.sin((1.0 - progress) * math.pi / 2)
  return step / warmup_steps


def create_schedulers(opts: dict[str, torch.optim.Optimizer], warmup_steps: int, total_steps: int) -> dict[str, torch.optim.lr_scheduler.LambdaLR]:
  """One LambdaLR per optimizer in `opts`, all sharing the same warmup/decay
  shape -- each scales its own optimizer's per-param-group base lr, so
  differently-tuned optimizers (actor/critic/explore/...) keep their own
  absolute lr while following the same schedule."""
  return {
    k: torch.optim.lr_scheduler.LambdaLR(opt, lambda step: lr_schedule(step, warmup_steps, total_steps))
    for k, opt in opts.items()
  }


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


def train_batch(model: Policy, opts: dict[str, torch.optim.Optimizer], scheds: dict[str, torch.optim.lr_scheduler.LambdaLR],
                 batch, batch_idx: int, log: Logger, env: gym.Env, ckpt_dir: str, best_reward: float, cfg: dict) -> float:
  """One gradient step: loss compute, backward+clip+step per loss key, stat
  logging, grad_steps increment, periodic eval trigger. Shared by the
  on-policy epoch loop (train_epoch) and the off-policy fixed-UTD loop in
  main() -- both just differ in how `batch` is produced."""
  losses, stats = model.train_step(batch)

  for k in losses:
    opts[k].zero_grad(set_to_none=True)
    losses[k].backward(retain_graph=True)
    norm = torch.nn.utils.clip_grad_norm_(
      [p for group in opts[k].param_groups for p in group["params"]],
      max_norm=1.0,
      error_if_nonfinite=True,
    )

    opts[k].step()
    scheds[k].step()

    stats[f"train/loss/{k}"] = losses[k].detach()
    stats[f"train/norm/{k}"] = norm
    stats[f"train/lr/{k}"]   = scheds[k].get_last_lr()[0]

  if batch_idx % cfg["log_every"] == 0:
    log.log(stats)

  log.increment({"grad_steps": 1})

  if log.get_step("grad_steps") % cfg["eval_every"] == 0:
    model.eval()
    best_reward = evaluate_and_log(model, env, log, ckpt_dir, best_reward, cfg)
    model.train()

  return best_reward


def log_model_stats(model: Policy, log: Logger, num_steps: int, cfg: dict) -> None:
  epoch_time = model.epoch_time()
  n_samples  = max(1, num_steps * cfg["batch_size"])
  log.log({
    "model/num_params":               model.num_params(),
    "model/inference_latency_ms":     model.measure_latency(),
    "model/epoch_time_s":             epoch_time,
    "model/epoch_time_per_sample_ms": 1000 * epoch_time / n_samples,
  })


def train_epoch(model: Policy, opts: dict[str, torch.optim.Optimizer], scheds: dict[str, torch.optim.lr_scheduler.LambdaLR],
                 train_loader, log: Logger, env: gym.Env, ckpt_dir: str, best_reward: float, cfg: dict) -> float:

  model.on_epoch_start()
  model.train()
  num_steps = 0
  for b, batch in enumerate(train_loader):
    num_steps += 1
    best_reward = train_batch(model, opts, scheds, batch, b, log, env, ckpt_dir, best_reward, cfg)

  model.on_epoch_end()

  log_model_stats(model, log, num_steps, cfg)

  return best_reward


def validate_run(model: Policy, opts: dict[str, torch.optim.Optimizer], buffer: ReplayBuffer,
                  rollout_env: gym.vector.VectorEnv, cfg: dict):
  """Cheap smoke test for `--validate` sweeps: one rollout chunk + one
  train_step + optimizer step, no eval/checkpointing/wandb logging."""
  buffer.rollout(model, env=rollout_env, n_steps=cfg["batch_size"] * 10,
                 seed=cfg.get("seed"))

  if model.on_policy:
    compute_gae(buffer.buffer, model.critic, gamma=model.gamma, lam=model.lam, device=model.device)

  train_loader = torch.utils.data.DataLoader(
    SingleStepDataset.from_replay(buffer),
    batch_size=cfg["batch_size"],
  )
  batch = next(iter(train_loader))

  model.train()
  model.on_epoch_start()
  losses, _ = model.train_step(batch)

  for k in losses:
    opts[k].zero_grad(set_to_none=True)
    losses[k].backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(
      [p for group in opts[k].param_groups for p in group["params"]],
      max_norm=1.0,
    )
    opts[k].step()

  model.on_epoch_end()

  print(f"[validate] {cfg['model']} on {cfg['env']}: OK ({len(losses)} loss term(s), "
        f"batch size {next(iter(batch.values())).shape[0]})")


def main(cfg: dict):
  seed = cfg.get("seed")
  if seed is not None:
    torch.manual_seed(seed)
    np.random.seed(seed)

  MODEL = cfg["model"]
  ENV   = cfg["env"]

  # Unconditional: also resets any override left over from a previous run in
  # this process's `runs:` series when the current run doesn't carry one.
  model_config.set_models_override(cfg.get("models"))

  now  = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  name = f"{MODEL.lower()}/{ENV.lower()}/{now}"

  environment: Environment = Environment.make(ENV)

  env = environment.get_env()  # single env: used for eval (rendering) and action/obs spaces

  rollout_env = environment.get_vector_env(1 if cfg.get("validate") else cfg["num_envs"])

  buffer = ReplayBuffer(cfg["buffer_size"], traj_len=environment.get_max_traj_length(env))

  checkpoint = cfg.get("checkpoint")
  if checkpoint is not None and os.path.exists(checkpoint):
    model = Policy.load(MODEL, checkpoint, env).to("cuda")
  else:
    model: Policy = Policy.make(MODEL, env).to("cuda")

  opts  = model.create_optimizer()

  # Estimated total grad-step budget for the whole run, matching the same
  # accounting the training loop below uses per-iteration -- just an estimate
  # (on-policy's actual batches/epoch varies with episode boundaries), fine
  # for an LR schedule since lr_schedule() flattens at 0 past it regardless.
  if model.on_policy:
    total_grad_steps = max(1, cfg["total_env_steps"] // cfg["batch_size"]) * cfg["epochs_per_rollout"]
  else:
    total_grad_steps = max(1, round(cfg["total_env_steps"] * cfg["utd_ratio"]))

  scheds = create_schedulers(opts, warmup_steps=cfg["lr_warmup_steps"], total_steps=total_grad_steps)

  if cfg.get("validate"):
    validate_run(model, opts, buffer, rollout_env, cfg)
    env.close()
    rollout_env.close()
    return

  tags = [MODEL, ENV, "online"]
  if cfg.get("sweep_tag"):
    tags.append(cfg["sweep_tag"])

  resolved_model_cfg = cfg.get("models") or model_config.snapshot_for(MODEL)

  log = Logger(
    "BA-RL",
    run_name=name,
    step_keys={"env_steps", "grad_steps"},
    tags=tags,
    config={**cfg, "resolved_model": resolved_model_cfg,
            "env_obs": str(env.observation_space), "env_act": str(env.action_space)},
    group=cfg["sweep_tag"]
  )

  log.register("eval/*",  "env_steps")
  log.register("train/*", "grad_steps")
  log.register("train/loss/*", "grad_steps")
  log.register("stats/*", "grad_steps")
  log.register("model/*", "grad_steps")
  log.register("rollout/*", "env_steps")

  ckpt_dir = os.path.join("checkpoints", name)
  os.makedirs(ckpt_dir, exist_ok=True)

  # Off-policy only: PPO's compute_gae needs extras["value"] on every trajectory,
  # which a random-action rollout never stashes (see ReplayBuffer.random_fill).
  if not model.on_policy and cfg.get("random_steps", 0) > 0:
    buffer.random_fill(rollout_env, n_steps=cfg["random_steps"] // cfg["num_envs"], seed=seed)
    log.increment({"env_steps": cfg["random_steps"]})
    log.message(f"Random-filled buffer with {cfg['random_steps']} env steps before training")

  best_reward = float("-inf")
  it = 0
  while log.get_step("env_steps") < cfg["total_env_steps"]:

    rollout_stats = buffer.rollout(model, env=rollout_env, n_steps=cfg["rollout_steps"] // cfg["num_envs"],
                    seed=seed if it == 0 else None)
    log.increment({"env_steps": cfg["rollout_steps"]})
    log.log({f"rollout/{k}": v for k, v in rollout_stats.items()})

    n_transitions = sum(len(traj) for traj in buffer.buffer)

    if model.on_policy:
      batches_per_epoch = -(-n_transitions // cfg["batch_size"])  # ceil div
      grad_steps_this_iter = batches_per_epoch * cfg["epochs_per_rollout"]
    else:
      grad_steps_this_iter = round(cfg["rollout_steps"] * cfg["utd_ratio"])

    log.message(f"Finished Rollout buffer contains {len(buffer.buffer)} episodes "
                f"({n_transitions} transitions, ~{grad_steps_this_iter} grad steps this iter)")
    log.log({"stats/buffer_transitions": n_transitions,
              "stats/utd_ratio": grad_steps_this_iter / cfg["rollout_steps"]})

    if model.on_policy:
      # GAE needs the trajectories in-order -- must run before SingleStepDataset shuffles them.
      compute_gae(buffer.buffer, model.critic, gamma=model.gamma, lam=model.lam, device=model.device)

      train_loader = torch.utils.data.DataLoader(
        SingleStepDataset.from_replay(buffer),
        batch_size=cfg["batch_size"],
        shuffle=True,
      )

      for _ in range(cfg["epochs_per_rollout"]):
        best_reward = train_epoch(model, opts, scheds, train_loader, log, env, ckpt_dir, best_reward, cfg)

      buffer.buffer = []  # never replay trajectories from a stale policy version

    else:
      model.on_epoch_start()
      model.train()
      for b in range(grad_steps_this_iter):
        batch = buffer.sample(cfg["batch_size"])
        best_reward = train_batch(model, opts, scheds, batch, b, log, env, ckpt_dir, best_reward, cfg)
      model.on_epoch_end()

      log_model_stats(model, log, grad_steps_this_iter, cfg)

    it += 1

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
    print(f"=== Run {i + 1}/{len(runs)}: {cfg['model']} on {cfg['env']} ===")
    main(cfg)
