import argparse
import datetime
import os

from barl.dataset import SingleStepDataset

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import numpy as np

import torch

from barl.logger import Logger
from barl import Policy
from barl.env import Environment

from barl.rollout import ReplayBuffer

torch.set_float32_matmul_precision("high")

EVAL_EVERY  = 10_000   # grad steps
EVAL_RUNS   = 50
LOG_EVERY   = 10       # batches


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


def evaluate_and_log(model: Policy, env: gym.Env, log: Logger, ckpt_dir: str, best_reward: float) -> float:
  data_runs = model.evaluate(env, runs=EVAL_RUNS)

  log.increment({"env_steps": EVAL_RUNS})

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

  return best_reward


def train_epoch(model: Policy, opts: dict[str, torch.optim.Optimizer], train_loader, log: Logger, env: gym.Env, ckpt_dir: str, best_reward: float, frozen: Policy = None) -> float:

  model.on_epoch_start()
  model.train()
  for b, batch in enumerate(train_loader):

    losses, stats = model.train_step(batch, frozen=frozen)

    for k in losses:
      opts[k].zero_grad(set_to_none=True)
      losses[k].backward(retain_graph=True)
      torch.nn.utils.clip_grad_norm_(
        [p for group in opts[k].param_groups for p in group["params"]],
        max_norm=1.0,
      )
      opts[k].step()

    if b % LOG_EVERY == 0:
      log.log(stats)

    log.increment({"grad_steps": 1})

    if log.get_step("grad_steps") % EVAL_EVERY == 0:
      model.eval()
      best_reward = evaluate_and_log(model, env, log, ckpt_dir, best_reward)
      model.train()

  model.on_epoch_end()

  return best_reward


if __name__ == "__main__":

  args = argparse.ArgumentParser()

  args.add_argument("--model", required=True)
  args.add_argument("--env", required=True)
  args.add_argument("--pretrained", default=None)
  args.add_argument("--checkpoint", default=None)
  args.add_argument("--epochs_per_rollout", type=int, default=16)
  args.add_argument("--traj_len", type=int, default=500)
  args.add_argument("--num_envs", type=int, default=32)
  args.add_argument("--num_iterations", type=int, default=100)
  args.add_argument("--offline_start_frac", type=float, default=1.0,
                     help="Fraction of transitions drawn from the offline dataset at the start of training")
  args.add_argument("--offline_end_frac", type=float, default=0.0,
                     help="Fraction of transitions drawn from the offline dataset once annealing has finished")
  args.add_argument("--anneal_iterations", type=int, default=50,
                     help="Number of outer (rollout) iterations over which the offline fraction is annealed")

  kwargs = args.parse_args()

  MODEL = kwargs.model
  ENV   = kwargs.env
  pretrained = kwargs.pretrained
  checkpoint = kwargs.checkpoint
  epochs_per_rollout = kwargs.epochs_per_rollout
  traj_len = kwargs.traj_len
  num_envs = kwargs.num_envs
  num_iterations = kwargs.num_iterations
  offline_start_frac = kwargs.offline_start_frac
  offline_end_frac = kwargs.offline_end_frac
  anneal_iterations = kwargs.anneal_iterations

  now  = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  name = f"{MODEL.lower()}/{ENV.lower()}/{now}"

  environment: Environment = Environment.make(ENV)

  env         = environment.get_env()  # single env: used for eval (rendering) and action/obs spaces
  rollout_env = gym.vector.SyncVectorEnv(
    [environment.get_env for _ in range(num_envs)],
    autoreset_mode=gym.vector.AutoresetMode.DISABLED,
  )

  offline_ds = environment.get_dataset()
  buffer     = ReplayBuffer(256, traj_len=traj_len)

  if checkpoint is not None and os.path.exists(checkpoint):
    model = Policy.load(MODEL, checkpoint, env).to("cuda")
  else:
    model: Policy = Policy.make(MODEL, env).to("cuda")

  frozen = None
  if pretrained is not None and os.path.exists(pretrained):
    frozen = Policy.load_any(path=pretrained, env=env).to("cuda")
    frozen.eval()
    frozen.requires_grad_(False)

  opts  = model.create_optimizer()

  log = Logger(
    "BA-RL",
    run_name=name,
    step_keys={"env_steps", "grad_steps"},
    tags=[MODEL, ENV, "offline_to_online"],
    config={"env": ENV, "model": MODEL, "hparams": model.hparams,
            "env_obs": str(env.observation_space), "env_act": str(env.action_space),
            "offline_start_frac": offline_start_frac, "offline_end_frac": offline_end_frac,
            "anneal_iterations": anneal_iterations}
  )

  log.register("eval/*",  "env_steps")
  log.register("train/*", "grad_steps")
  log.register("train/loss/*", "grad_steps")
  log.register("stats/*", "grad_steps")

  ckpt_dir = os.path.join("checkpoints", name)
  os.makedirs(ckpt_dir, exist_ok=True)

  best_reward = float("-inf")
  for iteration in range(num_iterations):

    buffer.rollout(model, env=rollout_env, n_steps=50_000 // num_envs)

    log.increment({"env_steps": 50_000})

    frac = offline_fraction(iteration, anneal_iterations, offline_start_frac, offline_end_frac)
    log.log({"train/offline_fraction": frac})

    combined_ds = make_combined_dataset(offline_ds, buffer, frac)

    print(f"Iteration {iteration}: offline_fraction={frac:.3f}, "
          f"online_trajectories={len(buffer.buffer)}, combined_trajectories={len(combined_ds.buffer)}")

    train_loader = torch.utils.data.DataLoader(
      combined_ds,
      batch_size=1024,
      shuffle=True,
    )

    for i in range(epochs_per_rollout):
      best_reward = train_epoch(model, opts, train_loader, log, env, ckpt_dir, best_reward, frozen=frozen)

  model.save(os.path.join(ckpt_dir, "final.pt"))
  env.close()
  rollout_env.close()
  log.finish()
