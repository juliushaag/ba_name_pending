"""The online training loop, for any number of seeds.

Seeds are a first-class axis rather than a separate code path: N policies are
built independently, their states stacked, and one `jax.jit(jax.vmap(...))`
steps all of them. N=1 is the same code with an axis of size 1, which costs
nothing measurable (1.68 vs 1.63 ms/step) -- so there is one loop to maintain
rather than two that drift.

The jit around the vmap is load-bearing. `jax.vmap` alone only traces; without
an enclosing jit the step is dispatched op-by-op and costs ~4x more at N=1 and
~2x at N=10.
"""

import datetime
import os
import time

import jax
import numpy as np
from flax import nnx

import barl_jax.model_config as model_config
from barl_jax.env import Environment
from barl_jax.logger import Logger
from barl_jax.metrics import DEBUG, PRIMARY, to_host
from barl_jax.multiseed import MultiSeedReplay, StackedPolicy, split_metrics, stack, unstack
from barl_jax.policy import TRAINABLE, Policy
from barl_jax.train import train_step

seeded_step = jax.jit(jax.vmap(train_step, in_axes=(None, 0, 0, None)),
                      static_argnums=(0, 3), donate_argnums=(1,))


class Run:
  """Everything a run needs, built once: envs, buffer, stacked policies, logger.

  Constructed by `setup()` so that `train()` and `validate()` share it rather
  than each rebuilding the world."""

  def __init__(self, cfg, num_seeds):
    self.cfg = cfg
    self.num_seeds = num_seeds
    self.model_name, self.env_name = cfg["model"], cfg["env"]
    self.base_seed = cfg.get("seed") or 0
    self.num_envs = cfg["num_envs"]


def setup(cfg: dict, num_seeds: int, *, validate_only: bool = False) -> Run:
  run = Run(cfg, num_seeds)
  np.random.seed(run.base_seed)

  # Unconditional: also resets anything left over from a previous run in this
  # process's `runs:` series when the current run doesn't carry one.
  model_config.set_models_override(cfg.get("models"))
  model_config.set_overrides(cfg.get("model_overrides"))

  problems = model_config.validate(run.model_name)
  if problems:
    raise ValueError(f"{run.model_name}: unusable model config, refusing to start:\n  "
                     + "\n  ".join(problems))

  run.environment = Environment.make(run.env_name)
  run.env = run.environment.get_env()       # single env: spaces and EnvStats
  run.env_stats = run.environment.get_env_stats(run.env)

  num_envs = 1 if validate_only else run.num_envs
  run.num_envs = num_envs
  run.rollout_env = run.environment.get_vector_env(num_seeds * num_envs)
  run.buffer = MultiSeedReplay(num_seeds, num_envs, cfg["buffer_size"],
                               traj_len=run.environment.get_max_traj_length(run.env))

  checkpoint = cfg.get("checkpoint")
  if checkpoint is not None and os.path.exists(checkpoint) and num_seeds == 1:
    policies = [Policy.load(run.model_name, checkpoint, run.env, run.env_stats)]
  else:
    policies = [Policy.make(run.model_name, run.env, env_stats=run.env_stats,
                            seed=run.base_seed + i) for i in range(num_seeds)]

  if policies[0].on_policy:
    raise NotImplementedError(
      f"{run.model_name} is on-policy, and the GAE path is not ported -- barl_jax "
      f"currently implements the off-policy loop only.")

  total_grad_steps = max(1, round(cfg["total_env_steps"] * cfg["utd_ratio"]))
  schedule = Policy.lr_schedule(cfg["lr_warmup_steps"], total_grad_steps)
  optimizers = [nnx.Optimizer(p, p.build_optimizer(lr_schedule=schedule), wrt=TRAINABLE)
                for p in policies]

  run.reference = policies[0]
  run.graphdef, run.state = stack(list(zip(policies, optimizers)))
  return run


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------


def _checkpoint_dir(run: Run, root: str, seed: int) -> str:
  """A single-seed run keeps the flat layout; several seeds get one dir each."""
  return root if run.num_seeds == 1 else os.path.join(root, f"seed{seed}")


def evaluate(run: Run, envs, log, ckpt_dir, best):
  """Each seed on its own, through the ordinary single-policy path."""
  cfg = run.cfg
  eval_env, render_env = envs
  rewards, lengths, successes = [], [], []

  # Entry/exit are logged because an eval is the one phase that can stall for
  # tens of minutes with nothing on stdout, and telling "stalled" from "running
  # normally, just slow" after the fact otherwise means reading a stack trace.
  started = time.perf_counter()
  log.message(f"Eval start @ env_step {log.get_step('env_steps')} "
              f"(every {cfg['eval_every']}), {run.num_seeds} seeds x {cfg['eval_runs']} episodes")

  for i in range(run.num_seeds):
    policy, _ = unstack(run.graphdef, run.state, i)
    # Only seed 0's video is ever logged, so only seed 0 renders one. Every seed
    # rendering a full episode and discarding all but the first cost a whole
    # extra rendered episode per seed per eval -- ~5 s each at 1000 frames.
    videos = cfg.get("eval_videos", 2) if i == 0 else 0
    runs = policy.evaluate(eval_env, runs=cfg["eval_runs"],
                           videos=videos, render_env=render_env)
    reward = float(np.mean([np.sum(r["rewards"]) for r in runs]))
    rewards.append(reward)
    lengths.append(np.mean([len(r["rewards"]) for r in runs]))
    successes.append(np.mean([int(not bool(r["terminated"])) for r in runs]))

    root = _checkpoint_dir(run, ckpt_dir, i)
    policy.save(os.path.join(root, "latest"))
    if reward > best[i]:
      best[i] = reward
      policy.save(os.path.join(root, "best"))

    if i == 0:  # videos from one seed are enough to watch
      for v, r in enumerate(x for x in runs if x["frames"]):
        log.log_video(f"eval/video_{v}", r["frames"])

  # Eval is already per-seed -- each seed was rolled out on its own above -- so
  # it goes straight to that seed's run rather than being averaged first.
  rewards = np.asarray(rewards)
  per_seed = [{"eval/avg_cum_reward": float(rewards[i]),
               "eval/avg_length":     float(lengths[i]),
               "eval/success_rate":   float(successes[i])} for i in range(run.num_seeds)]
  panel = {"eval/avg_cum_reward": rewards.mean(),
           "eval/avg_length":     np.mean(lengths),
           "eval/success_rate":   np.mean(successes)}
  log.log_per_seed(per_seed, panel=panel)
  for i, seed_best in enumerate(best):
    log.summary("best_reward", float(seed_best), seed_index=i)
  log.message(f"Eval done  @ env_step {log.get_step('env_steps')} "
              f"in {time.perf_counter() - started:.1f}s")
  return best


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------


def train(run: Run):
  cfg, num_seeds = run.cfg, run.num_seeds
  now  = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  name = f"{run.model_name.lower()}/{run.env_name.lower()}/{now}"
  if num_seeds > 1:
    name += f"_x{num_seeds}"

  n_videos  = cfg["eval_runs"] if cfg.get("eval_videos") is None else min(cfg["eval_videos"], cfg["eval_runs"])
  
  needs_unrendered = cfg["eval_runs"] > n_videos or num_seeds > 1
  eval_envs = (run.environment.get_vector_env(cfg["eval_runs"]) if needs_unrendered else None,
               run.environment.get_vector_env(n_videos, asynchronous=False) if n_videos else None)

  tags = [run.model_name, run.env_name, "online", "jax"]
  if num_seeds > 1:
    tags.append("multiseed")
  if cfg.get("sweep_tag"):
    tags.append(cfg["sweep_tag"])

  log = Logger(
    "BA-RL", run_name=name, step_keys={"env_steps", "grad_steps"}, tags=tags,
    config={**cfg, "backend": "jax", "num_seeds": num_seeds, "base_seed": run.base_seed,
            "resolved_model": cfg.get("models") or model_config.snapshot_for(run.model_name),
            **{f"resolved/{k}": v for k, v in model_config.resolved_flat(run.model_name).items()},
            **{f"env_stats/{k}": v for k, v in run.env_stats.static().items()
               if not isinstance(v, np.ndarray)},
            "env_obs": str(run.env.observation_space), "env_act": str(run.env.action_space)},
    group=name, seeds=[run.base_seed + i for i in range(num_seeds)]
  )

  log.register("eval/*", "env_steps")
  log.register("rollout/*", "env_steps")
  log.register("model/*", "grad_steps")
  for ns in [*run.reference.components, run.reference.algo_name(), "batch", "buffer"]:
    log.register(f"{ns}/*", "grad_steps")

  # A sweep hands every run its own directory (scripts/run_sweep.py) and its
  # weights belong next to the config and log that produced them. A standalone
  # run has no such directory, and keeps the global checkpoints/ tree keyed by
  # the same <model>/<env>/<timestamp> the wandb run is named after.
  ckpt_dir = os.path.abspath(os.path.join(cfg["run_dir"], "checkpoints") if cfg.get("run_dir")
                             else os.path.join("checkpoints", name))
  os.makedirs(ckpt_dir, exist_ok=True)
  log.message(f"Checkpoints: {ckpt_dir}")
  if num_seeds > 1:
    log.message(f"{num_seeds} seeds vmapped in one process "
                f"({num_seeds * run.num_envs} rollout envs, "
                f"{run.reference.num_params() / 1e6:.2f}M params each)")

  steps_per_iter = cfg["rollout_steps"] // run.num_envs

  if cfg.get("random_steps", 0) > 0:
    run.env_stats["live"].update(
      run.buffer.random_fill(run.rollout_env, n_steps=cfg["random_steps"] // run.num_envs,
                             seed=run.base_seed))
    log.message(f"Random-filled the buffer with {cfg['random_steps']} env steps before training")

  best = [float("-inf")] * num_seeds
  eval_every, next_eval = cfg["eval_every"], cfg["eval_every"]
  it = 0
  while log.get_step("env_steps") < cfg["total_env_steps"]:
    actor = StackedPolicy(run.graphdef, run.state, num_seeds, run.num_envs, run.reference)
    rollout_stats = run.buffer.rollout(actor, env=run.rollout_env, n_steps=steps_per_iter,
                                       seed=run.base_seed if it == 0 else None)
    log.advance({"env_steps": cfg["rollout_steps"]})

    log.log_per_seed([{f"rollout/{k}": v for k, v in seed.items()}
                      for seed in run.buffer.last_rollout_per_seed()],
                     panel={f"rollout/{k}": v for k, v in rollout_stats.items()})
   
    run.env_stats["live"].update(rollout_stats)

    grad_steps_this_iter = round(cfg["rollout_steps"] * cfg["utd_ratio"])
    log.message(f"Rollout done: {run.buffer.n_chunks()} chunks, "
                f"{run.buffer.min_transitions()} transitions per seed, "
                f"~{grad_steps_this_iter} grad steps this iter")
    log.log({"buffer/utd_ratio": grad_steps_this_iter / cfg["rollout_steps"]})
    log.log_per_seed([{"buffer/transitions": n} for n in run.buffer.transitions_per_seed()],
                     panel={"buffer/transitions": run.buffer.min_transitions()})

    epoch_start = time.perf_counter()
    for _ in range(grad_steps_this_iter):
      gstep = log.get_step("grad_steps")
      tiers = (tuple([PRIMARY] if gstep % cfg["log_every"] == 0 else [])
               + tuple([DEBUG] if gstep % cfg["debug_every"] == 0 else []))

      batch = run.buffer.sample(cfg["batch_size"])
      run.state, _, primary, debug = seeded_step(run.graphdef, run.state, batch, tiers)

      if tiers:
        primary, debug = to_host(primary), to_host(debug)
        primary_mean, primary_seeds = split_metrics(primary, num_seeds)
        _,            debug_seeds   = split_metrics(debug, num_seeds)
      
        log.log_per_seed(primary_seeds, debug_seeds, panel=primary_mean)

      log.advance({"grad_steps": 1})

    epoch_time = time.perf_counter() - epoch_start

    eval_time = 0.0
    if log.get_step("env_steps") >= next_eval:
      eval_start = time.perf_counter()
      best = evaluate(run, eval_envs, log, ckpt_dir, best)
      eval_time = time.perf_counter() - eval_start
      next_eval = (log.get_step("env_steps") // eval_every + 1) * eval_every
    
    n_samples = max(1, grad_steps_this_iter * cfg["batch_size"] * num_seeds)
    stats = {"model/num_params": run.reference.num_params(),
             "model/epoch_time_s": epoch_time,
             "model/eval_time_s": eval_time,
             "model/epoch_time_per_sample_ms": 1000 * epoch_time / n_samples}
    if num_seeds == 1:
      stats["model/inference_latency_ms"] = run.reference.measure_latency()
    log.log(stats)
    it += 1

  for i in range(num_seeds):
    policy, _ = unstack(run.graphdef, run.state, i)
    policy.save(os.path.join(_checkpoint_dir(run, ckpt_dir, i), "final"))

  close(run, eval_envs)
  log.finish()


def validate(run: Run):
  """Cheap smoke test for `--validate` sweeps: one rollout chunk + one
  train_step + optimizer step, no eval/checkpointing/wandb logging."""
  cfg = run.cfg
  actor = StackedPolicy(run.graphdef, run.state, run.num_seeds, run.num_envs, run.reference)
  run.buffer.rollout(actor, env=run.rollout_env, n_steps=cfg["batch_size"],
                     seed=run.base_seed)

  batch = run.buffer.sample(cfg["batch_size"])
  run.state, losses, primary, debug = seeded_step(run.graphdef, run.state, batch,
                                                  (PRIMARY, DEBUG))
  losses = jax.device_get(losses)

  for key, value in sorted(losses.items()):
    if not np.all(np.isfinite(value)):
      raise ValueError(f"{run.model_name} on {run.env_name}: loss {key!r} is {value}")

  shown = "  ".join(f"{k}={float(np.mean(v)):+.4f}" for k, v in sorted(losses.items()))
  print(f"  validate OK: {shown}  "
        f"({len(primary[0])} primary / {len(debug[0])} debug metrics)")
  close(run)


def close(run: Run, eval_envs=()):
  run.env.close()
  run.rollout_env.close()
  for e in eval_envs:
    if e is not None:
      e.close()
