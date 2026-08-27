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
import hashlib
import json
import os
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import barl_jax.model_config as model_config
from barl_jax.env import Environment
from barl_jax.logger import Logger
from barl_jax.metrics import DEBUG, PRIMARY, to_host
from barl_jax.multiseed import MultiSeedReplay, StackedPolicy, split_metrics, stack, unstack
from barl_jax.policy import TRAINABLE, Policy, run_episodes
from barl_jax.train import train_step

# `grad_step` rides in at in_axes=None: it is the same number for every seed, and
# a per-seed copy of it would batch the `policy_delay` predicate, which turns the
# lax.cond guarding a delayed actor loss into a select over both branches.
seeded_step = jax.jit(jax.vmap(train_step, in_axes=(None, 0, 0, None, None)),
                      static_argnums=(0, 3), donate_argnums=(1,))

# Longest the training loop stays silent before saying it is still alive.
PROGRESS_INTERVAL_S = 300

# The only thing that reaches wandb *during* a phase that can run for minutes.
# A killed run uploads no term.err, so without this the dashboard cannot tell
# "died between iterations" from "died inside the eval wave".
PHASE_TRAIN, PHASE_EVAL_WAVE, PHASE_EVAL_CKPT, PHASE_EVAL_VIDEO = 0, 1, 2, 3


class Run:
  """Everything a run needs, built once: envs, buffer, stacked policies, logger.

  Constructed by `setup()` so that `train()` and `validate()` share it rather
  than each rebuilding the world."""

  def __init__(self, cfg, num_seeds):
    self.cfg = cfg
    self.num_seeds = num_seeds
    self.model_name, self.env_name = cfg["model"], cfg["env"]
    self.base_seed = cfg["seed"]
    self.num_envs = cfg["num_envs"]


def setup(cfg: dict, num_seeds: int, *, validate_only: bool = False) -> Run:
  run = Run(cfg, num_seeds)
  np.random.seed(run.base_seed)

  # Unconditional: also resets anything left over from a previous run in this
  # process's `runs:` series when the current run doesn't carry one.
  model_config.set_snapshot(cfg.get("snapshot"))
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
  run.buffer = MultiSeedReplay(num_seeds, num_envs, cfg["buffer_size"])

  checkpoint = cfg["checkpoint"]
  if checkpoint is None:
    policies = [Policy.make(run.model_name, run.env, env_stats=run.env_stats,
                            seed=run.base_seed + i) for i in range(num_seeds)]
  elif num_seeds > 1:
    raise NotImplementedError(
      f"checkpoint={checkpoint!r}: a checkpoint holds one seed's state, so it cannot "
      f"resume a {num_seeds}-seed vmapped run -- restart with num_seeds: 1.")
  elif not os.path.exists(checkpoint):
    raise FileNotFoundError(f"checkpoint={checkpoint!r}: no such directory")
  else:
    # load_any, not Policy.load(run.model_name, ...): the meta sidecar carries the
    # registered class name, so resuming under a `model:` alias resolves to the
    # policy class that was saved instead of missing from the registry.
    policies = [Policy.load_any(checkpoint, run.env, run.env_stats)]

  if policies[0].on_policy:
    raise NotImplementedError(
      f"{run.model_name} is on-policy, and the GAE path is not ported -- barl_jax "
      f"currently implements the off-policy loop only.")

  total_grad_steps = max(1, round(cfg["total_env_steps"] * cfg["utd_ratio"]))
  schedule = Policy.lr_schedule(cfg["lr_warmup_steps"], total_grad_steps, cfg["lr_decay"])
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


def evaluate(run: Run, envs, log, ckpt_dir, best, *, want_video: bool):
  """Every seed in one vmapped pass, plus seed 0's video.

  Returns `(best, timings)`; the three phases are timed apart because they fail
  and stall for unrelated reasons -- the wave is sequential physics, the saves
  are orbax writes, the video is an EGL render plus an mp4 encode."""
  cfg = run.cfg
  eval_env, render_env = envs
  rewards, lengths, successes = [], [], []

  # Entry/exit are logged because an eval is the one phase that can stall for
  # tens of minutes with nothing on stdout, and telling "stalled" from "running
  # normally, just slow" after the fact otherwise means reading a stack trace.
  started = time.perf_counter()
  log.message(f"Eval start @ env_step {log.get_step('env_steps')} "
              f"(every {cfg['eval_every']}), {run.num_seeds} seeds x {cfg['eval_runs']} episodes")

  # Every seed's episodes in one vmapped pass, the same way the rollout collects
  # them: `eval_env` is `num_seeds * eval_runs` wide and slot j belongs to seed
  # j // eval_runs. Looping seeds instead cost N separate 1000-step loops, and
  # an episode is a sequential chain of device round-trips, so that N was paid
  # in full -- 29.7 s of a 39 s eval at 3 seeds.
  per_run  = cfg["eval_runs"]
  n_videos = per_run if cfg["eval_videos"] is None else min(cfg["eval_videos"], per_run)
  actor = StackedPolicy(run.graphdef, run.state, run.num_seeds, per_run, run.reference)

  # `eval/phase` is the one thing that reaches wandb *during* an eval, and it is
  # here because a killed run uploads no term.err: without it "died before the
  # eval" and "died inside the eval" are the same picture from the dashboard,
  # which is exactly where sweep 2026-08-26_17-59-33's DIME/DogRun task left off.
  log.log({}, quiet={"eval/phase": PHASE_EVAL_WAVE})
  t0 = time.perf_counter()
  all_runs = run_episodes(actor, eval_env, run.num_seeds * per_run, render=False)
  wave_s = time.perf_counter() - t0

  log.log({}, quiet={"eval/phase": PHASE_EVAL_CKPT})
  t0 = time.perf_counter()
  for i in range(run.num_seeds):
    runs = all_runs[i * per_run:(i + 1) * per_run]
    reward = float(np.mean([np.sum(r["rewards"]) for r in runs]))
    rewards.append(reward)
    lengths.append(np.mean([len(r["rewards"]) for r in runs]))
    successes.append(np.mean([float(r["success"]) for r in runs]))

    policy, _ = unstack(run.graphdef, run.state, i)
    root = _checkpoint_dir(run, ckpt_dir, i)
    policy.save(os.path.join(root, "latest"))
    if reward > best[i]:
      best[i] = reward
      policy.save(os.path.join(root, "best"))

  ckpt_s = time.perf_counter() - t0

  # Only seed 0's video is ever logged, so only seed 0 renders. It is an *extra*
  # episode rather than one of the counted `eval_runs`: it cannot join the
  # vmapped pass (a VectorEnv renders all its sub-envs or none, at ~140x the env
  # step), and holding every seed to the same episode count is what keeps the
  # per-seed numbers comparable.
  video_s = 0.0
  if n_videos and want_video:
    log.log({}, quiet={"eval/phase": PHASE_EVAL_VIDEO})
    t0 = time.perf_counter()
    policy, _ = unstack(run.graphdef, run.state, 0)
    for v, r in enumerate(policy._run_episodes(render_env, n_videos, render=True)):
      log.log_video(f"eval/video_{v}", r["frames"])
    video_s = time.perf_counter() - t0

  # Eval is already per-seed -- each seed owns its own block of slots above --
  # so it goes straight to that seed's run rather than being averaged first.
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
              f"in {time.perf_counter() - started:.1f}s "
              f"(wave {wave_s:.1f}s, ckpt {ckpt_s:.1f}s, video {video_s:.1f}s)")
  return best, {"time/eval_wave_s":  wave_s,
                "time/eval_ckpt_s":  ckpt_s,
                "time/eval_video_s": video_s}


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------


def wandb_config(run: Run) -> dict:
  """Everything that determines this run, flattened for the wandb config panel.

  Assembled from the resolved config rather than from any YAML layer, so what the
  panel shows *is* what ran: `model/*` carries every component kwarg including the
  ones that used to sit at a Python default, and `run/*` every run key -- there is
  no `cfg.get(key, default)` left anywhere for a value to hide behind. `snapshot`
  is the same thing nested, kept so a finished run can be replayed exactly."""
  cfg = run.cfg
  snapshot = cfg.get("snapshot") or model_config.snapshot_for(run.model_name)
  resolved = snapshot["resolved"]

  env_stats = {}
  for key, value in run.env_stats.static().items():
    env_stats[key] = value.tolist() if isinstance(value, np.ndarray) else value

  config = {f"run/{k}": v for k, v in cfg.items() if k not in ("snapshot", "model_overrides")}
  config.update({f"model/{k}": v for k, v in resolved.items()})
  config.update({f"model/derived.{k}": v for k, v in run.reference.derived_config().items()})
  config.update({f"env/{k}": v for k, v in env_stats.items()})
  config.update({
    "model": run.model_name, "env": run.env_name, "backend": "jax",
    "num_seeds": run.num_seeds, "base_seed": run.base_seed,
    "env/obs_space": str(run.env.observation_space),
    "env/act_space": str(run.env.action_space),
    "env/async_vector_env": run.environment.use_async_vector_env,
    "meta/git_sha": _git_sha(), "meta/config_hash": _config_hash(resolved, cfg),
    "snapshot": snapshot,
  })
  return config


def _git_sha() -> str:
  """The commit a run was launched from, plus a `-dirty` marker -- the other half
  of reproducibility that the config itself cannot carry."""
  try:
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                         text=True, timeout=5, cwd=os.path.dirname(__file__))
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                           text=True, timeout=5, cwd=os.path.dirname(__file__))
    if sha.returncode:
      return "unknown"
    return sha.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
  except Exception:
    return "unknown"


def _config_hash(resolved: dict, cfg: dict) -> str:
  """Stable digest of everything that defines this run except the seed, so
  "same config, different seed" is one group in wandb."""
  payload = {**{f"model.{k}": v for k, v in resolved.items()},
             **{f"run.{k}": v for k, v in cfg.items()
                if k not in ("seed", "num_seeds", "run_dir", "sweep_tag", "snapshot", "checkpoint")}}
  return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


def warmup(run: Run, envs, log) -> dict:
  """Pay the first eval's one-off costs at startup instead of seven hours in.

  Two things happen at the first eval and nowhere earlier, and both can take
  minutes or die without a python-level traceback:

  * `_stacked_infer` is static on `num_envs`, so the eval wave's batch shape
    (num_seeds x eval_runs) is one XLA compile the training shape never triggers;
  * mujoco builds its offscreen GL context lazily on the first `render()`
    (`dm_control/mujoco/engine.py`), and dog.xml is the only DMC model asking for
    a 1920x1080 framebuffer -- allocated after jax has already taken its share of
    the device.

  Neither is avoidable, but both are cheap to move to minute one, where the run
  has lost nothing and the failure is unambiguous."""
  cfg = run.cfg
  eval_env, render_env = envs
  out = {}

  t0 = time.perf_counter()
  actor = StackedPolicy(run.graphdef, run.state, run.num_seeds, cfg["eval_runs"], run.reference)
  probe = jnp.zeros((run.num_seeds * cfg["eval_runs"], run.env_stats["obs_dim"]), jnp.float32)
  jax.block_until_ready(actor.inference_fn()(probe, jax.random.PRNGKey(0)))
  out["time/warmup_compile_s"] = time.perf_counter() - t0

  if render_env is not None:
    t0 = time.perf_counter()
    render_env.reset(seed=run.base_seed)
    render_env.call("render")
    out["time/warmup_render_s"] = time.perf_counter() - t0

  log.message("Warm-up: eval-shape compile {:.1f}s, first render {:.1f}s".format(
    out["time/warmup_compile_s"], out.get("time/warmup_render_s", 0.0)))
  return out


def train(run: Run):
  cfg, num_seeds = run.cfg, run.num_seeds
  now  = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  name = f"{run.model_name.lower()}/{run.env_name.lower()}/{now}"
  if num_seeds > 1:
    name += f"_x{num_seeds}"

  # `eval_video_every: null` means never, so the render env is not built either
  # -- one fewer mujoco model resident, and no GL context to create at all.
  n_videos  = 0 if cfg["eval_video_every"] is None else (
    cfg["eval_runs"] if cfg["eval_videos"] is None else min(cfg["eval_videos"], cfg["eval_runs"]))

  # `num_seeds * eval_runs` wide so every seed's episodes run in one vmapped
  # pass -- still narrower than the rollout env, which is num_seeds * num_envs.
  eval_envs = (run.environment.get_vector_env(num_seeds * cfg["eval_runs"]),
               run.environment.get_vector_env(n_videos, asynchronous=False) if n_videos else None)

  tags = [run.model_name, run.env_name, "online", "jax"]
  if num_seeds > 1:
    tags.append("multiseed")
  if cfg["sweep_tag"]:
    tags.append(cfg["sweep_tag"])

  log = Logger(
    "BA-RL", run_name=name, step_keys={"env_steps", "grad_steps"}, tags=tags,
    config=wandb_config(run), group=name,
    finish_timeout=cfg["wandb_finish_timeout"],
    seeds=[run.base_seed + i for i in range(num_seeds)]
  )

  log.register("eval/*", "env_steps")
  log.register("rollout/*", "env_steps")
  log.register("model/*", "grad_steps")
  log.register("time/*", "env_steps")
  for ns in [*run.reference.components, run.reference.algo_name(), "batch", "buffer"]:
    log.register(f"{ns}/*", "grad_steps")

  # A sweep hands every run its own directory (scripts/grid.py) and its
  # weights belong next to the config and log that produced them. A standalone
  # run has no such directory, and keeps the global checkpoints/ tree keyed by
  # the same <model>/<env>/<timestamp> the wandb run is named after.
  ckpt_dir = os.path.abspath(os.path.join(cfg["run_dir"], "checkpoints") if cfg["run_dir"]
                             else os.path.join("checkpoints", name))
  os.makedirs(ckpt_dir, exist_ok=True)
  log.message(f"Checkpoints: {ckpt_dir}")
  if num_seeds > 1:
    log.message(f"{num_seeds} seeds vmapped in one process "
                f"({num_seeds * run.num_envs} rollout envs, "
                f"{run.reference.num_params() / 1e6:.2f}M params each)")

  warmup_stats = warmup(run, eval_envs, log)

  steps_per_iter = cfg["rollout_steps"] // run.num_envs

  if cfg["random_steps"] > 0:
    run.env_stats["live"].update(
      run.buffer.random_fill(run.rollout_env, n_steps=cfg["random_steps"] // run.num_envs,
                             seed=run.base_seed))
    log.message(f"Random-filled the buffer with {cfg['random_steps']} env steps before training")

  best = [float("-inf")] * num_seeds
  eval_every, next_eval = cfg["eval_every"], cfg["eval_every"]
  video_every, next_video = cfg["eval_video_every"], cfg["eval_video_every"]
  it = 0
  run_started = time.perf_counter()
  # `rollout_steps` may be as small as `num_envs` (one env step per slot per
  # iteration), which is thousands of iterations per 100k env steps -- so the
  # per-iteration narration is paced by wall clock rather than emitted every
  # time round. wandb still gets every iteration; only stdout is thinned.
  last_narration = 0.0
  # Without this a mid-loop exception skipped the final save, left the eval envs
  # to the interpreter and never flushed wandb, so the failure was invisible from
  # the dashboard. A fatal signal still bypasses it -- that is faulthandler's job.
  crash = None
  try:
    while log.get_step("env_steps") < cfg["total_env_steps"]:
      iter_start = time.perf_counter()
      actor = StackedPolicy(run.graphdef, run.state, num_seeds, run.num_envs, run.reference)
      rollout_stats = run.buffer.rollout(actor, env=run.rollout_env, n_steps=steps_per_iter,
                                         seed=run.base_seed if it == 0 else None)
      rollout_time = time.perf_counter() - iter_start
      log.advance({"env_steps": cfg["rollout_steps"]})

      log.log_per_seed([{f"rollout/{k}": v for k, v in seed.items()}
                        for seed in run.buffer.last_rollout_per_seed()],
                       panel={f"rollout/{k}": v for k, v in rollout_stats.items()})
   
      run.env_stats["live"].update(rollout_stats)

      grad_steps_this_iter = round(cfg["rollout_steps"] * cfg["utd_ratio"])
      narrate = (it == 0 or time.perf_counter() - last_narration > PROGRESS_INTERVAL_S)
      if narrate:
        log.message(f"Rollout done: {run.buffer.min_transitions()} transitions per seed, "
                    f"~{grad_steps_this_iter} grad steps this iter")
      log.log({"buffer/utd_ratio": grad_steps_this_iter / cfg["rollout_steps"]})
      log.log_per_seed([{"buffer/transitions": n} for n in run.buffer.transitions_per_seed()],
                       panel={"buffer/transitions": run.buffer.min_transitions()})

      # Same reason `evaluate()` announces itself: this is a phase that runs for
      # hours with nothing on stdout between it and the next rollout, and after
      # the fact a slow step is indistinguishable from a stall. `log_every` paces
      # wandb, which at a few seconds per step can be a long way apart.
      epoch_start = last_report = time.perf_counter()
      progress_every = max(1, grad_steps_this_iter // 10)

      for step in range(grad_steps_this_iter):
        gstep = log.get_step("grad_steps")
        tiers = (tuple([PRIMARY] if gstep % cfg["log_every"] == 0 else [])
                 + tuple([DEBUG] if gstep % cfg["debug_every"] == 0 else []))

        batch = run.buffer.sample(cfg["batch_size"])
        run.state, _, primary, debug = seeded_step(run.graphdef, run.state, batch, tiers,
                                                   jnp.asarray(gstep, jnp.int32))

        if tiers:
          primary, debug = to_host(primary), to_host(debug)
          primary_mean, primary_seeds = split_metrics(primary, num_seeds)
          _,            debug_seeds   = split_metrics(debug, num_seeds)
      
          log.log_per_seed(primary_seeds, debug_seeds, panel=primary_mean)

        log.advance({"grad_steps": 1})

        now, done = time.perf_counter(), step + 1
        if step == 0 and it == 0:
          # Reported on its own because it is not a step time: every distinct
          # `tiers` value is a separate jit trace, so the first few steps each pay
          # a compile the rest do not.
          log.message(f"  first grad step (incl. compile): {now - epoch_start:.1f}s")
          last_report = now
        # Whichever comes first: a tenth of the iteration, or PROGRESS_INTERVAL_S
        # of silence. The second is what a slow step needs -- a tenth of a
        # 10k-step iteration at seconds per step is an hour apart. Neither is
        # worth saying for an iteration that lasts a fraction of a second.
        elif narrate and done < grad_steps_this_iter and (done % progress_every == 0
                                              or now - last_report > PROGRESS_INTERVAL_S):
          rate = done / (now - epoch_start)
          log.message(f"  {done}/{grad_steps_this_iter} grad steps, {rate:.2f}/s, "
                      f"~{(grad_steps_this_iter - done) / rate / 60:.1f} min left this iter")
          last_report = now

      epoch_time = time.perf_counter() - epoch_start
      if narrate:
        log.message(f"Trained {grad_steps_this_iter} grad steps in {epoch_time / 60:.1f} min "
                    f"({1000 * epoch_time / grad_steps_this_iter:.0f} ms/step)")
        last_narration = time.perf_counter()

      eval_time, eval_phases = 0.0, {}
      if log.get_step("env_steps") >= next_eval:
        # A rendered episode is ~140x the cost of an env step plus a synchronous
        # mp4 encode, and 40 of them per run is an hour on the heavier models --
        # so the video runs on its own, coarser cadence than the eval itself.
        want_video = video_every is not None and log.get_step("env_steps") >= next_video
        eval_start = time.perf_counter()
        best, eval_phases = evaluate(run, eval_envs, log, ckpt_dir, best, want_video=want_video)
        eval_time = time.perf_counter() - eval_start
        next_eval = (log.get_step("env_steps") // eval_every + 1) * eval_every
        if want_video:
          next_video = (log.get_step("env_steps") // video_every + 1) * video_every
    
      n_samples = max(1, grad_steps_this_iter * cfg["batch_size"] * num_seeds)
      iter_time = time.perf_counter() - iter_start
      elapsed   = max(1e-9, time.perf_counter() - run_started)
      stats = {"model/num_params": run.reference.num_params(),
               "model/epoch_time_s": epoch_time,
               "model/eval_time_s": eval_time,
               "model/epoch_time_per_sample_ms": 1000 * epoch_time / n_samples,
               # The split the last sweep could not produce: which phase the wall
               # clock actually went into, per iteration, and the rate the 8-hour
               # budget is judged against.
               "time/rollout_s": rollout_time,
               "time/train_s":   epoch_time,
               "time/iter_s":    iter_time,
               "time/env_steps_per_hour": 3600.0 * log.get_step("env_steps") / elapsed,
               "eval/phase": PHASE_TRAIN,
               **eval_phases, **warmup_stats}
      if num_seeds == 1:
        stats["model/inference_latency_ms"] = run.reference.measure_latency()
      log.log(stats)
      it += 1
  except BaseException as exc:
    crash = exc
    raise
  finally:
    try:
      for i in range(num_seeds):
        policy, _ = unstack(run.graphdef, run.state, i)
        policy.save(os.path.join(_checkpoint_dir(run, ckpt_dir, i), "final"))
    except Exception as save_error:
      log.message(f"Could not write the final checkpoint: {save_error!r}")

    if crash is not None:
      log.message(f"Run failed: {type(crash).__name__}: {crash}")
      log.summary("crash", f"{type(crash).__name__}: {crash}")

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
                                                  (PRIMARY, DEBUG), jnp.asarray(0, jnp.int32))
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
