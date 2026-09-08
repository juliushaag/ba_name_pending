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
import math
import os
import subprocess
import time
from contextlib import ExitStack

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import barl_jax.config.tree as model_config
from barl_jax.checkpoints import load_policy, save_policy
from barl_jax.collection import Collector
from barl_jax.components import TRAINABLE
from barl_jax.config.model import experiment_fingerprint
from barl_jax.config.run import validate_run
from barl_jax.env import Environment
from barl_jax.evaluation import run_episodes
from barl_jax.execution import ExecutionSpec, validate_execution_spec
from barl_jax.metrics import DEBUG, PRIMARY, to_host
from barl_jax.multiseed import StackedPolicy, split_metrics, stack, unstack
from barl_jax.optimization import seeded_chunk, seeded_step
from barl_jax.policy import Policy
from barl_jax.result import presentable, write_json_atomic
from barl_jax.rollout import MultiSeedReplay
from barl_jax.seeding import derive_seed
from barl_jax.training_checkpoint import (
  read_training,
  restore_training,
  save_training,
  validate_resume,
)

# Longest the training loop stays silent before saying it is still alive.
PROGRESS_INTERVAL_S = 300


# The only thing that reaches wandb *during* a phase that can run for minutes.
# A killed run uploads no term.err, so without this the dashboard cannot tell
# "died between iterations" from "died inside the eval wave".
PHASE_TRAIN, PHASE_EVAL_WAVE, PHASE_EVAL_CKPT, PHASE_EVAL_VIDEO = 0, 1, 2, 3


def enable_compilation_cache(cache_dir) -> None:
  """Point jax at a persistent compilation cache, before anything compiles.

  jax's own thresholds are already what we want (`min_compile_time_secs` 1.0,
  `min_entry_size_bytes` 0), so the directory is the whole configuration."""
  if not cache_dir:
    return
  path = os.path.abspath(os.path.expanduser(str(cache_dir)))
  os.makedirs(path, exist_ok=True)
  jax.config.update("jax_compilation_cache_dir", path)


class Run:
  """Everything a run needs, built once: envs, buffer, stacked policies, logger.

  Constructed by `setup()` so that `train()` and `validate()` share it rather
  than each rebuilding the world."""

  def __init__(self, cfg, num_seeds):
    self.resources = ExitStack()
    self.cfg = cfg
    self.num_seeds = num_seeds
    self.model_name, self.env_name = cfg["model"], cfg["env"]
    self.base_seed = cfg["seed"]
    self.num_envs = cfg["num_envs"]
    self.progress = {}
    # Evaluation's own identity axis, independent of env_steps/grad_steps --
    # see evaluate()'s eval_seed. Not restored on resume, consistent with
    # resume's existing "environment episodes restart" semantics.
    self.eval_count = 0

  def own(self, resource):
    self.resources.callback(resource.close)
    return resource

  def __enter__(self):
    return self

  def __exit__(self, *exc):
    return self.resources.__exit__(*exc)


def setup(cfg: dict, num_seeds: int, *, validate_only: bool = False) -> Run:
  run = Run(cfg, num_seeds)
  try:
    return _setup(run, validate_only=validate_only)
  except BaseException:
    run.resources.close()
    raise


def _setup(run, *, validate_only):
  cfg, num_seeds = run.cfg, run.num_seeds

  run_problems = validate_run(cfg, num_seeds=num_seeds)
  if run_problems:
    raise ValueError(f"{run.model_name} on {run.env_name}: invalid run config, refusing to start:\n  "
                     + "\n  ".join(run_problems))

  enable_compilation_cache(cfg["compile_cache_dir"])

  mode = cfg.get("checkpoint_mode", "weights")
  resume_meta = None
  if mode == "resume":
    if cfg["checkpoint"] is None:
      raise ValueError("checkpoint_mode: resume requires checkpoint")
    resume_meta = read_training(cfg["checkpoint"])
    validate_resume(cfg, num_seeds, resume_meta)
    cfg = run.cfg = {**cfg, **{key: resume_meta["config"].get(key)
                            for key in ("snapshot", "model_overrides")}}
  if cfg["checkpoint"] and mode == "weights":
    from barl_jax.checkpoints import policy_spec
    saved_spec, _ = policy_spec(cfg["checkpoint"])
    cfg = run.cfg = {**cfg, "snapshot": saved_spec.data()["snapshot"],
                     "model_overrides": saved_spec.data()["overrides"]}
  run.config = model_config.ConfigTree(snapshot=cfg.get("snapshot"), overrides=cfg.get("model_overrides"))
  problems = model_config.validate(run.model_name, run.config)
  if problems:
    raise ValueError(f"{run.model_name}: unusable model config, refusing to start:\n  "
                     + "\n  ".join(problems))

  run.environment = Environment.make(run.env_name)
  run.env = run.own(run.environment.get_env())       # single env: spaces and EnvStats
  run.env_stats = run.environment.get_env_stats(run.env)

  num_envs = 1 if validate_only else run.num_envs
  run.num_envs = num_envs
  run.rollout_env = run.own(run.environment.get_vector_env(num_seeds * num_envs))
  # A seed's buffer never receives more than total_env_steps transitions --
  # preallocating past that just reserves address space (and, as training
  # fills it in, real RAM) no run will ever use.
  buffer_size = min(cfg["buffer_size"], cfg["total_env_steps"])
  run.collector = Collector(seed=run.base_seed)
  run.buffer = MultiSeedReplay(num_seeds, num_envs, buffer_size, seed=run.base_seed)

  checkpoint = cfg["checkpoint"]
  if checkpoint is None:
    policies = [Policy.make(run.model_name, run.env, env_stats=run.env_stats,
                            seed=run.base_seed + i, chunk_size=cfg["chunk_size"], config=run.config)
                for i in range(num_seeds)]
  elif not os.path.exists(checkpoint):
    raise FileNotFoundError(f"checkpoint={checkpoint!r}: no such directory")
  elif resume_meta is not None:
    policies = [load_policy(os.path.join(resume_meta["_dir"], f"seed{i}"), run.env, run.env_stats)
                for i in range(num_seeds)]
  else:
    policies = [load_policy(checkpoint, run.env, run.env_stats) for _ in range(num_seeds)]
    for i, policy in enumerate(policies):
      policy.train_steps[...] = jnp.zeros_like(policy.train_steps[...])
      policy.rngs = nnx.Rngs(run.base_seed + i)
      for component in policy.components.values():
        if hasattr(component, "target_steps"):
          component.target_steps[...] = jnp.zeros_like(component.target_steps[...])
    if policies[0].chunk_size != cfg["chunk_size"]:
      raise ValueError("Checkpoint chunk_size differs from the run")

  expected_name, _ = run.config.resolve_model_name(run.model_name)
  if policies[0].model_name != expected_name:
    raise ValueError(f"Checkpoint model {policies[0].model_name!r} differs from {expected_name!r}")

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
  # Constant for the run, and a full walk of the state tree (1.7 ms) -- at
  # `rollout_steps: 1` it was 6% of a single-seed run to recompute it per
  # iteration. Measured once here and logged from the cache.
  if resume_meta is not None:
    restore_training(run, resume_meta)
  run.num_params = run.reference.num_params()

  run.collection_spec = ExecutionSpec(mode=cfg["collection_mode"], num_candidates=cfg["collection_candidates"])
  run.eval_spec = ExecutionSpec(mode=cfg["eval_mode"], num_candidates=cfg["eval_candidates"])
  spec_problems = [f"{label}: {p}" for label, spec in
                   (("collection", run.collection_spec), ("eval", run.eval_spec))
                   for p in validate_execution_spec(spec, policy=run.reference)]
  if spec_problems:
    raise ValueError(f"{run.model_name}: invalid execution spec, refusing to start:\n  "
                     + "\n  ".join(spec_problems))
  return run


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------


def _checkpoint_dir(run: Run, root: str, seed: int) -> str:
  """Every seed's checkpoints live under its own directory, including seed 0."""
  return os.path.join(root, f"seed{seed}")


def evaluate(run: Run, envs, log, ckpt_dir, best, *, want_video: bool):
  """Every seed in one vmapped pass, plus seed 0's video.

  Returns `(best, timings)`; the three phases are timed apart because they fail
  and stall for unrelated reasons -- the wave is sequential physics, the saves
  are orbax writes, the video is an EGL render plus an mp4 encode."""
  cfg = run.cfg
  eval_env, render_env = envs
  rewards, lengths, successes, nonfinite = [], [], [], []

  # Stable per-evaluation identity, independent of training RNG (this reads
  # no policy/collector/replay RNG state, so eval never perturbs a training
  # trajectory) -- same run seed and eval_count always reproduce the same
  # episodes; a later eval_count never replays an earlier one's.
  eval_seed = derive_seed(run.base_seed, "eval", run.eval_count)
  run.eval_count += 1

  # Entry/exit are logged because an eval is the one phase that can stall for
  # tens of minutes with nothing on stdout, and telling "stalled" from "running
  # normally, just slow" after the fact otherwise means reading a stack trace.
  started = time.perf_counter()
  log.message(f"Eval start @ env_step {log.get_step('env_steps')} "
              f"(every {cfg['eval_every']}), {run.num_seeds} seeds x {cfg['eval_runs']} episodes, "
              f"eval_seed={eval_seed}")

  # One slot per seed; run_episodes reuses these slots for successive rounds.
  per_run  = cfg["eval_runs"]
  n_videos = per_run if cfg["eval_videos"] is None else min(cfg["eval_videos"], per_run)
  actor = StackedPolicy(run.graphdef, run.state, run.num_seeds, 1, run.reference,
                        execution_spec=run.eval_spec)

  # `eval/phase` is the one thing that reaches wandb *during* an eval, and it is
  # here because a killed run uploads no term.err: without it "died before the
  # eval" and "died inside the eval" are the same picture from the dashboard,
  # which is exactly where sweep 2026-08-26_17-59-33's DIME/DogRun task left off.
  log.log({}, quiet={"eval/phase": PHASE_EVAL_WAVE})
  t0 = time.perf_counter()
  all_runs = run_episodes(actor, eval_env, run.num_seeds * per_run, render=False, seed=eval_seed)
  wave_s = time.perf_counter() - t0

  log.log({}, quiet={"eval/phase": PHASE_EVAL_CKPT})
  t0 = time.perf_counter()
  for i in range(run.num_seeds):
    # Results are round-major: seed 0, seed 1, ... for each round.
    runs = all_runs[i::run.num_seeds]
    reward = float(np.mean([np.sum(r["rewards"]) for r in runs]))
    rewards.append(reward)
    lengths.append(np.mean([len(r["rewards"]) for r in runs]))
    successes.append(np.mean([float(r["success"]) for r in runs]))
    seen = sum(r["n_act"] for r in runs)
    nonfinite.append(sum(r["act_nonfinite"] for r in runs) / seen if seen else 0.0)

    policy, _ = unstack(run.graphdef, run.state, i)
    root = _checkpoint_dir(run, ckpt_dir, i)
    save_policy(policy, os.path.join(root, "latest"))
    if reward > best[i]:
      best[i] = reward
      save_policy(policy, os.path.join(root, "best"))

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
    video_seed = derive_seed(run.base_seed, "eval-video", run.eval_count - 1)
    for v, r in enumerate(run_episodes(policy, render_env, n_videos, render=True, seed=video_seed)):
      log.log_video(f"eval/video_{v}", r["frames"])
    video_s = time.perf_counter() - t0

  # Eval is already per-seed -- each seed owns one slot across all rounds --
  # so it goes straight to that seed's run rather than being averaged first.
  rewards = np.asarray(rewards)
  per_seed = [{"eval/avg_cum_reward": float(rewards[i]),
               "eval/avg_length":     float(lengths[i]),
               "eval/success_rate":   float(successes[i]),
               "eval/act_nonfinite":  float(nonfinite[i])} for i in range(run.num_seeds)]
  panel = {"eval/avg_cum_reward": rewards.mean(),
           "eval/avg_length":     np.mean(lengths),
           "eval/success_rate":   np.mean(successes),
           "eval/act_nonfinite":  np.mean(nonfinite)}
  log.log_per_seed(per_seed, panel=panel)

  # The counterpart to `rollout/act_nonfinite`, and said out loud rather than
  # only plotted: a non-finite action is now clamped on both paths, so the run
  # survives it, but every number this eval just produced came from an actor
  # that had already diverged.
  if any(f > 0 for f in nonfinite):
    worst = int(np.argmax(nonfinite))
    log.message(f"  WARNING: non-finite actions during eval -- seed {worst} "
                f"{nonfinite[worst]:.3%} of action components, "
                f"{sum(f > 0 for f in nonfinite)}/{run.num_seeds} seeds affected. "
                f"They were clamped to 0; this eval's numbers are not meaningful.")
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


def run_config_snapshot(run: Run) -> dict:
  """Everything that determines this run, flattened for the logger's config
  panel (wandb's, or whatever a future local equivalent shows).

  Assembled from the resolved config rather than from any YAML layer, so what the
  panel shows *is* what ran: `model/*` carries every component kwarg including the
  ones that used to sit at a Python default, and `run/*` every run key -- there is
  no `cfg.get(key, default)` left anywhere for a value to hide behind. `snapshot`
  is the same thing nested, kept so a finished run can be replayed exactly."""
  cfg = run.cfg
  snapshot = run.reference._snapshot
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
    "meta/git_sha": _git_sha(),
    "meta/config_hash": experiment_fingerprint(resolved, cfg),
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


def warmup(run: Run, envs, log) -> dict:
  """Pay the first eval's one-off costs at startup instead of seven hours in.

  Two things happen at the first eval and nowhere earlier, and both can take
  minutes or die without a python-level traceback:

  * `_stacked_infer` is static on `num_envs`, so the eval wave's batch shape
    (one environment per seed) can differ from the training shape;
  * mujoco builds its offscreen GL context lazily on the first `render()`
    (`dm_control/mujoco/engine.py`), and dog.xml is the only DMC model asking for
    a 1920x1080 framebuffer -- allocated after jax has already taken its share of
    the device.

  Neither is avoidable, but both are cheap to move to minute one, where the run
  has lost nothing and the failure is unambiguous."""
  eval_env, render_env = envs
  out = {}

  t0 = time.perf_counter()
  actor = StackedPolicy(run.graphdef, run.state, run.num_seeds, 1, run.reference,
                        execution_spec=run.eval_spec)
  probe = jnp.zeros((run.num_seeds, run.env_stats["obs_dim"]), jnp.float32)
  jax.block_until_ready(actor.inference_fn()(probe, jax.random.PRNGKey(0)))
  out["time/warmup_compile_s"] = time.perf_counter() - t0

  # Same reason as `num_params`, and worse: this is 11 blocking device
  # round-trips (10.4 ms), and per iteration it was 35% of a single-seed
  # HalfCheetah run. It is a property of the architecture, not of the weights,
  # so one measurement at startup says everything the per-iteration one did.
  out["model/inference_latency_ms"] = run.reference.measure_latency()

  if render_env is not None:
    t0 = time.perf_counter()
    render_env.reset(seed=run.base_seed)
    render_env.call("render")
    out["time/warmup_render_s"] = time.perf_counter() - t0

  log.message("Warm-up: eval-shape compile {:.1f}s, first render {:.1f}s, "
              "inference {:.2f}ms".format(out["time/warmup_compile_s"],
                                          out.get("time/warmup_render_s", 0.0),
                                          out["model/inference_latency_ms"]))
  return out


def train(run: Run):
  with run:
    return _train(run)


def _train(run: Run):
  from barl_jax.logger import make_logger
  cfg, num_seeds = run.cfg, run.num_seeds
  now  = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  name = f"{run.model_name.lower()}/{run.env_name.lower()}/{now}_x{num_seeds}"

  # `eval_video_every: null` means never, so the render env is not built either
  # -- one fewer mujoco model resident, and no GL context to create at all.
  n_videos  = 0 if cfg["eval_video_every"] is None else (
    cfg["eval_runs"] if cfg["eval_videos"] is None else min(cfg["eval_videos"], cfg["eval_runs"]))

  # Keep one simulator per seed resident, independently of the episode count.
  eval_envs = (run.own(run.environment.get_vector_env(num_seeds)),
               run.own(run.environment.get_vector_env(n_videos, asynchronous=False)) if n_videos else None)

  tags = [run.model_name, run.env_name, "online", "jax", "multiseed"]
  if cfg["sweep_tag"]:
    tags.append(cfg["sweep_tag"])

  # A sweep hands every run its own directory (barl_jax.launch.grid) and its
  # weights belong next to the config and log that produced them. A standalone
  # run has no such directory, and keeps the global checkpoints/ tree keyed by
  # the same <model>/<env>/<timestamp> the wandb run is named after.
  ckpt_dir = os.path.abspath(os.path.join(cfg["run_dir"], "checkpoints") if cfg["run_dir"]
                             else os.path.join("checkpoints", name))
  os.makedirs(ckpt_dir, exist_ok=True)
  # The run-level directory the objective sidecar lives in: the run's own dir
  # under a sweep, else the standalone <model>/<env>/<timestamp>/ tree. This is
  # the only place a finished run's result is readable without wandb.
  objective_dir = os.path.abspath(cfg["run_dir"]) if cfg["run_dir"] else os.path.join("checkpoints", name)

  log = make_logger(
    cfg.get("logger", "wandb"), name, step_keys={"env_steps", "grad_steps"}, tags=tags,
    config=run_config_snapshot(run), group=name,
    finish_timeout=cfg["wandb_finish_timeout"], run_dir=objective_dir,
    seeds=[run.base_seed + i for i in range(num_seeds)]
  )

  run.resources.callback(log.finish)

  log.register("eval/*", "env_steps")
  log.register("rollout/*", "env_steps")
  log.register("model/*", "grad_steps")
  log.register("time/*", "env_steps")
  for ns in [*run.reference.components, run.reference.algo_name(), "batch", "buffer"]:
    log.register(f"{ns}/*", "grad_steps")

  log.message(f"Checkpoints: {ckpt_dir}")
  log.message(f"{num_seeds} seed{'s' if num_seeds != 1 else ''} vmapped in one process "
              f"({num_seeds * run.num_envs} rollout envs, "
              f"{run.num_params / 1e6:.2f}M params each)")

  warmup_stats = warmup(run, eval_envs, log)

  steps_per_iter = cfg["rollout_steps"] // run.num_envs

  if cfg["random_steps"] > 0 and not run.progress:
    # Prefill primitive steps are not counted into env_steps/grad_steps -- no
    # log.advance() around this call. Only skipped on resume (`not run.progress`),
    # never re-run against an already-filled buffer.
    run.collector.random_fill(run.rollout_env, n_steps=cfg["random_steps"] // run.num_envs,
                           seed=run.base_seed, chunk_size=run.reference.chunk_size,
                           discount=run.reference.discount, sink=run.buffer)
    log.message(f"Random-filled the buffer with {cfg['random_steps']} env steps before training")

  if run.progress:
    log.advance({key: run.progress[key] for key in ("env_steps", "grad_steps")})
    log.message("Resumed optimizer, replay and counters; environment episodes restart")
  best = run.progress.get("best", [float("-inf")] * num_seeds)
  eval_every, next_eval = cfg["eval_every"], cfg["eval_every"]
  video_every, next_video = cfg["eval_video_every"], cfg["eval_video_every"]
  next_eval = (log.get_step("env_steps") // eval_every + 1) * eval_every
  if video_every is not None:
    next_video = (log.get_step("env_steps") // video_every + 1) * video_every
  # Resume persistence is its own cadence -- independent of eval_every, since
  # the replay dump it carries is what makes a save expensive, not the eval.
  resume_dir = os.path.join(ckpt_dir, "resume")
  resume_every = cfg["resume_every"]
  next_resume = (log.get_step("env_steps") // resume_every + 1) * resume_every
  it = run.progress.get("iteration", 0)
  chunk_seen = False
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
      actor = StackedPolicy(run.graphdef, run.state, num_seeds, run.num_envs, run.reference,
                            execution_spec=run.collection_spec)
      rollout_stats = run.collector.collect(actor, env=run.rollout_env, n_steps=steps_per_iter,
                                         seed=run.base_seed + int(run.progress.get("env_steps", 0))
                                         if run.collector.state is None else None,
                                         discount=actor.discount, sink=run.buffer)
      rollout_time = time.perf_counter() - iter_start
      log.advance({"env_steps": cfg["rollout_steps"]})

      log.log_per_seed([{f"rollout/{k}": v for k, v in seed.items()}
                        for seed in run.collector.per_seed(run.num_seeds, run.num_envs)],
                       panel={f"rollout/{k}": v for k, v in rollout_stats.items()})

      # Chunks can span collection calls: wait for replay data, then catch up
      # to the cumulative primitive-step budget (also preserves fractional UTD).
      grad_steps_this_iter = (
        max(0, round(log.get_step("env_steps") * cfg["utd_ratio"])
            - log.get_step("grad_steps"))
        if run.buffer.min_transitions() > 0 else 0)
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

      # Whole-iteration fast path. An iteration holding no tier-armed step is
      # one `lax.scan` over its whole gradient-step count and one host->device
      # transfer, instead of that many jit dispatches and transfers. Split at
      # the iteration rather than at the armed step so `k` takes exactly one
      # value for the run: a second chunk length is a second compile of the
      # scan, and at DIME's ~60 s per compile that costs more than it saves.
      # An iteration holds an armed step `grad_steps_this_iter / log_every` of
      # the time -- 13% at the 32-step iteration `rollout_steps: 16` gives.
      base = log.get_step("grad_steps")
      armed_here = any((base + i) % cfg["log_every"] == 0
                       or (base + i) % cfg["debug_every"] == 0
                       for i in range(grad_steps_this_iter))

      if grad_steps_this_iter and not armed_here:
        batches = run.buffer.sample_chunk(grad_steps_this_iter, cfg["batch_size"])
        run.state = seeded_chunk(run.graphdef, run.state, batches,
                                 grad_steps_this_iter, jnp.asarray(base, jnp.int32))
        log.advance({"grad_steps": grad_steps_this_iter})
        if not chunk_seen:
          chunk_seen = True
          jax.block_until_ready(run.state)
          log.message(f"  first chunked iteration ({grad_steps_this_iter} grad steps, "
                      f"incl. compile): {time.perf_counter() - epoch_start:.1f}s")

      for step in (range(grad_steps_this_iter) if armed_here else ()):
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
                    f"({1000 * epoch_time / max(1, grad_steps_this_iter):.0f} ms/step)")
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
        _write_objective(objective_dir, model=run.model_name, env=run.env_name,
                         seed=run.base_seed, num_seeds=run.num_seeds,
                         env_steps=int(log.get_step("env_steps")), best=best, status="running")
    
      n_samples = max(1, grad_steps_this_iter * cfg["batch_size"] * num_seeds)
      iter_time = time.perf_counter() - iter_start
      elapsed   = max(1e-9, time.perf_counter() - run_started)
      stats = {"model/num_params": run.num_params,
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
      log.log(stats)
      it += 1
      run.progress = {"env_steps": int(log.get_step("env_steps")),
                      "grad_steps": int(log.get_step("grad_steps")),
                      "iteration": it, "best": list(best)}
      if log.get_step("env_steps") >= next_resume:
        save_training(run, resume_dir, run.progress, keep=cfg["resume_keep"])
        next_resume = (log.get_step("env_steps") // resume_every + 1) * resume_every
  except BaseException as exc:
    crash = exc
    raise
  finally:
    try:
      for i in range(num_seeds):
        policy, _ = unstack(run.graphdef, run.state, i)
        save_policy(policy, os.path.join(_checkpoint_dir(run, ckpt_dir, i), "final"))
    except Exception as save_error:
      log.message(f"Could not write the final checkpoint: {save_error!r}")

    # Clean completion always gets a resume save, off-cadence or not: the run
    # is done, so this is the last chance and it costs nothing to take it.
    if crash is None:
      save_training(run, resume_dir, run.progress, keep=cfg["resume_keep"])

    if crash is not None:
      log.message(f"Run failed: {type(crash).__name__}: {crash}")
      log.summary("crash", f"{type(crash).__name__}: {crash}")

    _write_objective(objective_dir, model=run.model_name, env=run.env_name,
                     seed=run.base_seed, num_seeds=run.num_seeds,
                     env_steps=int(log.get_step("env_steps")), best=best,
                     status="failed" if crash is not None else "finished")




def validate(run: Run):
  with run:
    return _validate(run)


def _validate(run: Run):
  """Cheap smoke test for `--validate` sweeps: one rollout chunk + one
  train_step + optimizer step, no eval/checkpointing/wandb logging."""
  cfg = run.cfg
  actor = StackedPolicy(run.graphdef, run.state, run.num_seeds, run.num_envs, run.reference,
                        execution_spec=run.collection_spec)
  run.collector.collect(actor, env=run.rollout_env, n_steps=cfg["batch_size"],
                     seed=run.base_seed, discount=actor.discount, sink=run.buffer)

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


def _write_objective(objective_dir: str, *, model: str, env: str, seed: int,
                     num_seeds: int, env_steps: int, best: list, status: str) -> str:
  """The run's result as a small JSON sidecar in its own directory, so a sweep
  or an HPO driver can rank finished runs without touching wandb.

  `objective` is the mean best reward across seeds -- the same number
  `best_reward` summarises per seed -- and `best` the per-seed values. Written
  after every eval (status "running", so a live driver can watch it) and again
  in the trainer's finally (status "finished" / "failed").

  Strict presentation JSON (barl_jax.result): a seed with no eval yet is still
  `-inf` in `best` (evaluate()'s own "beat the previous best" comparison needs
  the real sentinel), but here that becomes `null` plus `objective_reason`, so
  a driver reading the file with any JSON parser sees "no result" rather than
  a value it has to know to special-case."""
  finite_best = [float(b) for b in best if math.isfinite(b)]
  objective = float(np.mean(finite_best)) if finite_best else None
  reason = None if finite_best else "no evaluation has completed for any seed yet"
  payload = {
    "model": model, "env": env, "seed": seed, "num_seeds": num_seeds,
    "env_steps": env_steps, "status": status,
    "objective": objective, "objective_reason": reason,
    "best": [presentable(float(b)) for b in best],
  }
  path = os.path.join(objective_dir, "objective.json")
  write_json_atomic(path, payload)
  return path
