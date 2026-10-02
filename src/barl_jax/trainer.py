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
import math
import os
import subprocess
import time
from contextlib import ExitStack

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from barl_jax.checkpoints import load_policy, save_policy, saved_model
from barl_jax.collection import Collector
from barl_jax.components import TRAINABLE
from barl_jax.config.run import RunConfig, validate_run
from barl_jax.config.serialization import flatten
from barl_jax.env import Environment
from barl_jax.evaluation import run_episodes
from barl_jax.executors import executor_from_config
from barl_jax.metrics import DEBUG, PRIMARY, to_host
from barl_jax.multiseed import SeedActor, StackedPolicy, split_metrics, stack, unstack
from barl_jax.optimization import seeded_chunk, seeded_step
from barl_jax.policy import Policy
from barl_jax.replay import MultiSeedReplay
from barl_jax.result import presentable, write_json_atomic
from barl_jax.seeding import derive_seed
from barl_jax.training_checkpoint import (
  MUTABLE_ON_RESUME,
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

  def __init__(self, cfg: RunConfig, num_seeds: int):
    # CLI `--seeds` wins over the stored value; keep the stored object as the
    # source of every other run setting.
    if num_seeds != cfg.num_seeds:
      cfg = cfg.replace(num_seeds=num_seeds)
    self.resources = ExitStack()
    self.cfg = cfg
    self.num_seeds = num_seeds
    self.model_name, self.env_name = cfg.model, cfg.env
    self.base_seed = cfg.seed
    self.num_envs = cfg.num_envs
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


def setup(cfg: RunConfig, num_seeds: int, *, validate_only: bool = False) -> Run:
  run = Run(cfg, num_seeds)
  try:
    return _setup(run, validate_only=validate_only)
  except BaseException:
    run.resources.close()
    raise


def _setup(run, *, validate_only):
  problems = validate_run(run.cfg, num_seeds=run.num_seeds)
  if problems:
    raise ValueError(f"{run.model_name} on {run.env_name}: invalid run config, refusing to start:\n  "
                     + "\n  ".join(problems))
  enable_compilation_cache(run.cfg.compile_cache_dir)

  resume_meta = _resolve_config(run)
  _build_environments(run, validate_only)
  policies = _build_policies(run, resume_meta)

  cfg = run.cfg
  params = cfg.params
  total_grad_steps = max(1, round(cfg.total_env_steps * params.utd_ratio))
  schedule = Policy.lr_schedule(params.lr_warmup_steps, total_grad_steps, params.lr_decay)
  optimizers = [nnx.Optimizer(p, p.build_optimizer(lr_schedule=schedule), wrt=TRAINABLE)
                for p in policies]
  run.reference = policies[0]
  run.graphdef, run.state = stack(list(zip(policies, optimizers)))
  if resume_meta is not None:
    restore_training(run, resume_meta)
  # A full walk of the state tree (1.7 ms) -- measured once, not per iteration.
  run.num_params = run.reference.num_params()

  run.collection_executor = executor_from_config(cfg.collection_executor)
  run.eval_executor = executor_from_config(cfg.eval_executor)
  problems = [f"{label}: {p}" for label, executor in
              (("collection", run.collection_executor), ("eval", run.eval_executor))
              for p in executor.validate(policy=run.reference)]
  if problems:
    raise ValueError(f"{run.model_name}: invalid execution spec, refusing to start:\n  "
                     + "\n  ".join(problems))
  return run


def _resolve_config(run) -> dict | None:
  """Apply checkpoint semantics to `run.cfg`; returns the resume metadata, if any.

  The saved model always takes precedence over current defaults. On resume the
  saved run config is reused too, apart from the mutable fields listed in
  `training_checkpoint.MUTABLE_ON_RESUME`."""
  cfg = run.cfg
  if cfg.checkpoint_mode == "resume":
    if cfg.checkpoint is None:
      raise ValueError("checkpoint_mode: resume requires checkpoint")
    meta = read_training(cfg.checkpoint)
    validate_resume(cfg, run.num_seeds, meta)
    run.cfg = RunConfig.from_dict(meta["config"]).replace(
      **{key: getattr(cfg, key) for key in MUTABLE_ON_RESUME})
    return meta
  if cfg.checkpoint:
    run.cfg = cfg.replace(model_config=saved_model(cfg.checkpoint)[0])
  return None


def _build_environments(run, validate_only: bool) -> None:
  cfg = run.cfg
  run.environment = Environment.make(run.env_name)
  run.env = run.own(run.environment.get_env())       # single env: spaces and EnvStats
  run.env_stats = run.environment.get_env_stats(run.env)

  run.num_envs = 1 if validate_only else run.num_envs
  run.rollout_env = run.own(run.environment.get_vector_env(run.num_seeds * run.num_envs))
  # A seed's buffer never receives more than total_env_steps transitions.
  buffer_size = min(cfg.buffer_size, cfg.total_env_steps)
  run.collector = Collector(seed=run.base_seed)
  run.buffer = MultiSeedReplay(run.num_seeds, run.num_envs, buffer_size, seed=run.base_seed)


def _build_policies(run, resume_meta) -> list[Policy]:
  cfg, checkpoint = run.cfg, run.cfg.checkpoint
  if checkpoint is None:
    policies = [Policy.make(run.model_name, run.env, cfg.model_config,
                            env_stats=run.env_stats, seed=run.base_seed + i)
                for i in range(run.num_seeds)]
  elif not os.path.exists(checkpoint):
    raise FileNotFoundError(f"checkpoint={checkpoint!r}: no such directory")
  elif resume_meta is not None:
    policies = [load_policy(os.path.join(resume_meta["_dir"], f"seed{i}"), run.env, run.env_stats)
                for i in range(run.num_seeds)]
  else:
    policies = [load_policy(checkpoint, run.env, run.env_stats) for _ in range(run.num_seeds)]
    for i, policy in enumerate(policies):
      policy.train_steps[...] = jnp.zeros_like(policy.train_steps[...])
      policy.rngs = nnx.Rngs(run.base_seed + i)
      for component in policy.components.values():
        if hasattr(component, "target_steps"):
          component.target_steps[...] = jnp.zeros_like(component.target_steps[...])
  if policies[0].model_name != run.model_name:
    raise ValueError(f"Checkpoint model {policies[0].model_name!r} differs from {run.model_name!r}")
  return policies


# Evaluation


def _checkpoint_dir(run: Run, root: str, seed: int) -> str:
  """Every seed's checkpoints live under its own directory, including seed 0."""
  return os.path.join(root, f"seed{seed}")


def summarize_eval(all_runs: list, num_seeds: int) -> list[dict]:
  """Per-seed eval metrics from `run_episodes` results, which are round-major:
  seed 0, seed 1, ... for each round."""
  out = []
  for i in range(num_seeds):
    runs = all_runs[i::num_seeds]
    seen = sum(r["n_act"] for r in runs)
    out.append({
      "eval/avg_cum_reward": float(np.mean([np.sum(r["rewards"]) for r in runs])),
      "eval/avg_length":     float(np.mean([len(r["rewards"]) for r in runs])),
      "eval/success_rate":   float(np.mean([float(r["success"]) for r in runs])),
      "eval/act_nonfinite":  sum(r["act_nonfinite"] for r in runs) / seen if seen else 0.0,
    })
  return out


def evaluate(run: Run, envs, log, ckpt_dir, best, *, want_video: bool):
  """Every seed in one vmapped pass, plus one video episode per seed.

  Returns `(best, timings)`; the three phases are timed apart because they fail
  and stall for unrelated reasons -- the wave is sequential physics, the saves
  are orbax writes, the video is an EGL render plus an mp4 encode."""
  cfg = run.cfg
  eval_env, render_env = envs

  # Evaluation identity is independent of training RNG: the same run seed and
  # eval_count always reproduce the same episodes, and eval never perturbs a
  # training trajectory.
  eval_seed = derive_seed(run.base_seed, "eval", run.eval_count)
  run.eval_count += 1

  started = time.perf_counter()
  log.message(f"Eval start @ env_step {log.get_step('env_steps')} "
              f"(every {cfg.eval_every}), {run.num_seeds} seeds x {cfg.eval_runs} episodes, "
              f"eval_seed={eval_seed}")

  # One slot per seed; run_episodes reuses these slots for successive rounds.
  actor = StackedPolicy(run.graphdef, run.state, run.num_seeds, 1, run.reference,
                        executor=run.eval_executor)

  # `eval/phase` is the one thing that reaches wandb *during* an eval: a killed
  # run uploads no term.err, so without it "died before the eval" and "died
  # inside the eval" look the same from the dashboard.
  log.log({}, quiet={"eval/phase": PHASE_EVAL_WAVE})
  t0 = time.perf_counter()
  all_runs = run_episodes(actor, eval_env, run.num_seeds * cfg.eval_runs, render=False, seed=eval_seed)
  wave_s = time.perf_counter() - t0

  log.log({}, quiet={"eval/phase": PHASE_EVAL_CKPT})
  t0 = time.perf_counter()
  per_seed = summarize_eval(all_runs, run.num_seeds)
  for i, metrics in enumerate(per_seed):
    reward = metrics["eval/avg_cum_reward"]
    policy, _ = unstack(run.graphdef, run.state, i)
    root = _checkpoint_dir(run, ckpt_dir, i)
    save_policy(policy, os.path.join(root, "latest"))
    if reward > best[i]:
      best[i] = reward
      save_policy(policy, os.path.join(root, "best"))
  ckpt_s = time.perf_counter() - t0

  # One video episode on `render_env` (sized 1, synchronous -- a forked worker's
  # render() does not survive the fork). It is an extra episode rather than one
  # of the counted `eval_runs`: a VectorEnv renders all its sub-envs or none.
  # Capped at one seed because every render sub-env owns an offscreen GL
  # context; num_seeds of them at once exhausts the GPU memory XLA leaves free.
  # The rendered seed rotates across evals so every seed is sampled over the run.
  video_s = 0.0
  if want_video:
    log.log({}, quiet={"eval/phase": PHASE_EVAL_VIDEO})
    t0 = time.perf_counter()
    video_seed = derive_seed(run.base_seed, "eval-video", run.eval_count - 1)
    seed_index = (run.eval_count - 1) % run.num_seeds
    policy, _ = unstack(run.graphdef, run.state, seed_index)
    episode = run_episodes(SeedActor(policy, run.eval_executor), render_env, 1,
                           render=True, seed=video_seed)[0]
    log.log_video("eval/video", episode["frames"], seed_index=seed_index)
    video_s = time.perf_counter() - t0

  panel = {key: np.mean([m[key] for m in per_seed]) for key in per_seed[0]}
  log.log_per_seed(per_seed, panel=panel)

  nonfinite = [m["eval/act_nonfinite"] for m in per_seed]
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


# Training


def run_config_snapshot(run: Run) -> dict:
  """Everything that determines this run, flattened for the logger's config
  panel (wandb's, or whatever a future local equivalent shows).

  Assembled from the resolved typed config rather than from any YAML layer,
  so what the panel shows *is* what ran: `model/*` carries every component
  value including hidden tier/style defaults, and `run/*` every run setting.
  `snapshot` is the same model nested, kept so a finished run replays exactly
  from its saved spec."""
  cfg = run.cfg
  model_dict = cfg.model_config.to_dict()
  flat = flatten(model_dict)

  env_stats = {}
  for key, value in run.env_stats.items():
    env_stats[key] = value.tolist() if isinstance(value, np.ndarray) else value

  run_dict = cfg.to_dict()
  config = {f"run/{k}": v for k, v in run_dict.items()
            if k not in ("model_overrides", "snapshot")}
  config.update({f"model/{k}": v for k, v in flat.items()})
  config.update({f"model/derived.{k}": v for k, v in run.reference.derived_config().items()})
  config.update({f"env/{k}": v for k, v in env_stats.items()})
  config.update({
    "model": run.model_name, "env": run.env_name, "backend": "jax",
    "num_seeds": run.num_seeds, "base_seed": run.base_seed,
    "env/obs_space": str(run.env.observation_space),
    "env/act_space": str(run.env.action_space),
    "env/async_vector_env": run.environment.use_async_vector_env,
    "meta/git_sha": _git_sha(),
    "meta/config_hash": experiment_fingerprint(flat, run_dict),
    "snapshot": {"model": model_dict},
  })
  return config


_EXPERIMENT_IDENTITY_KEYS = frozenset({"seed", "num_seeds", "run_dir", "sweep_tag", "name",
                                       "snapshot", "checkpoint"})


def experiment_fingerprint(model_flat: dict, run_cfg: dict) -> str:
  """A short hash of everything that defines the experiment apart from its identity."""
  payload = {**{f"model.{k}": v for k, v in model_flat.items()},
             **{f"run.{k}": v for k, v in run_cfg.items() if k not in _EXPERIMENT_IDENTITY_KEYS}}
  return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


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
                        executor=run.eval_executor)
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
  cfg, num_seeds = run.cfg, run.num_seeds
  now  = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  name = f"{run.model_name.lower()}/{run.env_name.lower()}/{now}_x{num_seeds}"

  # A sweep hands every run its own directory (launch.grid); a standalone
  # run keeps the global checkpoints/ tree keyed like its wandb run.
  ckpt_dir = os.path.abspath(os.path.join(cfg.run_dir, "checkpoints") if cfg.run_dir
                             else os.path.join("checkpoints", name))
  os.makedirs(ckpt_dir, exist_ok=True)
  # Where the result sidecar lives: the only way to read a finished run without wandb.
  objective_dir = os.path.abspath(cfg.run_dir) if cfg.run_dir else os.path.join("checkpoints", name)

  log = _open_logger(run, name, ckpt_dir, objective_dir)
  eval_env = run.own(run.environment.get_vector_env(num_seeds))
  render_env = _open_render_env(run, log)
  eval_envs = (eval_env, render_env)

  warmup_stats = warmup(run, eval_envs, log)

  if cfg.params.random_steps > 0 and not run.progress:
    # Prefill steps count toward neither env_steps nor grad_steps; skipped on
    # resume, which already has a filled buffer.
    run.collector.random_fill(run.rollout_env, n_steps=cfg.params.random_steps // run.num_envs,
                              seed=run.base_seed, chunk_size=run.reference.chunk_size,
                              discount=run.reference.discount, sink=run.buffer)
    log.message(f"Random-filled the buffer with {cfg.params.random_steps} env steps before training")
  if run.progress:
    log.advance({key: run.progress[key] for key in ("env_steps", "grad_steps")})
    log.message("Resumed optimizer, replay and counters; environment episodes restart")

  _TrainLoop(run, log, eval_envs, render_env is not None, ckpt_dir, objective_dir,
             warmup_stats).execute()


def _open_logger(run: Run, name: str, ckpt_dir: str, objective_dir: str):
  from barl_jax.logger import make_logger
  cfg, num_seeds = run.cfg, run.num_seeds
  tags = [run.model_name, run.env_name, "online", "jax", "multiseed"]
  if cfg.sweep_tag:
    tags.append(cfg.sweep_tag)
  # The logger's display name; `name` stays the automatic one that keys the
  # checkpoint path.
  label = cfg.name or name
  log = make_logger(
    cfg.logger, label, step_keys={"env_steps", "grad_steps"}, tags=tags,
    config=run_config_snapshot(run), group=label,
    finish_timeout=cfg.wandb_finish_timeout, run_dir=objective_dir,
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
  return log


def _open_render_env(run: Run, log):
  """The one-slot video env, or None when videos are off or it cannot be built.

  Built after the logger exists so a construction failure is reported and the
  run carries on without video: a headless node's GL/EGL context is needed only
  for this optional artifact and must never take a healthy run down."""
  if run.cfg.eval_video_every is None:
    return None
  try:
    return run.own(run.environment.get_vector_env(1, asynchronous=False, render=True))
  except Exception as exc:
    log.message(f"WARNING: could not create the video render env ({exc!r}); "
                f"disabling eval videos for this run.")
    return None


class _TrainLoop:
  """The online loop: collect, gradient steps, periodic eval and resume saves.

  One `iteration()` collects `rollout_steps` env steps, then runs the gradient
  steps the utd ratio owes -- as one `lax.scan` dispatch when no metric tier is
  armed inside it, else step by step."""

  def __init__(self, run: Run, log, eval_envs, wants_video: bool, ckpt_dir: str,
               objective_dir: str, warmup_stats: dict):
    self.run, self.cfg, self.log = run, run.cfg, log
    self.eval_envs, self.ckpt_dir, self.objective_dir = eval_envs, ckpt_dir, objective_dir
    self.warmup_stats = warmup_stats
    self.video_every = self.cfg.eval_video_every if wants_video else None
    self.steps_per_iter = self.cfg.rollout_steps // run.num_envs
    self.best = run.progress.get("best", [float("-inf")] * run.num_seeds)
    self.next_eval = self._next_boundary(self.cfg.eval_every)
    # Resume persistence has its own cadence: the replay dump is what makes a save costly.
    self.resume_dir = os.path.join(ckpt_dir, "resume")
    self.next_resume = self._next_boundary(self.cfg.resume_every)
    self.it = run.progress.get("iteration", 0)
    self.chunk_seen = False
    self.prefetched = None  # (k, batches) drawn during the previous iteration's scan
    # Stdout is paced by wall clock, not by iteration: at `rollout_steps: 1` there
    # are thousands of iterations per 100k env steps. wandb still gets every one.
    self.last_narration = 0.0
    self.started = time.perf_counter()

  def _next_boundary(self, every: int) -> int:
    return (self.log.get_step("env_steps") // every + 1) * every

  def execute(self):
    run, log = self.run, self.log
    crash = None
    try:
      while log.get_step("env_steps") < self.cfg.total_env_steps:
        self.iteration()
    except BaseException as exc:
      crash = exc
      raise
    finally:
      # A mid-loop exception must still save weights, write the result and flush
      # the logger, or the failure is invisible from the dashboard. A fatal signal
      # bypasses this; that is faulthandler's job.
      try:
        for i in range(run.num_seeds):
          policy, _ = unstack(run.graphdef, run.state, i)
          save_policy(policy, os.path.join(_checkpoint_dir(run, self.ckpt_dir, i), "final"))
      except Exception as save_error:
        log.message(f"Could not write the final checkpoint: {save_error!r}")

      if crash is None:
        save_training(run, self.resume_dir, run.progress, keep=self.cfg.resume_keep)
      else:
        log.message(f"Run failed: {type(crash).__name__}: {crash}")
        log.summary("crash", f"{type(crash).__name__}: {crash}")

      self._write_objective("failed" if crash is not None else "finished")

  def _write_objective(self, status: str) -> None:
    run = self.run
    _write_objective(self.objective_dir, model=run.model_name, env=run.env_name,
                     seed=run.base_seed, num_seeds=run.num_seeds,
                     env_steps=int(self.log.get_step("env_steps")), best=self.best, status=status)

  def iteration(self):
    run, cfg, log = self.run, self.cfg, self.log
    iter_start = time.perf_counter()
    self._collect()
    grad_steps = self._grad_steps_owed()
    narrate = self.it == 0 or time.perf_counter() - self.last_narration > PROGRESS_INTERVAL_S
    if narrate:
      log.message(f"Rollout done: {run.buffer.min_transitions()} transitions per seed, "
                  f"~{grad_steps} grad steps this iter")
    log.log({"buffer/utd_ratio": grad_steps / cfg.rollout_steps})
    log.log_per_seed([{"buffer/transitions": n} for n in run.buffer.transitions_per_seed()],
                     panel={"buffer/transitions": run.buffer.min_transitions()})

    self._train(grad_steps, narrate)
    self._prefetch()

    eval_time, eval_phases = 0.0, {}
    if log.get_step("env_steps") >= self.next_eval:
      eval_start = time.perf_counter()
      eval_phases = self._evaluate()
      eval_time = time.perf_counter() - eval_start

    self._log_iteration(iter_start, eval_time, eval_phases)
    self.it += 1
    run.progress = {"env_steps": int(log.get_step("env_steps")),
                    "grad_steps": int(log.get_step("grad_steps")),
                    "iteration": self.it, "best": list(self.best)}
    if log.get_step("env_steps") >= self.next_resume:
      save_training(run, self.resume_dir, run.progress, keep=cfg.resume_keep)
      self.next_resume = self._next_boundary(cfg.resume_every)

  def _collect(self) -> float:
    run, log = self.run, self.log
    start = time.perf_counter()
    actor = StackedPolicy(run.graphdef, run.state, run.num_seeds, run.num_envs, run.reference,
                          executor=run.collection_executor)
    first_seed = (run.base_seed + int(run.progress.get("env_steps", 0))
                  if run.collector.state is None else None)
    rollout_stats = run.collector.collect(actor, env=run.rollout_env, n_steps=self.steps_per_iter,
                                          seed=first_seed, discount=actor.discount, sink=run.buffer)
    elapsed = time.perf_counter() - start
    log.advance({"env_steps": self.cfg.rollout_steps})
    log.log_per_seed([{f"rollout/{k}": v for k, v in seed.items()}
                      for seed in run.collector.per_seed(run.num_seeds, run.num_envs)],
                     panel={f"rollout/{k}": v for k, v in rollout_stats.items()})
    return elapsed

  def _grad_steps_owed(self) -> int:
    """Chunks span collection calls: wait for replay data, then catch up to the
    cumulative primitive-step budget (which also preserves fractional UTD)."""
    if self.run.buffer.min_transitions() <= 0:
      return 0
    log = self.log
    return max(0, round(log.get_step("env_steps") * self.cfg.params.utd_ratio) - log.get_step("grad_steps"))

  def _prefetch(self) -> None:
    """Draw the next iteration's batches while the scan just dispatched runs on the
    device. Env steps advance by exactly `rollout_steps`, so the next `k` is known.
    Skipped when this iteration ends in a resume save or the loop exits: the
    snapshot's replay RNG must not be ahead of the batches actually consumed."""
    run, cfg, log = self.run, self.cfg, self.log
    env_steps = log.get_step("env_steps")
    k = max(0, round((env_steps + cfg.rollout_steps) * cfg.params.utd_ratio) - log.get_step("grad_steps"))
    if (k == 0 or run.buffer.min_transitions() <= 0
        or env_steps >= self.next_resume or env_steps >= cfg.total_env_steps):
      return
    self.prefetched = (k, run.buffer.sample_chunk(k, cfg.params.batch_size))

  def _take_chunk(self, k: int):
    prefetched, self.prefetched = self.prefetched, None
    if prefetched is not None and prefetched[0] == k:
      return prefetched[1]
    return self.run.buffer.sample_chunk(k, self.cfg.params.batch_size)

  def _train(self, grad_steps: int, narrate: bool) -> None:
    run, cfg, log = self.run, self.cfg, self.log
    epoch_start = last_report = time.perf_counter()
    progress_every = max(1, grad_steps // 10)

    # Whole-iteration fast path: an iteration holding no tier-armed step is one
    # `lax.scan` and one host->device transfer instead of that many dispatches.
    # Split at the iteration, not at the armed step, so `k` takes one value for
    # the run -- a second chunk length is a second compile of the scan.
    base = log.get_step("grad_steps")
    armed_here = any((base + i) % cfg.log_every == 0 or (base + i) % cfg.debug_every == 0
                     for i in range(grad_steps))

    batches = self._take_chunk(grad_steps) if grad_steps else None
    if grad_steps and not armed_here:
      run.state = seeded_chunk(run.graphdef, run.state, batches, grad_steps,
                               jnp.asarray(base, jnp.int32))
      log.advance({"grad_steps": grad_steps})
      if not self.chunk_seen:
        self.chunk_seen = True
        jax.block_until_ready(run.state)
        log.message(f"  first chunked iteration ({grad_steps} grad steps, "
                    f"incl. compile): {time.perf_counter() - epoch_start:.1f}s")

    for step in (range(grad_steps) if armed_here else ()):
      gstep = log.get_step("grad_steps")
      tiers = (tuple([PRIMARY] if gstep % cfg.log_every == 0 else [])
               + tuple([DEBUG] if gstep % cfg.debug_every == 0 else []))

      batch = jax.tree.map(lambda value: value[step], batches)
      run.state, _, primary, debug = seeded_step(run.graphdef, run.state, batch, tiers,
                                                 jnp.asarray(gstep, jnp.int32))
      if tiers:
        primary, debug = to_host(primary), to_host(debug)
        primary_mean, primary_seeds = split_metrics(primary, run.num_seeds)
        _, debug_seeds = split_metrics(debug, run.num_seeds)
        log.log_per_seed(primary_seeds, debug_seeds, panel=primary_mean)
      log.advance({"grad_steps": 1})

      now, done = time.perf_counter(), step + 1
      if step == 0 and self.it == 0:
        # Not a step time: every distinct `tiers` value is a separate jit trace.
        log.message(f"  first grad step (incl. compile): {now - epoch_start:.1f}s")
        last_report = now
      elif narrate and done < grad_steps and (done % progress_every == 0
                                              or now - last_report > PROGRESS_INTERVAL_S):
        rate = done / (now - epoch_start)
        log.message(f"  {done}/{grad_steps} grad steps, {rate:.2f}/s, "
                    f"~{(grad_steps - done) / rate / 60:.1f} min left this iter")
        last_report = now

    if narrate:
      self.last_narration = time.perf_counter()

  def _evaluate(self) -> dict:
    run, cfg, log = self.run, self.cfg, self.log
    # A rendered episode costs ~140x an env step plus an mp4 encode, so video runs on
    # a coarser cadence: every `video_every`th eval (eval_count is not yet incremented).
    want_video = self.video_every is not None and (run.eval_count + 1) % self.video_every == 0
    self.best, phases = evaluate(run, self.eval_envs, log, self.ckpt_dir, self.best,
                                 want_video=want_video)
    self.next_eval = self._next_boundary(cfg.eval_every)
    self._write_objective("running")
    return phases

  def _log_iteration(self, iter_start, eval_time, eval_phases) -> None:
    run, log = self.run, self.log
    elapsed = max(1e-9, time.perf_counter() - self.started)
    log.log({"model/num_params": run.num_params,
             "model/eval_time_s": eval_time,
             "time/iter_s":    time.perf_counter() - iter_start,
             "time/env_steps_per_hour": 3600.0 * log.get_step("env_steps") / elapsed,
             "eval/phase": PHASE_TRAIN,
             **eval_phases, **self.warmup_stats})


def validate(run: Run):
  with run:
    return _validate(run)


def _validate(run: Run):
  """Cheap smoke test for `--validate` sweeps: one rollout chunk + one
  train_step + optimizer step, no eval/checkpointing/wandb logging."""
  cfg = run.cfg
  actor = StackedPolicy(run.graphdef, run.state, run.num_seeds, run.num_envs, run.reference,
                        executor=run.collection_executor)
  run.collector.collect(actor, env=run.rollout_env, n_steps=cfg.params.batch_size,
                     seed=run.base_seed, discount=actor.discount, sink=run.buffer)

  batch = run.buffer.sample(cfg.params.batch_size)
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
