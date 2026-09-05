"""A small, runnable benchmark harness -- not a benchmarking framework.
Measures import latency in a fresh process, async vector-env worker RSS,
collection throughput (transitions/s), and compiled gradient throughput
(ms/grad-step), and reports everything needed to judge whether a number is
comparable to another run: device, library versions, shapes, batch/chunk/seed
counts, precision, and compilation-cache setting.

    uv run python scripts/benchmark.py
    uv run python scripts/benchmark.py --model SAC --env HalfCheetah --seeds 3 --batch-size 256
    uv run python scripts/benchmark.py --json

Uses `jax.block_until_ready` around every timed JAX call and times warmup
(compile) separately from the steady-state loop -- conflating them is the
most common way a JAX benchmark lies. The test suite disables some XLA
optimization for speed (tests/conftest.py's XLA_FLAGS); this script does not
inherit that, since a performance number measured under it would not
transfer to a real run.

No hard pass/fail thresholds here deliberately -- run-to-run variance on
shared or laptop hardware is large enough that a hardcoded number would be
noise, not signal. Compare two of this script's own JSON outputs on the same
machine instead."""

import argparse
import json
import os
import resource
import subprocess
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("BARL_ALLOW_CPU", "1")


def _import_latency() -> float:
  """Fresh-process import time -- a warm process (this one) already paid it,
  so this is measured as a subprocess, the only way to see it cold."""
  code = "import time; t0 = time.perf_counter(); import barl_jax.trainer; print(time.perf_counter() - t0)"
  result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
  return float(result.stdout.strip())


def _async_worker_rss_kb() -> int | None:
  """Peak RSS of every child process spawned so far (`RUSAGE_CHILDREN`), delta
  across building and closing one async vector-env worker -- an approximation
  (it is every child this process has ever reaped, not just this one), but
  needs no extra dependency (no psutil) and is the right order of magnitude
  for "does this leak/balloon", which is what the number is for."""
  if not hasattr(resource, "RUSAGE_CHILDREN"):
    return None  # not available on this platform (e.g. some sandboxes)
  from barl_jax.env import Environment
  before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
  environment = Environment.make("HalfCheetah")
  env = environment.get_vector_env(1, asynchronous=True)
  env.reset(seed=0)
  env.close()
  after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
  return max(0, after - before)


def _versions() -> dict:
  import importlib.metadata as md

  import jax
  out = {"python": sys.version.split()[0], "jax": jax.__version__,
        "jax_backend": jax.default_backend(), "jax_devices": [str(d) for d in jax.devices()]}
  for dist in ("flax", "optax", "jaxlib"):
    try:
      out[dist] = md.version(dist)
    except md.PackageNotFoundError:
      out[dist] = None
  return out


def _throughput(model: str, env_name: str, *, num_seeds: int, num_envs: int, batch_size: int,
               chunk_size: int, collect_steps: int, grad_steps: int) -> dict:
  import jax
  import jax.numpy as jnp
  from flax import nnx

  from barl_jax.collection import Collector
  from barl_jax.components import TRAINABLE
  from barl_jax.env import Environment
  from barl_jax.multiseed import StackedPolicy, stack
  from barl_jax.optimization import seeded_step
  from barl_jax.policy import Policy
  from barl_jax.rollout import MultiSeedReplay

  environment = Environment.make(env_name)
  env = environment.get_env()
  stats = environment.get_env_stats(env)
  rollout_env = environment.get_vector_env(num_seeds * num_envs)

  policies = [Policy.make(model, env, env_stats=stats, seed=i, chunk_size=chunk_size)
             for i in range(num_seeds)]
  optimizers = [nnx.Optimizer(p, p.build_optimizer(lr_schedule=None), wrt=TRAINABLE) for p in policies]
  graphdef, state = stack(list(zip(policies, optimizers)))

  buffer = MultiSeedReplay(num_seeds, num_envs, max(collect_steps * num_envs, batch_size), seed=0)
  collector = Collector(seed=0)
  actor = StackedPolicy(graphdef, state, num_seeds, num_envs, policies[0])

  t0 = time.perf_counter()
  collector.collect(actor, env=rollout_env, n_steps=collect_steps, seed=0,
                    discount=actor.discount, sink=buffer)
  collect_s = time.perf_counter() - t0
  transitions = collect_steps * num_envs * num_seeds
  rollout_env.close()

  batch = buffer.sample(batch_size)
  t0 = time.perf_counter()
  state, *_ = seeded_step(graphdef, state, batch, (), jnp.asarray(0, jnp.int32))
  jax.block_until_ready(state)
  compile_s = time.perf_counter() - t0

  batch = buffer.sample(batch_size)
  t0 = time.perf_counter()
  for step in range(grad_steps):
    state, *_ = seeded_step(graphdef, state, batch, (), jnp.asarray(step + 1, jnp.int32))
  jax.block_until_ready(state)
  warm_s = time.perf_counter() - t0

  env.close()
  return {
    "model": model, "env": env_name,
    "num_seeds": num_seeds, "num_envs": num_envs, "batch_size": batch_size, "chunk_size": chunk_size,
    "transitions_collected": transitions, "collect_s": collect_s,
    "transitions_per_s": transitions / collect_s if collect_s > 0 else None,
    "grad_step_compile_s": compile_s,
    "grad_steps_measured": grad_steps, "grad_step_warm_s_total": warm_s,
    "ms_per_grad_step_warm": 1000 * warm_s / grad_steps if grad_steps else None,
  }


if __name__ == "__main__":
  args = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  args.add_argument("--model", default="SAC")
  args.add_argument("--env", default="HalfCheetah")
  args.add_argument("--seeds", type=int, default=1)
  args.add_argument("--num-envs", type=int, default=1)
  args.add_argument("--batch-size", type=int, default=256)
  args.add_argument("--chunk-size", type=int, default=1)
  args.add_argument("--collect-steps", type=int, default=32)
  args.add_argument("--grad-steps", type=int, default=20)
  args.add_argument("--skip-rss", action="store_true", help="Skip the async-worker RSS probe")
  args.add_argument("--json", action="store_true")
  kwargs = args.parse_args()

  report = {"versions": _versions(), "import_latency_s": _import_latency()}
  if not kwargs.skip_rss:
    report["async_worker_rss_kb"] = _async_worker_rss_kb()
  report["throughput"] = _throughput(
    kwargs.model, kwargs.env, num_seeds=kwargs.seeds, num_envs=kwargs.num_envs,
    batch_size=kwargs.batch_size, chunk_size=kwargs.chunk_size,
    collect_steps=kwargs.collect_steps, grad_steps=kwargs.grad_steps)

  if kwargs.json:
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
  else:
    print("versions:", report["versions"])
    print(f"import latency: {report['import_latency_s']:.3f}s")
    if "async_worker_rss_kb" in report:
      rss = report["async_worker_rss_kb"]
      print(f"async worker RSS delta: {rss / 1024:.1f} MB" if rss is not None else
            "async worker RSS delta: unavailable on this platform")
    t = report["throughput"]
    print(f"{t['model']} on {t['env']}: {t['num_seeds']} seeds x {t['num_envs']} envs, "
          f"batch {t['batch_size']}, chunk {t['chunk_size']}")
    print(f"  collection: {t['transitions_per_s']:.0f} transitions/s "
          f"({t['transitions_collected']} in {t['collect_s']:.2f}s)")
    print(f"  grad step compile: {t['grad_step_compile_s']:.2f}s (cold, one call)")
    print(f"  grad step warm: {t['ms_per_grad_step_warm']:.1f} ms/step "
          f"({t['grad_steps_measured']} steps in {t['grad_step_warm_s_total']:.2f}s)")
