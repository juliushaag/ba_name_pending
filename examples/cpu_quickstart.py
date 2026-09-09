"""A complete, deterministic, CPU-only round trip through barl_jax: a tiny
environment, a real policy, collector, replay, training steps, a policy
checkpoint save/load, and a training resume -- everything the README
describes, runnable on a laptop with no accelerator, no W&B account, no
simulator download, and no render dependency.

    BARL_ALLOW_CPU=1 uv run python -m examples.cpu_quickstart

Must run as `-m examples.cpu_quickstart` from the repo root: the "external
environment" path below (`env: "examples.cpu_quickstart:TinyPendulum"`) is
resolved by `importlib.import_module("examples.cpu_quickstart")`, which
needs the repo root on `sys.path` -- exactly what `-m` gives it.

Cold and warm timing are reported separately (trainer.py's own "first grad
step (incl. compile)" / "first chunked iteration" messages): the first call
at each batch shape pays XLA compilation, which does not repeat. Measured
end to end at ~16s (~3.1s of that the first grad step's compile, ~2.0s the
first chunked iteration's) on a 2024 laptop CPU, no GPU visible -- recompute
this on whatever machine actually matters before repeating the number
elsewhere.
"""

import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("BARL_ALLOW_CPU", "1")

import gymnasium as gym
import jax
import jax.numpy as jnp

import barl_jax  # noqa: F401 -- populates the policy/component registries
from barl_jax.env import Environment

RUN_DIR = Path("/tmp/barl_jax_cpu_quickstart")


class TinyPendulum(Environment):
  """gymnasium's own `Pendulum-v1` (classic_control: numpy only, no
  simulator, no render dependency unless `render()` is actually called)
  through the same `Environment` contract every registered env uses. Not
  registered centrally (no `@register_environment`) -- selected purely via
  the external `module:Class` path, exactly as a real external environment
  package would be."""

  # Per-step reward is in [-(pi^2 + 0.1*8^2 + 0.001*2^2), 0] = [-16.2736..., 0]
  # (gymnasium's own PendulumEnv.step); these bound a categorical critic's
  # fixed support (barl_jax.critics.categorical.C51Critic) over the 200-step
  # episode gymnasium's own env spec sets.
  stats = {"return_min": -16.2736044 * 200, "return_max": 0.0}

  def get_env(self, *, render: bool = False) -> gym.Env:
    return gym.make("Pendulum-v1")

  def get_dataset(self):
    raise NotImplementedError("TinyPendulum has no offline dataset.")


def base_config() -> dict:
  from barl_jax.config.run import run_defaults
  return {
    **run_defaults(),
    "model": "SAC", "env": "examples.cpu_quickstart:TinyPendulum",
    "seed": 0, "num_seeds": 1,
    "logger": "none",                # no W&B account or network needed
    "total_env_steps": 300, "random_steps": 100,
    "rollout_steps": 1, "num_envs": 1, "buffer_size": 300,
    "batch_size": 32, "utd_ratio": 1.0,
    "eval_every": 300, "eval_runs": 1, "eval_video_every": None,  # no render dependency
    "log_every": 50, "debug_every": 300,
    "resume_every": 300, "resume_keep": 1,
    "run_dir": str(RUN_DIR), "compile_cache_dir": None,
    # A real net, just a small one -- this is a plumbing demonstration, not a
    # capacity claim. tests/conftest.py's TINY fixture shrinks the same way.
    "model_overrides": {"components.actor.latent_dim": 8, "components.critic.latent_dim": 8},
  }


def main():
  from barl_jax.checkpoints import load_policy
  from barl_jax.trainer import setup, train

  if RUN_DIR.exists():
    shutil.rmtree(RUN_DIR)  # a clean run each time -- this is a demonstration, not a resumable job

  started = time.perf_counter()

  # 1) A full training run: collector, replay, training steps, eval,
  #    policy checkpoints (latest/best/final), and a resume checkpoint --
  #    trainer.train() end to end, exactly as scripts/submit.py
  #    calls it, only with `logger: none` in place of wandb.
  cfg = base_config()
  run = setup(cfg, num_seeds=1)
  train(run)
  print(f"\nTraining done in {time.perf_counter() - started:.1f}s total.")

  ckpt_dir = RUN_DIR / "checkpoints"
  final_dir = ckpt_dir / "seed0" / "final"
  resume_dir = ckpt_dir / "resume"
  assert final_dir.exists(), f"expected a final policy checkpoint at {final_dir}"
  assert (resume_dir / "pointer.json").exists(), f"expected a resume checkpoint at {resume_dir}"

  # 2) Policy checkpoint save/load: a fresh env, no training state, just the
  #    saved architecture and weights -- what an inference-only consumer uses.
  environment = TinyPendulum()
  env = environment.get_env()
  env_stats = environment.get_env_stats(env)  # return_min/max: not derivable from gym spaces alone
  policy = load_policy(final_dir, env, env_stats)
  obs, _ = env.reset(seed=0)
  action, _ = policy.inference_fn()(jnp.asarray(obs)[None], jax.random.PRNGKey(0))
  assert bool(jnp.all(jnp.isfinite(action))), "loaded policy produced a non-finite action"
  print(f"Loaded policy checkpoint from {final_dir}, action shape {action.shape}.")

  # 3) Training resume: optimizer, replay contents, sampling/collector RNGs
  #    and counters are restored exactly; environment episodes restart (see
  #    README's Checkpoints section -- this is training-state continuation,
  #    not bit-identical simulator trajectory continuation). `total_env_steps`
  #    is a training-config value resume must match exactly (see
  #    training_checkpoint.validate_resume's mutable-key set) -- unchanged
  #    here, so restoring lands exactly at the original run's final counters,
  #    which is what this checks against.
  resumed_cfg = {**cfg, "checkpoint": str(resume_dir), "checkpoint_mode": "resume"}
  resumed = setup(resumed_cfg, num_seeds=1)
  assert resumed.progress == run.progress, (
    f"resumed progress {resumed.progress} does not match the original run's final "
    f"{run.progress}")
  for a, b in zip(jax.tree.leaves(resumed.state), jax.tree.leaves(run.state)):
    assert (jnp.asarray(a) == jnp.asarray(b)).all(), "resumed optimizer/policy state differs"
  for buf_a, buf_b in zip(resumed.buffer.seeds, run.buffer.seeds):
    assert buf_a._size == buf_b._size
    for key in buf_a._fields:
      assert (buf_a._fields[key][:buf_a._size] == buf_b._fields[key][:buf_b._size]).all(), \
        f"resumed replay field {key!r} differs from the original run's"
  print(f"Resume restored progress={resumed.progress}, optimizer/policy state, and "
        f"{resumed.buffer.min_transitions()} replay transitions per seed -- all bit-identical "
        f"to the original run's final state.")

  print(f"\nDone in {time.perf_counter() - started:.1f}s total "
        f"(cold: first compile at each batch shape; see the 'incl. compile' lines above).")


if __name__ == "__main__":
  main()
