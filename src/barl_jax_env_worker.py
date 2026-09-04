"""Picklable entry point for `AsyncVectorEnv` subprocess workers -- deliberately
kept *outside* the `barl_jax` package.

A rollout/eval worker only ever calls step()/reset() on one concrete
`Environment` subclass and hands numpy back over a pipe; it never touches jax,
torch or any policy/critic. But `AsyncVectorEnv` unpickles whatever picklable
factory it is given, and unpickling anything whose `__module__` starts with
`barl_jax.` runs `barl_jax/__init__.py` first -- which eagerly imports every
policy/critic (jax) and `barl_jax.environments.mani_skill` (which drags in
PyTorch transitively via sapien) -- regardless of how light the one env class
actually needed is. Measured cost of that on a DogRun worker: ~850MB of
private memory (JAX backend init alone, ~750MB) for libraries the worker
never calls, on top of the ~40MB it actually needs (dm_control + mujoco) --
and a worker's JAX backend init competes with the real training process for
GPU memory it will never use.

`make_env` lives here, outside `barl_jax`, so pickling it by reference never
touches that package. It loads just the two files a family module needs --
`env.py` and `environments/<module_name>.py` -- directly by path under their
real dotted names, so the target module's `from barl_jax.env import ...`
resolves without ever running the real `__init__.py`. Safe because a
worker's whole job is stepping one env forever; it never needs the real,
fully-registered package again."""

import importlib.util
import sys
import types
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "barl_jax"


def _load(name: str, path: Path):
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


def make_env(module_name: str, class_name: str):
  """Build one `gym.Env`: `class_name` from `barl_jax/environments/{module_name}.py`,
  loaded without importing the rest of barl_jax."""
  if "barl_jax" not in sys.modules:
    sys.modules["barl_jax"] = types.ModuleType("barl_jax")  # satisfies `from barl_jax.env import ...`
    _load("barl_jax.env", _SRC / "env.py")
  full_name = f"barl_jax.environments.{module_name}"
  module = sys.modules.get(full_name) or _load(full_name, _SRC / "environments" / f"{module_name}.py")
  return getattr(module, class_name)().get_env()
