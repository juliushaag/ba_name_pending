"""Versioned policy checkpoints and reconstruction from saved model specifications."""
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import orbax.checkpoint as ocp
from flax import nnx

from barl_jax.config.model import ModelSpec
from barl_jax.config.tree import ConfigTree

POLICY_VERSION = 2


@contextmanager
def checkpoint_directory(path):
  path = Path(path).absolute()
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent))
  backup = temporary.with_name(temporary.name + "-previous")
  committed = False
  try:
    yield temporary
    if path.exists():
      os.replace(path, backup)
    try:
      os.replace(temporary, path)
      committed = True
    except BaseException:
      if backup.exists():
        os.replace(backup, path)
      raise
  finally:
    shutil.rmtree(temporary, ignore_errors=True)
    if committed:
      shutil.rmtree(backup, ignore_errors=True)


def save_policy(policy, path):
  meta = {
    "format_version": POLICY_VERSION,
    "model": policy.model_name,
    "policy_class": type(policy).__name__,
    "policy_type": f"{type(policy).__module__}:{type(policy).__qualname__}",
    "profile": policy._profile,
    "kwargs": policy._raw_kwargs,
    "overrides": policy._overrides,
    "snapshot": policy._snapshot,
  }
  with checkpoint_directory(path) as destination:
    with ocp.StandardCheckpointer() as ckpt:
      ckpt.save(destination / "state", nnx.state(policy))
    (destination / "meta.json").write_text(json.dumps(meta, indent=2))


def policy_spec(path, name=None):
  meta = json.loads((Path(path) / "meta.json").read_text())
  version = meta.get("format_version")
  if version != POLICY_VERSION:
    raise ValueError(f"Unsupported policy checkpoint version {version}")
  saved_name = meta["model"]
  if name is not None and name != saved_name:
    raise ValueError(f"Checkpoint contains {saved_name!r}, requested {name!r}")
  config = ConfigTree(snapshot=meta.get("snapshot"), overrides=meta.get("overrides"))
  spec = ModelSpec.resolve(saved_name, config, meta.get("profile"))
  implementation = meta["policy_type"]
  return replace(spec, policy_class=implementation), meta


def load_policy(path, env, env_stats=None, name=None, **kwargs):
  from barl_jax.policy import Policy
  path = Path(path).absolute()
  spec, meta = policy_spec(path, name)
  policy = Policy.make(spec.name, env, spec=spec, env_stats=env_stats,
                       **{**meta.get("kwargs", {}), **kwargs})
  target = nnx.state(policy)
  with ocp.StandardCheckpointer() as ckpt:
    restored = ckpt.restore(path / "state", target)
  nnx.update(policy, restored)
  return policy
