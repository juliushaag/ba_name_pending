"""Versioned policy checkpoints and reconstruction from saved model specifications.

`save_policy`/`load_policy` are the only two operations that actually construct
or restore learning state (an nnx `Policy`, an orbax array tree); `policy_spec`
reads a checkpoint's `meta.json` and rebuilds its `ModelSpec` from nothing but
that JSON, so callers that only want metadata (`checkpoint_info.py`, config
diffing) can import this module and call it without pulling in jax/flax/orbax
at all -- those imports live inside the two functions that need them."""
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from barl_jax.config.model import ModelSpec

POLICY_VERSION = 3


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
  import orbax.checkpoint as ocp
  from flax import nnx

  meta = {
    "format_version": POLICY_VERSION,
    "model": policy.model_name,
    "policy_class": type(policy).__name__,
    "policy_type": f"{type(policy).__module__}:{type(policy).__qualname__}",
    "profile": policy._profile,
    "kwargs": policy._raw_kwargs,
    "spec": policy._spec.data(),
  }
  with checkpoint_directory(path) as destination:
    with ocp.StandardCheckpointer() as ckpt:
      ckpt.save(destination / "state", nnx.state(policy))
    (destination / "meta.json").write_text(json.dumps(meta, indent=2))


def policy_spec(path, name=None):
  import json as _json
  meta = _json.loads((Path(path) / "meta.json").read_text())
  version = meta.get("format_version")
  if version != POLICY_VERSION:
    raise ValueError(f"Unsupported policy checkpoint version {version}")
  saved_name = meta["model"]
  if name is not None and name != saved_name:
    raise ValueError(f"Checkpoint contains {saved_name!r}, requested {name!r}")
  # Saved spec is authoritative; current defaults never replace saved values.
  saved = meta.get("spec")
  if not isinstance(saved, dict) or "model" not in saved:
    raise ValueError(f"Checkpoint {path} has no versioned model spec")
  spec = ModelSpec(
    name=saved.get("name", saved_name),
    profile=saved.get("profile"),
    policy_class=saved["model"].get("policy", meta.get("policy_class", "")),
    payload=json.dumps(saved, sort_keys=True, separators=(",", ":")),
    schema_version=int(saved.get("schema_version", 1)),
  )
  implementation = meta["policy_type"]
  return replace(spec, policy_class=implementation), meta


def load_policy(path, env, env_stats=None, name=None, **kwargs):
  import orbax.checkpoint as ocp
  from flax import nnx

  from barl_jax.policy import Policy
  path = Path(path).absolute()
  spec, meta = policy_spec(path, name)
  # `.get("kwargs", {})` only falls back on a *missing* key -- an older
  # checkpoint whose meta.json has `"kwargs": null` (the key present, value
  # None) makes it through unchanged, and `{**None, **kwargs}` raises exactly
  # "'NoneType' object is not a mapping". `or {}` catches both cases.
  policy = Policy.make(spec.name, env, spec=spec, env_stats=env_stats,
                       **{**(meta.get("kwargs") or {}), **kwargs})
  target = nnx.state(policy)
  with ocp.PyTreeCheckpointer() as ckpt:
    restored = ckpt.restore(
      path / "state", 
      args=ocp.args.PyTreeRestore(
        item=target,
        restore_args=ocp.checkpoint_utils.construct_restore_args(target),
        partial_restore=True,
    ))
  nnx.update(policy, restored)
  return policy
