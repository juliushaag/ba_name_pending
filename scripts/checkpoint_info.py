"""Inspect a checkpoint's metadata without constructing a policy or an
environment -- reading JSON sidecars and orbax's own metadata API (shapes and
dtypes, no array data restored) rather than `load_policy`/`trainer.setup`.

Handles both kinds this repo writes:

  * a **policy** checkpoint (`meta.json` + `state/`, see `checkpoints.py`) --
    `.../seed0/best`, `.../seed0/final`, or a standalone `checkpoint:` target.
  * a **training** (resume) root (`pointer.json` + `gen-*/`, see
    `training_checkpoint.py`) -- `.../checkpoints/resume`.

    uv run python scripts/checkpoint_info.py checkpoints/sac/halfcheetah/<ts>/seed0/best
    uv run python scripts/checkpoint_info.py checkpoints/sac/halfcheetah/<ts>/checkpoints/resume
    uv run python scripts/checkpoint_info.py <path> --json
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from barl_jax.config.model import model_fingerprint  # noqa: E402 -- after the JAX_PLATFORMS default

_GEN_PATTERN = "gen-"


def _array_stats(tree) -> tuple[int, int]:
  """(#arrays, #scalar elements) over an orbax metadata tree -- ArrayMetadata
  leaves only, never restoring the array data itself."""
  if hasattr(tree, "shape"):
    n = 1
    for d in tree.shape:
      n *= d
    return 1, n
  if isinstance(tree, dict):
    arrays = scalars = 0
    for v in tree.values():
      a, s = _array_stats(v)
      arrays += a
      scalars += s
    return arrays, scalars
  return 0, 0


def _shape_summary(state_dir: Path) -> dict:
  try:
    import orbax.checkpoint as ocp
    meta = ocp.StandardCheckpointer().metadata(state_dir)
    arrays, scalars = _array_stats(meta.item_metadata.tree)
    return {"num_arrays": arrays, "num_scalar_elements": scalars}
  except Exception as e:
    return {"shape_inspection_error": f"{type(e).__name__}: {e}"}


def inspect_policy(path: Path) -> dict:
  meta = json.loads((path / "meta.json").read_text())
  info = {
    "kind": "policy", "path": str(path),
    "format_version": meta.get("format_version"),
    "model": meta.get("model"), "profile": meta.get("profile"),
    "policy_class": meta.get("policy_class"), "policy_implementation": meta.get("policy_type"),
  }
  snapshot = meta.get("snapshot") or {}
  if "resolved" in snapshot:
    info["config_hash"] = model_fingerprint(snapshot["resolved"])
  info.update(_shape_summary(path / "state"))
  return info


def inspect_training(path: Path) -> dict:
  from barl_jax.training_checkpoint import VERSION, _existing_generations, _generation_dir
  pointer = json.loads((path / "pointer.json").read_text())
  gen = pointer["generation"]
  gen_dir = _generation_dir(path, gen)
  training = json.loads((gen_dir / "training.json").read_text())
  version = training.get("format_version")
  info = {
    "kind": "training (resume)", "path": str(path),
    "format_version": version,
    "current_generation": gen,
    "generations_on_disk": [f"{_GEN_PATTERN}{g:06d}" for g in _existing_generations(path)],
    "num_seeds": training.get("num_seeds"), "capacity": training.get("capacity"),
    "progress": training.get("progress"),
    "episode_restore": training.get("episode_restore"),
    "resume_limitations": (
      "ordinary resume restarts environment episodes and discards pending action "
      "chunks; optimizer state, replay contents, sampling/collector RNGs and step "
      "counters are restored exactly (see training_checkpoint.restore_training)."),
  }
  if version != VERSION:
    info["warning"] = f"format_version {version} != this code's {VERSION} -- read_training would reject it"
  cfg = training.get("config") or {}
  resolved = (cfg.get("snapshot") or {}).get("resolved")
  if resolved is not None:
    info["config_hash"] = model_fingerprint(resolved)
  for i in range(training.get("num_seeds") or 0):
    seed_dir = gen_dir / f"seed{i}"
    if (seed_dir / "meta.json").exists():
      info.setdefault("policy_shapes", {})[f"seed{i}"] = _shape_summary(seed_dir / "state")
  return info


def inspect(path: str) -> dict:
  p = Path(path)
  if (p / "pointer.json").exists():
    return inspect_training(p)
  if (p / "meta.json").exists():
    return inspect_policy(p)
  raise SystemExit(f"{p}: neither a policy checkpoint (no meta.json) nor a training "
                   f"resume root (no pointer.json)")


if __name__ == "__main__":
  args = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  args.add_argument("path")
  args.add_argument("--json", action="store_true")
  kwargs = args.parse_args()

  info = inspect(kwargs.path)
  if kwargs.json:
    print(json.dumps(info, indent=2, sort_keys=True, default=str))
  else:
    width = max(len(k) for k in info)
    for k, v in info.items():
      print(f"{k:<{width}}  {v}")
