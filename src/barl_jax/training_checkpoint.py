"""Training-state resume at an iteration boundary, restarting environment
episodes.

Each save is an immutable, numbered generation (`gen-000000/`, `gen-000001/`,
...) under the resume root, written to a temp directory and atomically
published (`checkpoints.checkpoint_directory`) before `pointer.json` is
updated to name it current. Retention (`resume_keep`) only deletes older
generations *after* that publish -- so a crash mid-write, or one that fills
the disk, leaves the previous generation exactly as it was, never the only
recoverable one mid-deletion to make room for its replacement. Policy
checkpoints (`latest`/`best`/`final`, see `checkpoints.py`) are not
generational and are not touched by this retention."""

import json
import shutil
from pathlib import Path

import numpy as np

from barl_jax.checkpoints import checkpoint_directory
from barl_jax.result import write_json_atomic

# `read_training`/`validate_resume` are pure metadata readers -- json and
# pathlib only, so a caller that just wants a resume root's generation/config
# (checkpoint_info.py) never pulls in jax/flax/orbax. `save_training`/
# `restore_training` actually construct or restore learning state and import
# orbax and multiseed.unstack locally, where that state is built.

# v2: the replay row field carrying the accumulated discounted return was
# renamed rew -> chunk_return (docs/ALGORITHM_MAP.md). A v1 checkpoint's
# replay .npz has no chunk_return field, so restore_training would silently
# leave the buffer empty rather than actually resuming it -- rejected below
# instead, with no shim that reinterprets the old field name.
VERSION = 2

_GEN_PREFIX = "gen-"
_GEN_WIDTH = 6


def _generation_dir(root: Path, gen: int) -> Path:
  return root / f"{_GEN_PREFIX}{gen:0{_GEN_WIDTH}d}"


def _existing_generations(root: Path) -> list[int]:
  if not root.exists():
    return []
  return sorted(int(p.name[len(_GEN_PREFIX):]) for p in root.glob(f"{_GEN_PREFIX}*")
               if p.is_dir() and p.name[len(_GEN_PREFIX):].isdigit())


def _dir_size(path: Path) -> int:
  return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _check_disk_budget(root: Path, required_bytes: int) -> None:
  """Best-effort preflight, not a full simulation of buffer growth: the new
  generation briefly coexists with the previous one (retention only deletes
  after publish), so free space should cover at least one more copy of it.
  Skipped when there is no previous generation to estimate from."""
  if required_bytes <= 0:
    return
  probe = root if root.exists() else root.parent
  free = shutil.disk_usage(probe).free
  if free < required_bytes * 1.2:
    raise OSError(
      f"Refusing to start a new resume checkpoint generation under {root}: "
      f"{free / 1e9:.2f} GB free, the previous generation is "
      f"~{required_bytes / 1e9:.2f} GB -- writing one that never finishes is "
      f"worse than not starting it. Free disk space, or lower buffer_size / "
      f"resume_keep.")


def _prune_generations(root: Path, *, keep: int, published: int) -> None:
  for gen in _existing_generations(root)[:-keep] if keep > 0 else []:
    if gen == published:
      continue
    shutil.rmtree(_generation_dir(root, gen), ignore_errors=True)


# A buffer's `snapshot()` (rollout.py) mixes ring-sized arrays (the fields
# themselves, and a PrioritizedReplayBuffer's priority/generation arrays) with
# small JSON-safe scalars (size, write, rng state, max priority). This module
# owns splitting that apart for storage -- arrays into the .npz, scalars into
# training.json -- not what any of those entries mean; that stays the
# buffer's own contract.
def _snapshot_to_npz(snapshot: dict) -> tuple[dict, dict]:
  arrays = dict(snapshot["fields"])
  meta = {}
  for key, value in snapshot.items():
    if key == "fields":
      continue
    if isinstance(value, np.ndarray):
      arrays[f"_{key}"] = value
    else:
      meta[key] = value
  return arrays, meta


def _npz_to_snapshot(archive, meta: dict) -> dict:
  fields, extra = {}, {}
  for key in archive.files:
    if key.startswith("_"):
      extra[key[1:]] = archive[key]
    else:
      fields[key] = archive[key]
  return {**meta, "fields": fields, **extra}


def save_training(run, root, progress, *, keep: int = 2) -> None:
  import orbax.checkpoint as ocp

  from barl_jax.checkpoints import save_policy
  from barl_jax.multiseed import unstack

  root = Path(root).absolute()
  root.mkdir(parents=True, exist_ok=True)
  existing = _existing_generations(root)
  gen = (existing[-1] + 1) if existing else 0
  if existing:
    _check_disk_budget(root, _dir_size(_generation_dir(root, existing[-1])))

  with checkpoint_directory(_generation_dir(root, gen)) as destination:
    for i in range(run.num_seeds):
      policy, _ = unstack(run.graphdef, run.state, i)
      save_policy(policy, destination / f"seed{i}")
    with ocp.StandardCheckpointer() as ckpt:
      ckpt.save(destination / "training_state", run.state)
    replay = []
    for i, buffer in enumerate(run.buffer.seeds):
      arrays, meta = _snapshot_to_npz(buffer.snapshot())
      np.savez(destination / f"replay{i}.npz", **arrays)
      replay.append(meta)
    meta = {
      "format_version": VERSION,
      "config": run.cfg.to_dict(),
      "num_seeds": run.num_seeds,
      "capacity": run.buffer.max_transitions,
      "progress": progress,
      "replay": replay,
      "collector_rng": run.collector.rng.bit_generator.state,
      "episode_restore": "restart",
    }
    (destination / "training.json").write_text(json.dumps(meta, indent=2))

  # Only after the generation is fully committed does it become the resume
  # point, and only then may retention remove anything older.
  write_json_atomic(root / "pointer.json", {"generation": gen})
  _prune_generations(root, keep=keep, published=gen)


def read_training(root) -> dict:
  root = Path(root)
  pointer_path = root / "pointer.json"
  if not pointer_path.exists():
    raise ValueError(f"{root} is a policy checkpoint or has no resume generations; "
                     f"use checkpoint_mode: weights")
  pointer = json.loads(pointer_path.read_text())
  gen_dir = _generation_dir(root, pointer["generation"])
  meta_path = gen_dir / "training.json"
  if not meta_path.exists():
    raise ValueError(f"{root}: pointer.json names generation {pointer['generation']}, "
                     f"but {gen_dir} is missing or incomplete")
  meta = json.loads(meta_path.read_text())
  if meta.get("format_version") != VERSION:
    raise ValueError(
      f"Unsupported training checkpoint version {meta.get('format_version')} (expected "
      f"{VERSION}). Format version 2 renamed the replay's accumulated-return field "
      f"rew -> chunk_return; a version-1 checkpoint's replay contents cannot be "
      f"reinterpreted under the new name and must be rejected, not guessed at.")
  meta["_dir"] = str(gen_dir)
  return meta


def validate_resume(cfg, num_seeds, meta):
  from barl_jax.config.run import RunConfig
  if num_seeds != meta["num_seeds"]:
    raise ValueError("Resume requires the saved num_seeds")
  mutable = {"checkpoint", "checkpoint_mode", "run_dir", "sweep_tag", "validate",
             "log_every", "debug_every", "eval_every", "eval_runs",
             "eval_video_every", "compile_cache_dir", "wandb_finish_timeout",
             "logger", "resume_every", "resume_keep",
             # Evaluation strategy may change across a resume (recorded in
             # each eval's own results); collection strategy is a training
             # config and is deliberately NOT in this set -- changing it
             # changes the behavior policy that produced the replay buffer.
             "eval_executor"}
  saved = meta["config"]
  current = cfg.to_dict() if isinstance(cfg, RunConfig) else dict(cfg)
  # `model_overrides`/`env_metadata`/`snapshot` carry the saved model itself;
  # the run already reuses the saved model (see trainer._setup), so exclude
  # them from the "did training config change" comparison -- otherwise every
  # resume would report its own model as changed.
  for key in ("model_overrides", "env_metadata", "snapshot"):
    saved.pop(key, None) if isinstance(saved, dict) else None
    current.pop(key, None)
  def _cfg_get(mapping, key):
    return mapping.get(key) if isinstance(mapping, dict) else getattr(mapping, key, None)
  changed = [key for key, value in (saved.items() if isinstance(saved, dict) else [])
             if key not in mutable and _cfg_get(current, key) != value]
  if changed:
    raise ValueError(f"Resume changes training configuration: {', '.join(sorted(changed))}")


def restore_training(run, meta) -> None:
  import orbax.checkpoint as ocp

  from barl_jax.multiseed import unstack

  path = Path(meta["_dir"])
  if run.buffer.max_transitions != meta["capacity"]:
    raise ValueError("Resume requires the saved replay capacity")
  with ocp.StandardCheckpointer() as ckpt:
    run.state = ckpt.restore(path / "training_state", run.state)
  for i, (buffer, saved) in enumerate(zip(run.buffer.seeds, meta["replay"])):
    with np.load(path / f"replay{i}.npz", allow_pickle=False) as archive:
      buffer.restore(_npz_to_snapshot(archive, saved))
  run.collector.rng.bit_generator.state = meta["collector_rng"]
  run.progress = meta["progress"]
  run.reference, _ = unstack(run.graphdef, run.state, 0)
