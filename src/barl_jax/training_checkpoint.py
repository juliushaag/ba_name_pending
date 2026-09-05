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
import orbax.checkpoint as ocp

from barl_jax.checkpoints import checkpoint_directory, save_policy
from barl_jax.multiseed import unstack
from barl_jax.result import write_json_atomic

VERSION = 1

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


def save_training(run, root, progress, *, keep: int = 2) -> None:
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
      fields = buffer._fields or {}
      np.savez(destination / f"replay{i}.npz",
               **{k: v[:buffer._size] for k, v in fields.items()})
      replay.append({"size": buffer._size, "write": buffer._write,
                     "rng": buffer.rng.bit_generator.state})
    meta = {
      "format_version": VERSION,
      "config": {**run.cfg, "snapshot": run.reference._snapshot},
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
    raise ValueError(f"Unsupported training checkpoint version {meta.get('format_version')}")
  meta["_dir"] = str(gen_dir)
  return meta


def validate_resume(cfg, num_seeds, meta):
  if num_seeds != meta["num_seeds"]:
    raise ValueError("Resume requires the saved num_seeds")
  mutable = {"checkpoint", "checkpoint_mode", "run_dir", "sweep_tag", "validate",
             "log_every", "debug_every", "eval_every", "eval_runs", "eval_videos",
             "eval_video_every", "compile_cache_dir", "wandb_finish_timeout",
             "logger", "resume_every", "resume_keep",
             "snapshot", "model_overrides"}
  changed = [key for key, value in meta["config"].items()
             if key not in mutable and cfg.get(key) != value]
  if changed:
    raise ValueError(f"Resume changes training configuration: {', '.join(sorted(changed))}")
  for key in ("snapshot", "model_overrides"):
    if cfg.get(key) and cfg[key] != meta["config"].get(key):
      raise ValueError(f"Resume must use the saved {key}")


def restore_training(run, meta) -> None:
  path = Path(meta["_dir"])
  if run.buffer.max_transitions != meta["capacity"]:
    raise ValueError("Resume requires the saved replay capacity")
  with ocp.StandardCheckpointer() as ckpt:
    run.state = ckpt.restore(path / "training_state", run.state)
  for i, (buffer, saved) in enumerate(zip(run.buffer.seeds, meta["replay"])):
    with np.load(path / f"replay{i}.npz", allow_pickle=False) as archive:
      fields = {key: archive[key] for key in archive.files}
    if fields:
      buffer._allocate(fields)
      for key, values in fields.items():
        buffer._fields[key][:len(values)] = values
    buffer._size, buffer._write = saved["size"], saved["write"]
    buffer.rng.bit_generator.state = saved["rng"]
  run.collector.rng.bit_generator.state = meta["collector_rng"]
  run.progress = meta["progress"]
  run.reference, _ = unstack(run.graphdef, run.state, 0)
