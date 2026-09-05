"""Training-state resume at an iteration boundary, restarting environment episodes."""

import json
from pathlib import Path

import numpy as np
import orbax.checkpoint as ocp

from barl_jax.checkpoints import checkpoint_directory, save_policy
from barl_jax.multiseed import unstack

VERSION = 1


def save_training(run, path, progress):
  with checkpoint_directory(path) as destination:
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


def read_training(path):
  path = Path(path)
  if not (path / "training.json").exists():
    raise ValueError(f"{path} is a policy checkpoint; use checkpoint_mode: weights")
  meta = json.loads((path / "training.json").read_text())
  if meta.get("format_version") != VERSION:
    raise ValueError(f"Unsupported training checkpoint version {meta.get('format_version')}")
  return meta


def validate_resume(cfg, num_seeds, meta):
  if num_seeds != meta["num_seeds"]:
    raise ValueError("Resume requires the saved num_seeds")
  mutable = {"checkpoint", "checkpoint_mode", "run_dir", "sweep_tag", "validate",
             "log_every", "debug_every", "eval_every", "eval_runs", "eval_videos",
             "eval_video_every", "compile_cache_dir", "wandb_finish_timeout",
             "snapshot", "model_overrides"}
  changed = [key for key, value in meta["config"].items()
             if key not in mutable and cfg.get(key) != value]
  if changed:
    raise ValueError(f"Resume changes training configuration: {', '.join(sorted(changed))}")
  for key in ("snapshot", "model_overrides"):
    if cfg.get(key) and cfg[key] != meta["config"].get(key):
      raise ValueError(f"Resume must use the saved {key}")


def restore_training(run, path, meta):
  path = Path(path).absolute()
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
