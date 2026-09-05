"""Strict, atomic presentation JSON for run results (`objective.json` and
anything else a driver reads without touching wandb).

"Strict" means every number in the file is a value `json.loads` on any
language can read back: `nan`/`inf`/`-inf` (real, meaningful floats in this
codebase, e.g. bounds) are for exact computational state, never for a result
a driver ranks -- there, "no real number yet" is `null`, plus an explicit
`*_reason` saying why, so a consumer never mistakes "not computed" for "the
worst score"."""

import json
import math
import os
import tempfile
from pathlib import Path


def presentable(value):
  """`float('nan'/'inf'/'-inf') -> None`; everything else passed through.
  Only for presentation payloads -- exact/metadata JSON (checkpoint specs,
  config snapshots) keeps real infinities where they are meaningful."""
  if isinstance(value, float) and not math.isfinite(value):
    return None
  return value


def write_json_atomic(path, payload: dict) -> None:
  """Writes `payload` to `path` via a temp file in the same directory,
  flushed and fsynced before an atomic rename -- a reader never observes a
  partially written file, and a crash mid-write leaves the previous version
  (or nothing) rather than a truncated one."""
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
  try:
    with os.fdopen(fd, "w") as f:
      json.dump(payload, f, indent=2, sort_keys=True)
      f.flush()
      os.fsync(f.fileno())
    os.replace(tmp_name, path)
  except BaseException:
    try:
      os.unlink(tmp_name)
    except OSError:
      pass
    raise


def read_json(path) -> dict | None:
  """`None` for missing or malformed files -- callers must handle "no result
  yet" and "corrupt result" without mistaking either for a real one."""
  path = Path(path)
  try:
    with open(path) as f:
      return json.load(f)
  except (OSError, json.JSONDecodeError):
    return None
