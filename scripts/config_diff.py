"""Diffs a model's currently-resolved config against a frozen `resolved`
snapshot (a saved checkpoint, or a bare snapshot JSON file), keyed the same
way `scripts/print_config.py --json` and a run's own `snapshot` are --
`params.<k>`, `training.<k>`, `components.<slot>.<k>`.

Every changed key is reported with *where the new value comes from*
(`barl_jax.config.tree.ConfigTree.explain`'s provenance chain), so a changed
class default and a changed explicit override read differently even though
both just look like "the number is different":

    uv run python scripts/config_diff.py SAC --against checkpoints/sac/halfcheetah/<ts>/seed0/best
    uv run python scripts/config_diff.py SAC --against checkpoints/sac/halfcheetah/<ts>/checkpoints/resume
    uv run python scripts/config_diff.py SAC --against old_snapshot.json --set params.lr=1e-4
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import barl_jax  # noqa: F401 -- populates the policy/component registries
import barl_jax.config.tree as model_config

_MISSING = object()


def _read_snapshot(path: str) -> dict:
  """A policy checkpoint's meta.json, a training root's pointer/gen/training.json,
  or a bare `{"resolved": {...}}` (or resolved-dict-only) JSON file."""
  p = Path(path)
  if p.is_file():
    data = json.loads(p.read_text())
    return data.get("resolved", data)
  if (p / "pointer.json").exists():
    from barl_jax.training_checkpoint import read_training
    snapshot = (read_training(p).get("config") or {}).get("snapshot") or {}
    return snapshot.get("resolved", {})
  if (p / "meta.json").exists():
    meta = json.loads((p / "meta.json").read_text())
    return (meta.get("snapshot") or {}).get("resolved", {})
  raise SystemExit(f"{p}: not a snapshot file, policy checkpoint, or training resume root")


def _flatten_explain(explained: dict) -> dict:
  flat = {}
  for k, info in explained["params"].items():
    flat[f"params.{k}"] = info
  for k, info in explained["training"].items():
    flat[f"training.{k}"] = info
  for slot, block in explained["components"].items():
    flat[f"components.{slot}.class"] = {"value": block["class"], "source": "structure", "overrides": []}
    if block["tier"]:
      flat[f"components.{slot}.tier"] = {"value": block["tier"], "source": "structure", "overrides": []}
    for k, info in block["kwargs"].items():
      flat[f"components.{slot}.{k}"] = info
  return flat


def _changed_at(source: str) -> str:
  if source.startswith("decl "):
    return "declaration/default"
  if source.startswith("override:"):
    return "explicit override"
  if source.startswith("tier:"):
    return "tier"
  return "model/profile YAML"


def diff(old_resolved: dict, name: str, *, config: model_config.ConfigTree) -> list[dict]:
  new_resolved = config.resolved(name)
  new_explained = _flatten_explain(config.explain(name))
  rows = []
  for key in sorted(set(old_resolved) | set(new_resolved)):
    if key in ("policy", "profile"):
      continue
    old_v = old_resolved.get(key, _MISSING)
    new_v = new_resolved.get(key, _MISSING)
    if old_v == new_v:
      continue
    info = new_explained.get(key)
    rows.append({
      "key": key,
      "old": None if old_v is _MISSING else old_v,
      "new": None if new_v is _MISSING else new_v,
      "status": "added" if old_v is _MISSING else "removed" if new_v is _MISSING else "changed",
      "new_source": info["source"] if info else None,
      "changed_at": _changed_at(info["source"]) if info else None,
    })
  return rows


if __name__ == "__main__":
  args = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  args.add_argument("model", help="Policy class name or config/models/ alias")
  args.add_argument("--against", required=True,
                    help="Snapshot JSON, policy checkpoint, or training resume root to diff against")
  args.add_argument("--set", action="append", metavar="PATH=VALUE",
                    help="Override the *current* side before resolving, as --set on print_config.py")
  args.add_argument("--json", action="store_true")
  kwargs = args.parse_args()

  overrides = {}
  for assignment in kwargs.set or []:
    path, sep, value = assignment.partition("=")
    if not sep:
      raise SystemExit(f"--set {assignment!r} is not of the form path=value")
    import yaml
    overrides[path.strip()] = yaml.safe_load(value)

  old = _read_snapshot(kwargs.against)
  config = model_config.ConfigTree(overrides=overrides)
  rows = diff(old, kwargs.model, config=config)

  if kwargs.json:
    print(json.dumps(rows, indent=2, sort_keys=True, default=str))
  elif not rows:
    print(f"{kwargs.model}: no difference from {kwargs.against}")
  else:
    width = max(len(r["key"]) for r in rows)
    for r in rows:
      print(f"{r['key']:<{width}}  {r['status']:<8}  {r['old']!r:<20} -> {r['new']!r:<20}  "
            f"[{r['changed_at'] or '?'}: {r['new_source'] or 'removed'}]")
