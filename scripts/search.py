#!/usr/bin/env python3
"""Search spaces: axes with sampling distributions, sampled into the same
`{algo, env, seed, model.*, run.*}` combos grid.py already materializes.

An HPO trial is structurally the same thing as one grid combo -- the grid's
`itertools.product` just cannot produce one. So this module only *samples*;
materialization, snapshot freezing, the config check, the executor handoff and
the run path are grid.py / submit.py / executors.py / online_learning.py's,
untouched. That is where the search belongs rather than next to the trainer: a
sampled trial has to pass the same `config_tree.validate()` and snapshot gates
a hand-written grid point does, or it was never a real configuration.

An axis spec, one entry per axis in `axes:`:

    algo:  {type: categorical, values: [SAC, DIME]}       # required
    env:   {type: categorical, values: [HalfCheetah]}     # required
    seed:  {type: categorical, values: [0, 1, 2]}         # optional base seed
    run.batch_size: {type: int, low: 64, high: 1024, log: true}
    run.utd_ratio:  {type: float, low: 0.5, high: 4.0}
    model.components.critic.tier: {type: categorical, values: [low, medium, large]}

A `run.*` value lands in the run entry and a `model.*` value in
`model_overrides:`, exactly where grid.py puts them -- so a sampled point is
checked, snapshotted and logged like any `--set` override, and a trial's wandb
config is as complete an account of what ran as a hand-written run's.

Sampling is `random`-based so the seam carries no new dependency; the whole
backend is the one `sample_trial` call, so an Optuna/TPE or ASHA sampler is a
drop-in replacement of that function rather than a change to anything else
here. Random search is the only baseline that needs no tuning itself.
"""

import importlib.util
import math
import random
from pathlib import Path

import yaml

from barl_jax.config.toolkit import CONFIG_DIR, load_yaml_with_includes

RESERVED = ("algo", "env", "seed")

_KNOWN_TYPES = ("categorical", "int", "float", "bool")


def _check_axis(name: str, spec) -> None:
  if not isinstance(spec, dict) or "type" not in spec:
    raise ValueError(f"search axis {name!r} needs a 'type': one of {_KNOWN_TYPES}")
  t = spec["type"]
  if t not in _KNOWN_TYPES:
    raise ValueError(f"search axis {name!r}: type {t!r} not one of {_KNOWN_TYPES}")
  if name not in RESERVED and not name.startswith(("model.", "run.")):
    raise ValueError(
      f"search axis {name!r} must be {RESERVED}, or be prefixed 'model.' or 'run.'")
  if t == "categorical" and not spec.get("values"):
    raise ValueError(f"search axis {name!r}: categorical needs a non-empty 'values'")
  if t in ("int", "float") and ("low" not in spec or "high" not in spec):
    raise ValueError(f"search axis {name!r}: {t} needs 'low' and 'high'")


def load_search(path: str, root=CONFIG_DIR) -> dict:
  """A `$()`-resolved search spec, validated enough to fail before a trial does."""
  spec = load_yaml_with_includes(path, config_root=root)
  axes = spec.get("axes")
  if not isinstance(axes, dict) or not axes:
    raise ValueError(f"{path}: needs a non-empty 'axes' dict")
  for name, axis in axes.items():
    _check_axis(name, axis)
  for required in ("algo", "env"):
    if required not in axes or axes[required].get("type") != "categorical":
      raise ValueError(f"{path}: axes must include a categorical {required!r}")
  return spec


def sample_axis(name: str, spec: dict, rng: random.Random):
  t = spec["type"]
  if t == "categorical":
    return spec["values"][rng.randrange(len(spec["values"]))]
  if t == "bool":
    return bool(rng.randint(0, 1))
  if t == "int":
    low, high = int(spec["low"]), int(spec["high"])
    if low > high:
      raise ValueError(f"search axis {name!r}: low {low} > high {high}")
    if spec.get("log"):
      return int(round(math.exp(rng.uniform(math.log(low), math.log(high)))))
    return rng.randint(low, high)
  low, high = float(spec["low"]), float(spec["high"])
  if spec.get("log"):
    return math.exp(rng.uniform(math.log(low), math.log(high)))
  return rng.uniform(low, high)


def sample_trial(axes: dict, rng: random.Random) -> dict:
  """One point of the search space as a full grid combo: `algo`/`env`/`seed`
  plus dotted `model.*`/`run.*` overrides. This is the seam an Optuna sampler
  would replace wholesale."""
  return {name: sample_axis(name, spec, rng) for name, spec in axes.items()}


def sample_trials(axes: dict, n: int, rng: random.Random) -> list[dict]:
  """`n` distinct sampled combos. Deduplicated by exact value, so a small space
  does not waste runs on the same point twice; a space smaller than `n` returns
  fewer trials rather than repeating."""
  target = min(n, _discrete_space_size(axes) or n)
  seen, out = set(), []
  while len(out) < target:
    trial = sample_trial(axes, rng)
    key = frozenset(trial.items())
    if key in seen:
      continue
    seen.add(key)
    out.append(trial)
  return out


def _discrete_space_size(axes: dict) -> int | None:
  """The exact number of unique points when every axis is finitely enumerable.

  `categorical`, `bool`, and `int` axes are discrete; a `float` axis is not.
  This is what lets `sample_trials()` stop once it has exhausted a finite
  search space instead of looping forever on duplicates."""
  total = 1
  for spec in axes.values():
    t = spec["type"]
    if t == "categorical":
      total *= len(spec["values"])
    elif t == "bool":
      total *= 2
    elif t == "int":
      total *= int(spec["high"]) - int(spec["low"]) + 1
    else:
      return None
  return total


def check_searchability(spec: dict, *, tree=None) -> list[str]:
  """A search axis on `model.<slot>.<key>`/`model.params.<key>` must name a
  declared **HParam** (a searchable knob) of every algo the search varies.
  `Config` declarations -- structural or just deliberately non-searchable --
  are refused: a hyperparameter search is over explicit optimization knobs,
  not every value that happens to be configurable. Runs after a
  barl_jax import (hpo.py), since resolving a key needs the class registries."""
  import barl_jax  # noqa: F401 -- registries
  from barl_jax.config.tree import ConfigTree
  from barl_jax.networks.registry import get_component
  from barl_jax.policy import Policy

  axes, algos = spec.get("axes") or {}, (spec.get("axes") or {}).get("algo", {}).get("values") or []
  problems = []
  tree = tree or ConfigTree()
  for axis in axes:
    if not axis.startswith("model."):
      continue
    rest = axis[len("model."):]
    name = rest.rpartition(".")[2]
    for algo in algos:
      policy_name, _profile = tree.resolve_model_name(algo)
      if not tree.policy_cfg(policy_name):
        continue
      pn = tree.policy_class_name(policy_name)
      try:
        cls = Policy.class_for(pn)
      except KeyError:
        cls = None
      if cls is None:
        continue
      if rest.startswith("params."):
        key = rest[len("params."):]
        if key in getattr(cls, "_configs", {}):
          problems.append(f"{axis}: '{key}' is a non-search Config on {pn}, not a searchable HParam")
        elif key not in getattr(cls, "_hparams", {}):
          problems.append(f"{axis}: '{key}' is not a declared HParam of {pn}")
      elif rest.startswith("components."):
        parts = rest.split(".")
        if len(parts) == 2:
          continue
        slot, key = parts[1], parts[2]
        if key == "tier":
          problems.append(f"{axis}: a slot's tier is structural capacity, not a searchable HParam")
          continue
        if slot not in cls.SLOTS:
          continue
        # slot resolution is model-name-keyed (ConfigTree.slot_class), not
        # policy-class-keyed.
        comp = get_component(tree.slot_class(policy_name, slot))
        if key in getattr(comp, "_configs", {}):
          problems.append(f"{axis}: '{key}' is a non-search Config on {comp.__name__}, "
                          f"not a searchable HParam")
        elif key not in getattr(comp, "_hparams", {}):
          # A style-surface key (activation, dtype, ...) is owned by the
          # component's mlp_preset/time_emb selector -- structural, not searchable.
          owner = next((f"{n!s}" for n, d in getattr(comp, "_configs", {}).items()
                        if d.style is not None and key in d.style.get(d.default, {})),
                       None)
          if owner is not None:
            problems.append(f"{axis}: '{key}' is part of {comp.__name__}'s {owner} "
                            f"style, not a searchable HParam")
          else:
            problems.append(f"{axis}: '{key}' is not a declared HParam of {comp.__name__}")
  return problems


if __name__ == "__main__":
  import argparse
  import sys

  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("--search", required=True, help="path to a search spec YAML")
  ap.add_argument("--n-trials", type=int, default=5)
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()

  spec = load_search(args.search)
  for i, trial in enumerate(sample_trials(spec["axes"], args.n_trials, random.Random(args.seed))):
    print(f"[{i:03d}] " + "  ".join(f"{k}={v}" for k, v in trial.items()))
  sys.exit(0)
