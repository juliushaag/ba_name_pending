"""The search layer: sampled trials are valid grid combos, and the objective
sidecar is written where an HPO driver can read it.

The claim this pins is the design's whole point: a search trial is structurally
identical to a grid combo, so materializing one must produce the same
self-contained, snapshot-carrying config a `--set` grid point produces -- which
is what lets an HPO driver reuse grid.py / executors.py / online_learning.py
without any of them learning what search is.
"""

import json
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import grid
from search import load_search, sample_trial, sample_trials

from barl_jax.config.toolkit import CONFIG_DIR
SPEC = str(CONFIG_DIR / "search" / "demo.yaml")


def test_sampled_values_stay_inside_their_space():
  axes = load_search(SPEC)["axes"]
  rng = random.Random(0)
  for _ in range(200):
    t = sample_trial(axes, rng)
    assert t["algo"] in axes["algo"]["values"]
    assert t["env"] in axes["env"]["values"]
    assert axes["run.batch_size"]["low"] <= t["run.batch_size"] <= axes["run.batch_size"]["high"]
    assert axes["run.utd_ratio"]["low"] <= t["run.utd_ratio"] <= axes["run.utd_ratio"]["high"]
    assert isinstance(t["run.batch_size"], int)
    assert isinstance(t["run.utd_ratio"], float)


def test_trials_are_distinct():
  trials = sample_trials(load_search(SPEC)["axes"], 50, random.Random(1))
  assert len({frozenset(t.items()) for t in trials}) == len(trials)


def test_finite_search_space_returns_fewer_trials_than_requested():
  axes = {
    "algo": {"type": "categorical", "values": ["SAC"]},
    "env": {"type": "categorical", "values": ["HalfCheetah"]},
    "seed": {"type": "categorical", "values": [0, 1]},
  }
  trials = sample_trials(axes, 10, random.Random(0))
  assert len(trials) == 2
  assert {t["seed"] for t in trials} == {0, 1}


def test_trial_materializes_like_a_grid_combo(tmp_path):
  """A sampled trial flows through grid.materialize exactly like a grid point:
  run.* keys land in the run entry, model.* keys in model_overrides, and the
  snapshot is taken *under those overrides* -- so it replays the sampled point
  whatever config/ says later."""
  axes = load_search(SPEC)["axes"]
  trial = sample_trial(axes, random.Random(2))
  trial["seed"] = 0

  defaults = {"sweep_tag": "test", "num_seeds": 1}
  sweep_dir = str(tmp_path / "sweep")
  os.makedirs(sweep_dir, exist_ok=True)
  runs = grid.materialize([trial], ["SAC_HalfCheetah_1seeds_t000"], defaults=defaults,
                          sweep_dir=sweep_dir, num_seeds=1)

  config_path = os.path.join(sweep_dir, runs[0].config)
  assert os.path.exists(config_path)
  import yaml
  with open(config_path) as f:
    cfg = yaml.safe_load(f)

  entry = cfg["runs"][0]
  # run.* -> run entry, model.* -> model_overrides (grid.materialize's split).
  for key in ("batch_size", "utd_ratio"):
    assert entry[key] == trial[f"run.{key}"]
  for key in ("components.critic.gamma", "params.entropy_scale"):
    assert entry["model_overrides"][key] == trial[f"model.{key}"]
  # The snapshot is resolved under this trial's own overrides.
  resolved = cfg["snapshot"]["resolved"]
  assert resolved["components.critic.gamma"] == trial["model.components.critic.gamma"]
  assert resolved["params.entropy_scale"] == trial["model.params.entropy_scale"]


def test_search_rejects_structural_config_axes():
  """A search axis must target a declared HParam; a non-search Config key (a
  slot's `tier`, an actor's `dtype`) is refused."""
  from search import check_searchability
  axes = dict(load_search(SPEC)["axes"])
  axes["model.components.actor.dtype"] = {"type": "categorical", "values": ["float32", "bfloat16"]}
  axes["model.components.critic.tier"] = {"type": "categorical", "values": ["medium", "large"]}
  problems = check_searchability({"axes": axes})
  assert problems, "expected the structural axes to be refused"
  covered = "\n".join(problems)
  assert "mlp_preset style" in covered and "tier is structural" in covered
  # The demo spec itself is clean.
  assert not check_searchability(load_search(SPEC))


def test_search_rejects_structural_config_axes_for_alias_algos(tmp_path):
  """Alias algos resolve through aliases.yaml before their policy/component
  classes are checked, so the structural/HParam gate must hold for aliases too."""
  from search import check_searchability
  from barl_jax.config.toolkit import CONFIG_DIR
  from barl_jax.config.tree import ConfigTree

  root = tmp_path / "config"
  root.mkdir()
  for name in ("experiments", "models"):
    (root / name).mkdir()

  for name in ("aliases.yaml", "sac.yaml"):
    (root / "models" / name).write_text((CONFIG_DIR / "models" / name).read_text())

  (root / "models" / "sac.yaml").write_text(
    (root / "models" / "sac.yaml").read_text() + """
  profiles:
    big:
      actor: {tier: large}
""")
  (root / "models" / "aliases.yaml").write_text(
    "aliases:\n  sac_big: {policy: SAC, profile: big}\n")

  tree = ConfigTree(root)
  spec = {"axes": {
    "algo": {"type": "categorical", "values": ["sac_big"]},
    "env": {"type": "categorical", "values": ["HalfCheetah"]},
    "model.components.actor.tier": {"type": "categorical", "values": ["medium", "large"]},
  }}

  problems = check_searchability(spec, tree=tree)
  assert any("tier is structural" in problem for problem in problems)


def test_objective_sidecar(tmp_path):
  """trainer._write_objective writes a rankable JSON beside the run, and ranks
  non-finite below any real reward."""
  from barl_jax import trainer

  path = trainer._write_objective(
    str(tmp_path), model="SAC", env="HalfCheetah", seed=0, num_seeds=1,
    env_steps=100_000, best=[42.0], status="finished")
  with open(path) as f:
    data = json.load(f)
  assert data["objective"] == 42.0
  assert data["status"] == "finished"

  bad = trainer._write_objective(
    str(tmp_path / "other"), model="SAC", env="HalfCheetah", seed=1, num_seeds=1,
    env_steps=0, best=[float("-inf")], status="failed")
  with open(bad) as f:
    assert json.load(f)["objective"] == float("-inf")
