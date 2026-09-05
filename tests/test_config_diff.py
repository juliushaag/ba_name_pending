"""scripts/config_diff.py's diff() -- compares a frozen `resolved` snapshot
against the live-resolved config, and must say *why* each changed key
differs (declaration/default vs. an explicit override vs. a tier/profile
choice), not merely that the numbers differ."""

import importlib.util
import pathlib

import barl_jax.config.tree as model_config

SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "config_diff.py"


def load_config_diff():
  spec = importlib.util.spec_from_file_location("config_diff", SCRIPT)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


config_diff = load_config_diff()


def test_identical_snapshot_has_no_diff():
  tree = model_config.ConfigTree()
  old = tree.resolved("SAC")
  assert config_diff.diff(old, "SAC", config=tree) == []


def test_explicit_override_is_labeled_as_such():
  baseline = model_config.ConfigTree().resolved("SAC")
  overridden = model_config.ConfigTree(overrides={"params.policy_delay": 7})
  rows = {r["key"]: r for r in config_diff.diff(baseline, "SAC", config=overridden)}
  assert rows["params.policy_delay"]["status"] == "changed"
  assert rows["params.policy_delay"]["new"] == 7
  assert rows["params.policy_delay"]["changed_at"] == "explicit override"


def test_tier_change_is_labeled_as_a_tier(monkeypatch):
  baseline = model_config.ConfigTree().resolved("SAC")
  changed = model_config.ConfigTree(overrides={"components.critic.tier": "large"})
  rows = {r["key"]: r for r in config_diff.diff(baseline, "SAC", config=changed)}
  assert rows["components.critic.latent_dim"]["changed_at"] == "tier"


def test_a_key_missing_from_the_old_snapshot_is_added():
  tree = model_config.ConfigTree()
  full = tree.resolved("SAC")
  missing_one = {k: v for k, v in full.items() if k != "params.policy_delay"}
  rows = {r["key"]: r for r in config_diff.diff(missing_one, "SAC", config=tree)}
  assert rows["params.policy_delay"]["status"] == "added"
  assert rows["params.policy_delay"]["old"] is None


def test_a_key_no_longer_resolved_is_removed():
  tree = model_config.ConfigTree()
  full = tree.resolved("SAC")
  stale = {**full, "params.long_gone_knob": 1.0}
  rows = {r["key"]: r for r in config_diff.diff(stale, "SAC", config=tree)}
  assert rows["params.long_gone_knob"]["status"] == "removed"
  assert rows["params.long_gone_knob"]["new"] is None
  assert rows["params.long_gone_knob"]["changed_at"] is None
