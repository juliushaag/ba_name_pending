"""scripts/scaffold.py -- generates a new environment or algorithm and
smoke-tests it in a temporary project. Covers: it writes the file, the
smoke test actually catches a broken generated/pointed-at class, and it
refuses to overwrite an existing file."""

import importlib.util
import pathlib

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "scaffold.py"


def load_scaffold():
  spec = importlib.util.spec_from_file_location("scaffold", SCRIPT)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


scaffold = load_scaffold()


def test_scaffold_env_writes_and_smoke_tests(tmp_path):
  scaffold.scaffold_env("QuickstartCartPole", "CartPole-v1", tmp_path)
  path = tmp_path / "quickstartcartpole_env.py"
  assert path.exists()
  assert "class QuickstartCartPole(Environment):" in path.read_text()


def test_scaffold_env_refuses_to_overwrite(tmp_path):
  scaffold.scaffold_env("Dup", "CartPole-v1", tmp_path)
  with pytest.raises(SystemExit, match="already exists"):
    scaffold.scaffold_env("Dup", "CartPole-v1", tmp_path)


def test_scaffold_env_smoke_test_catches_an_invalid_gym_id(tmp_path):
  with pytest.raises(Exception):
    scaffold.scaffold_env("BadGymId", "NoSuchEnv-v0", tmp_path)
  # The file is still written (a scaffold to fix, not silently discarded) --
  # only the smoke test failed.
  assert (tmp_path / "badgymid_env.py").exists()


def test_scaffold_algorithm_writes_and_smoke_tests(tmp_path):
  scaffold.scaffold_algorithm("QuickstartAlgo", "StochasticActor", "C51Critic", tmp_path)
  path = tmp_path / "quickstartalgo.yaml"
  assert path.exists()
  text = path.read_text()
  assert "policy: MaxEntRLPolicy" in text
  assert "StochasticActor" in text and "C51Critic" in text


def test_scaffold_algorithm_refuses_to_overwrite(tmp_path):
  scaffold.scaffold_algorithm("Dup", "StochasticActor", "C51Critic", tmp_path)
  with pytest.raises(SystemExit, match="already exists"):
    scaffold.scaffold_algorithm("Dup", "StochasticActor", "C51Critic", tmp_path)


def test_scaffold_algorithm_smoke_test_catches_an_unregistered_class(tmp_path):
  with pytest.raises(SystemExit, match="not in registered components"):
    scaffold.scaffold_algorithm("BadActor", "NoSuchActor", "C51Critic", tmp_path)
  assert (tmp_path / "badactor.yaml").exists()
