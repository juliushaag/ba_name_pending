"""Training result sidecar serialization."""
import json


def test_objective_sidecar(tmp_path):
  """trainer._write_objective writes a rankable, strict-JSON sidecar: a real
  reward as a plain float, and "no evaluation yet" as `null` plus a reason --
  never a non-finite float, which plain `json.loads` elsewhere cannot parse."""
  from barl_jax import trainer

  path = trainer._write_objective(
    str(tmp_path), model="SAC", env="HalfCheetah", seed=0, num_seeds=1,
    env_steps=100_000, best=[42.0], status="finished")
  with open(path) as f:
    data = json.load(f)
  assert data["objective"] == 42.0
  assert data["objective_reason"] is None
  assert data["status"] == "finished"

  bad = trainer._write_objective(
    str(tmp_path / "other"), model="SAC", env="HalfCheetah", seed=1, num_seeds=1,
    env_steps=0, best=[float("-inf")], status="failed")
  with open(bad) as f:
    data = json.load(f)
  assert data["objective"] is None
  assert data["objective_reason"]
  assert data["best"] == [None]
