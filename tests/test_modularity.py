import json
import subprocess
import sys

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

import barl_jax.config.tree as model_config
from barl_jax import optimization, trainer
from barl_jax.checkpoints import load_policy, save_policy
from barl_jax.components import ConfigOwner
from barl_jax.config.model import ModelSpec
from barl_jax.config.tree import ConfigTree
from barl_jax.env import Environment
from barl_jax.hparam import Config
from barl_jax.policy import Policy
from barl_jax.training_checkpoint import save_training


class ToyEnv(gym.Env):
  observation_space = gym.spaces.Box(-10, 10, (3,), dtype=np.float32)
  action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)

  def __init__(self, value=2.0):
    self.value = value
    self.closed = False

  def reset(self, *, seed=None, options=None):
    super().reset(seed=seed)
    self.steps = 0
    return np.full(3, self.value, np.float32), {}

  def step(self, action):
    self.steps += 1
    return np.full(3, self.value, np.float32), 1.0, False, self.steps >= 4, {}

  def close(self):
    self.closed = True


class ToyFamily(Environment):
  stats = {"return_min": -10.0, "return_max": 10.0}

  def __init__(self, value=7.0):
    self.value = value
    self.created = []

  def get_env(self, *, render: bool = False):
    env = ToyEnv(self.value)
    self.created.append(env)
    return env

  def get_dataset(self):
    raise NotImplementedError


def tiny_config(**overrides):
  return ConfigTree(overrides={"components.actor.latent_dim": 8,
                               "components.critic.latent_dim": 8, **overrides})


def make_policy(config=None):
  return Policy.make("SAC", ToyEnv(), config=config or tiny_config(), env_stats=ToyFamily().get_env_stats(ToyEnv()))


def run_config(tmp_path):
  from barl_jax.config.run import run_defaults
  return {**run_defaults(), "model": "SAC", "env": "Toy", "num_seeds": 1,
          "batch_size": 4, "total_env_steps": 16, "random_steps": 0,
          "buffer_size": 16, "eval_video_every": None,
          "run_dir": str(tmp_path), "compile_cache_dir": str(tmp_path / "cache"),
          "model_overrides": tiny_config().overrides}


def test_imports_do_not_initialize_runtime():
  code = '''
import sys
import barl_jax
import barl_jax.config.tree
assert not set(("jax", "mujoco", "torch", "orbax")) & sys.modules.keys()
from barl_jax.env import Environment
assert "DogRun" in Environment.available()
assert not set(("jax", "mujoco", "dm_control", "mani_skill", "humanoid_bench")) & sys.modules.keys()
'''
  subprocess.run([sys.executable, "-c", code], check=True)


def test_async_factory_keeps_instance_settings():
  family = ToyFamily(9.0)
  env = family.get_vector_env(2, asynchronous=True)
  try:
    obs, _ = env.reset(seed=0)
    np.testing.assert_array_equal(obs, np.full((2, 3), 9.0))
  finally:
    env.close()


def test_declaration_overrides_ancestor_in_subclass():
  class Base(ConfigOwner):
    knob = Config(1)
  class Child(Base):
    knob = Config(2)
  assert Base()._decls["knob"].default == 1
  assert Child()._decls["knob"].default == 2


def test_config_dict_resolves_each_field_at_its_own_flat_key():
  """A ConfigDict field overrides exactly like a standalone Config would --
  grouping only changes how the class declares and the instance reads back,
  never the override surface a run's `--set` sees."""
  from barl_jax.config.toolkit import ConfigMixin
  from barl_jax.hparam import ConfigDict

  class Owner(ConfigOwner):
    mlp = ConfigDict(activation='gelu', norm='none', norm_kwargs={})

  owner = Owner()
  owner.resolve_config(ConfigMixin("", {"activation": "silu"}))
  assert owner.mlp.activation == "silu"          # overridden at its flat key
  assert owner.mlp.norm == "none"                # untouched default
  assert owner.mlp.todict() == {"activation": "silu", "norm": "none", "norm_kwargs": {}}
  hash(owner.mlp)                                # identity-hashable regardless of the dict field


def test_config_dict_group_is_replaced_wholesale_by_a_subclass():
  from barl_jax.config.toolkit import ConfigMixin
  from barl_jax.hparam import ConfigDict

  class Base(ConfigOwner):
    mlp = ConfigDict(activation='gelu', norm='none')
  class Child(Base):
    mlp = ConfigDict(activation='relu', norm='layer')

  child = Child()
  child.resolve_config(ConfigMixin("", {}))
  assert child.mlp.activation == "relu" and child.mlp.norm == "layer"


def test_model_spec_is_isolated_from_config_mutation():
  config = tiny_config(**{"params.policy_delay": 2})
  spec = ModelSpec.resolve("SAC", config)
  config.overrides["params.policy_delay"] = 7
  p = Policy.make("SAC", ToyEnv(), spec=spec, env_stats=ToyFamily().get_env_stats(ToyEnv()))
  assert p.policy_delay == 2
  data = spec.data()
  data["params"]["policy_delay"] = 20
  assert p.resolve_params()["policy_delay"] == 2


def test_incompatible_objective_is_rejected_before_building():
  config = tiny_config(**{"components.critic.class": "NCLQLCritic"})
  assert any("does not support 'td'" in p for p in model_config.validate("SAC", config))
  with pytest.raises(TypeError, match="does not support 'td'"):
    make_policy(config)


def test_running_normalization_handles_unequal_batches_and_constant_channels():
  p = make_policy()
  first = jnp.array([[1., 3., 5.], [3., 3., 7.]])
  second = jnp.array([[5., 3., 9.]])
  p.update_observation_stats(first)
  p.update_observation_stats(second)
  all_obs = np.concatenate([first, second])
  np.testing.assert_allclose(p.obs_mean[...], all_obs.mean(0))
  np.testing.assert_allclose(p.obs_std[...], all_obs.std(0), rtol=1e-6)
  assert np.isfinite(p.normalize_observations(second)).all()
  p.normalize_obs = False
  p.update_observation_stats(second)
  assert float(p.obs_count[...]) == 3
  np.testing.assert_array_equal(p.normalize_observations(second), second)


def test_checkpoint_uses_saved_class_and_does_not_mutate_config(tmp_path, monkeypatch):
  policy = make_policy()
  policy.update_observation_stats(jnp.ones((3, 3)))
  save_policy(policy, tmp_path / "policy")
  original_init = ConfigTree.__init__
  def changed_defaults(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    if self.snapshot is None:
      self.models["SAC"] = {"policy": "Nonexistent"}
  monkeypatch.setattr(ConfigTree, "__init__", changed_defaults)
  restored = load_policy(tmp_path / "policy", ToyEnv(), ToyFamily().get_env_stats(ToyEnv()))
  assert type(restored) is type(policy)
  assert ConfigTree().policy_class_name("SAC") == "Nonexistent"
  for a, b in zip(jax.tree.leaves(nnx.state(policy)), jax.tree.leaves(nnx.state(restored))):
    if jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key):
      a, b = jax.random.key_data(a), jax.random.key_data(b)
    np.testing.assert_array_equal(a, b)


def test_future_checkpoint_version_fails_clearly(tmp_path):
  policy = make_policy()
  save_policy(policy, tmp_path / "policy")
  path = tmp_path / "policy/meta.json"
  meta = json.loads(path.read_text())
  meta["format_version"] = 999
  path.write_text(json.dumps(meta))
  with pytest.raises(ValueError, match="Unsupported policy checkpoint version"):
    load_policy(path.parent, ToyEnv())


def test_setup_failure_closes_already_created_environments(tmp_path, monkeypatch):
  family = ToyFamily()
  monkeypatch.setattr(Environment, "make", lambda name: family)
  def fail(*args, **kwargs):
    raise RuntimeError("build failure")
  monkeypatch.setattr(Policy, "make", fail)
  with pytest.raises(RuntimeError, match="build failure"):
    trainer.setup(run_config(tmp_path), 1)
  assert family.created and all(env.closed for env in family.created)


def test_validation_failure_closes_run(tmp_path, monkeypatch):
  family = ToyFamily()
  monkeypatch.setattr(Environment, "make", lambda name: family)
  run = trainer.setup(run_config(tmp_path), 1)
  def fail(*args, **kwargs):
    raise RuntimeError("rollout failure")
  monkeypatch.setattr(run.collector, "collect", fail)
  with pytest.raises(RuntimeError, match="rollout failure"):
    trainer.validate(run)
  assert all(env.closed for env in family.created)


def test_training_resume_restores_next_update_and_replay_rng(tmp_path, monkeypatch):
  family = ToyFamily()
  monkeypatch.setattr(Environment, "make", lambda name: family)
  cfg = run_config(tmp_path)
  with trainer.setup(cfg, 1) as original:
    original.collector.random_fill(original.rollout_env, 8, seed=0, sink=original.buffer)
    batch = original.buffer.sample(4)
    original.state, *_ = optimization.seeded_step(original.graphdef, original.state, batch, (), jnp.asarray(0))
    progress = {"env_steps": 8, "grad_steps": 1, "iteration": 1, "best": [1.]}
    save_training(original, tmp_path / "resume", progress)
    expected_batch = original.buffer.sample(4)
    expected_state, *_ = optimization.seeded_step(original.graphdef, original.state, expected_batch, (), jnp.asarray(1))
    resume_cfg = {**cfg, "checkpoint": str(tmp_path / "resume"), "checkpoint_mode": "resume"}
    with trainer.setup(resume_cfg, 1) as restored:
      assert restored.progress == progress
      actual_batch = restored.buffer.sample(4)
      for key in actual_batch:
        np.testing.assert_array_equal(actual_batch[key], expected_batch[key])
      actual_state, *_ = optimization.seeded_step(restored.graphdef, restored.state, actual_batch, (), jnp.asarray(1))
      for a, b in zip(jax.tree.leaves(actual_state), jax.tree.leaves(expected_state)):
        if jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key):
          a, b = jax.random.key_data(a), jax.random.key_data(b)
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
      assert restored.collector.state is None


def test_resume_rejects_an_incompatible_checkpoint_version(tmp_path, monkeypatch):
  """An old-format resume checkpoint must fail loudly at read_training, before
  trainer.setup ever touches the run's own state -- not be silently reinterpreted
  under the new replay field name."""
  import barl_jax.training_checkpoint as training_checkpoint

  family = ToyFamily()
  monkeypatch.setattr(Environment, "make", lambda name: family)
  cfg = run_config(tmp_path)
  with trainer.setup(cfg, 1) as original:
    original.collector.random_fill(original.rollout_env, 8, seed=0, sink=original.buffer)
    save_training(original, tmp_path / "resume", {"env_steps": 8, "grad_steps": 0,
                                                   "iteration": 0, "best": [1.]})

  # Downgrade the just-written generation's own format_version, as if it had
  # been written by the pre-rename code.
  gen_dir = training_checkpoint._generation_dir(tmp_path / "resume", 0)
  meta_path = gen_dir / "training.json"
  meta = json.loads(meta_path.read_text())
  meta["format_version"] = 1
  meta_path.write_text(json.dumps(meta))

  resume_cfg = {**cfg, "checkpoint": str(tmp_path / "resume"), "checkpoint_mode": "resume"}
  with pytest.raises(ValueError, match="Unsupported training checkpoint version"):
    trainer.setup(resume_cfg, 1)


class MemoryLogger:
  def __init__(self, *args, step_keys, **kwargs):
    self.steps = dict.fromkeys(step_keys, 0)
    self.finished = False

  def get_step(self, name):
    return self.steps[name]

  def advance(self, values):
    for key, value in values.items():
      self.steps[key] += value

  def finish(self):
    self.finished = True

  def __getattr__(self, name):
    return lambda *args, **kwargs: None


def test_warmup_failure_closes_logger_and_all_environments(tmp_path, monkeypatch):
  import barl_jax.logger as logging
  family = ToyFamily()
  loggers = []
  def logger(kind, *args, **kwargs):
    instance = MemoryLogger(*args, **kwargs)
    loggers.append(instance)
    return instance
  monkeypatch.setattr(Environment, "make", lambda name: family)
  monkeypatch.setattr(logging, "make_logger", logger)
  def fail(*args):
    raise RuntimeError("warmup failed")
  monkeypatch.setattr(trainer, "warmup", fail)
  run = trainer.setup(run_config(tmp_path), 1)
  with pytest.raises(RuntimeError, match="warmup failed"):
    trainer.train(run)
  assert loggers[0].finished
  assert all(env.closed for env in family.created)


def test_complete_training_writes_resumable_state(tmp_path, monkeypatch):
  import barl_jax.logger as logging
  family = ToyFamily()
  monkeypatch.setattr(Environment, "make", lambda name: family)
  monkeypatch.setattr(logging, "make_logger", lambda kind, *a, **kw: MemoryLogger(*a, **kw))
  monkeypatch.setattr(trainer, "warmup", lambda *args: {})
  cfg = {**run_config(tmp_path), "total_env_steps": 2, "buffer_size": 2}
  run = trainer.setup(cfg, 1)
  trainer.train(run)
  result = json.loads((tmp_path / "objective.json").read_text())
  assert result["status"] == "finished"
  assert all(env.closed for env in family.created)
  path = tmp_path / "checkpoints/resume"
  with trainer.setup({**cfg, "checkpoint": str(path), "checkpoint_mode": "resume"}, 1) as restored:
    assert restored.progress["env_steps"] == 2
    assert restored.progress["grad_steps"] == 4


@pytest.mark.parametrize("kind", ["local", "none"])
def test_training_runs_without_wandb(tmp_path, monkeypatch, kind):
  """`logger: local`/`logger: none` are real barl_jax.logger sinks, not test
  fakes -- neither one imports or touches wandb, so both are the acceptance
  case for "a training smoke run works without W&B installed"."""
  family = ToyFamily()
  monkeypatch.setattr(Environment, "make", lambda name: family)
  monkeypatch.setattr(trainer, "warmup", lambda *args: {})
  cfg = {**run_config(tmp_path), "logger": kind, "total_env_steps": 2, "buffer_size": 2,
         "eval_every": 1, "eval_runs": 1}
  run = trainer.setup(cfg, 1)
  trainer.train(run)

  result = json.loads((tmp_path / "objective.json").read_text())
  assert result["status"] == "finished"
  assert all(env.closed for env in family.created)

  if kind == "local":
    lines = (tmp_path / "metrics.jsonl").read_text().strip().splitlines()
    assert lines
    for line in lines:
      json.loads(line)  # every row is strict JSON, parseable with no special-casing
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert "best_reward" in summary["per_seed"]["0"]


def test_unversioned_checkpoints_are_rejected(tmp_path):
  (tmp_path / "meta.json").write_text(json.dumps({"model": "SAC"}))
  with pytest.raises(ValueError, match="Unsupported policy checkpoint version"):
    load_policy(tmp_path, ToyEnv())


def test_failed_checkpoint_write_keeps_previous_generation(tmp_path):
  from barl_jax.checkpoints import checkpoint_directory
  path = tmp_path / "checkpoint"
  path.mkdir()
  (path / "marker").write_text("previous")
  with pytest.raises(RuntimeError):
    with checkpoint_directory(path) as candidate:
      (candidate / "marker").write_text("partial")
      raise RuntimeError("disk failure")
  assert (path / "marker").read_text() == "previous"


def _resume_run(tmp_path, monkeypatch):
  from barl_jax.env import Environment
  family = ToyFamily()
  monkeypatch.setattr(Environment, "make", lambda name: family)
  cfg = run_config(tmp_path)
  run = trainer.setup(cfg, 1)
  run.collector.random_fill(run.rollout_env, 4, seed=0, sink=run.buffer)
  return run


def test_resume_generations_are_numbered_and_pointer_tracks_the_latest(tmp_path, monkeypatch):
  from barl_jax.training_checkpoint import read_training
  root = tmp_path / "resume"
  with _resume_run(tmp_path, monkeypatch) as run:
    save_training(run, root, {"env_steps": 1}, keep=2)
    save_training(run, root, {"env_steps": 2}, keep=2)
    assert (root / "gen-000000").is_dir()
    assert (root / "gen-000001").is_dir()
    assert json.loads((root / "pointer.json").read_text()) == {"generation": 1}
    assert read_training(root)["progress"] == {"env_steps": 2}
    assert read_training(root)["_dir"] == str(root / "gen-000001")


def test_resume_retention_keeps_only_the_newest_generations(tmp_path, monkeypatch):
  with _resume_run(tmp_path, monkeypatch) as run:
    root = tmp_path / "resume"
    for step in range(5):
      save_training(run, root, {"env_steps": step}, keep=2)
    remaining = sorted(p.name for p in root.glob("gen-*"))
    assert remaining == ["gen-000003", "gen-000004"]


def test_resume_retention_never_deletes_the_only_recoverable_generation(tmp_path, monkeypatch):
  """A write that crashes after `checkpoint_directory` commits but before
  pointer.json is published would otherwise strand an unreferenced generation
  -- but nothing publishes-then-fails here, so the true guarantee under test
  is the ordering itself: prune runs strictly after the new pointer is
  written, so the previous generation is never the only one on disk in the
  window between "new one committed" and "pointer updated"."""
  from barl_jax.training_checkpoint import _existing_generations
  with _resume_run(tmp_path, monkeypatch) as run:
    root = tmp_path / "resume"
    save_training(run, root, {"env_steps": 0}, keep=1)
    save_training(run, root, {"env_steps": 1}, keep=1)
    assert _existing_generations(root) == [1]


def test_disk_budget_rejects_a_save_with_insufficient_free_space(tmp_path, monkeypatch):
  import shutil as shutil_module

  from barl_jax.training_checkpoint import _check_disk_budget

  class FakeUsage:
    free = 100

  monkeypatch.setattr(shutil_module, "disk_usage", lambda path: FakeUsage())
  with pytest.raises(OSError, match="Refusing to start"):
    _check_disk_budget(tmp_path, required_bytes=10_000)
  # A previous generation of 0 bytes (or none at all) skips the check --
  # nothing to estimate the new write's size from.
  _check_disk_budget(tmp_path, required_bytes=0)


@pytest.mark.parametrize("utd_ratio", [0.5, 2.0])
def test_chunked_training_counts_primitive_steps(tmp_path, monkeypatch, utd_ratio):
  import barl_jax.logger as logging
  family = ToyFamily()
  monkeypatch.setattr(Environment, "make", lambda name: family)
  monkeypatch.setattr(logging, "make_logger", lambda kind, *a, **kw: MemoryLogger(*a, **kw))
  monkeypatch.setattr(trainer, "warmup", lambda *args: {})
  cfg = {**run_config(tmp_path), "total_env_steps": 8, "num_envs": 2,
         "rollout_steps": 2, "chunk_size": 4, "utd_ratio": utd_ratio}
  run = trainer.setup(cfg, 1)
  trainer.train(run)
  assert run.progress["env_steps"] == 8
  assert run.progress["grad_steps"] == round(8 * utd_ratio)
  assert run.buffer.min_transitions() == 2
  assert np.array_equal(run.buffer.seeds[0]._fields["horizon"][:2], [4, 4])


def test_expanded_mlp_snapshot_restores_without_the_removed_selector(tmp_path):
  original = make_policy()
  snapshot = original._snapshot
  snapshot["resolved"]["components.actor.mlp_preset"] = "layer"
  snapshot["resolved"]["components.critic.mlp_preset"] = "crossq"
  restored = make_policy(ConfigTree(snapshot=snapshot))
  assert restored.actor.mlp.activation == original.actor.mlp.activation
  assert restored.critic.mlp.norm == original.critic.mlp.norm
  save_policy(restored, tmp_path / "policy")
  loaded = load_policy(tmp_path / "policy", ToyEnv(), ToyFamily().get_env_stats(ToyEnv()))
  for a, b in zip(jax.tree.leaves(nnx.state(original)), jax.tree.leaves(nnx.state(loaded))):
    if jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key):
      a, b = jax.random.key_data(a), jax.random.key_data(b)
    np.testing.assert_array_equal(a, b)
