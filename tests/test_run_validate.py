"""barl_jax.config.run.validate_run -- the run-level half of preflight
validation (model validation lives alongside it in config/run.py). Each case
is a run config invalid in exactly one way, checked in isolation against a
baseline that is otherwise valid."""

import pytest

from barl_jax.config.run import RunConfig, run_defaults, validate_run


def base(**overrides) -> dict:
  return {**run_defaults(), "model": "SAC", "env": "HalfCheetah", **overrides}


def test_valid_defaults_pass():
  assert validate_run(base(), num_seeds=1) == []


def params_problems(**params) -> list[str]:
  """Problems of a SAC run whose policy params are overridden with `params`."""
  cfg = RunConfig.defaults("SAC", "HalfCheetah")
  return cfg.replace(model_config=cfg.model_config.overridden({"params": params})).problems()


def test_bool_is_rejected_where_int_is_declared():
  assert validate_run(base(total_env_steps=True), num_seeds=1)
  with pytest.raises(TypeError):
    params_problems(batch_size=True)


def test_run_level_keys_no_longer_accept_policy_params():
  for key in ("batch_size", "utd_ratio", "lr_warmup_steps", "lr_decay"):
    with pytest.raises(KeyError):
      RunConfig.from_dict({"model": "SAC", "env": "HalfCheetah", key: 1})


def test_non_finite_utd_ratio_is_rejected():
  assert params_problems(utd_ratio=float("nan"))
  assert params_problems(utd_ratio=float("inf"))
  assert params_problems(utd_ratio=0.0)


def test_unknown_checkpoint_mode_is_rejected():
  assert validate_run(base(checkpoint_mode="restart"), num_seeds=1)


def test_resume_without_a_checkpoint_is_rejected():
  assert validate_run(base(checkpoint_mode="resume", checkpoint=None), num_seeds=1)


def test_resume_with_a_checkpoint_is_accepted():
  assert validate_run(base(checkpoint_mode="resume", checkpoint="/tmp/x"), num_seeds=1) == []


def test_unknown_lr_decay_is_rejected():
  assert params_problems(lr_decay="quadratic")


def test_unknown_logger_kind_is_rejected():
  assert validate_run(base(logger="tensorboard"), num_seeds=1)


def test_zero_cadence_keys_are_rejected():
  """log_every/debug_every/eval_every are modulo denominators in trainer.py;
  zero there is a ZeroDivisionError, not "log nothing"."""
  for key in ("log_every", "debug_every", "eval_every", "rollout_steps",
              "num_envs", "buffer_size", "total_env_steps"):
    assert validate_run(base(**{key: 0}), num_seeds=1), key
  assert params_problems(batch_size=0)


def test_random_steps_and_lr_warmup_may_be_zero():
  assert params_problems(random_steps=0) == []
  assert params_problems(random_steps=-1)
  assert params_problems(lr_warmup_steps=0) == []
  assert params_problems(lr_warmup_steps=-1)


def test_eval_video_every_may_be_null():
  assert validate_run(base(eval_video_every=None), num_seeds=1) == []


def test_rollout_steps_below_num_envs_is_rejected():
  """rollout_steps // num_envs floors to 0 -- an iteration that collects
  nothing and a loop that never advances, not a small iteration."""
  assert validate_run(base(rollout_steps=1, num_envs=4), num_seeds=1)


def test_rollout_steps_equal_to_num_envs_is_accepted():
  assert validate_run(base(rollout_steps=4, num_envs=4), num_seeds=1) == []


def test_num_seeds_must_be_a_positive_int():
  assert validate_run(base(), num_seeds=0)
  assert validate_run(base(), num_seeds=True)


def test_problems_name_the_key_and_the_value():
  problems = params_problems(utd_ratio=-1.0)
  assert any("utd_ratio" in p and "-1.0" in p for p in problems)


def test_vector_step_budgets_must_be_divisible_by_num_envs():
  assert validate_run(base(rollout_steps=5, num_envs=4), num_seeds=1)
  cfg = RunConfig.defaults("SAC", "HalfCheetah").replace(num_envs=4, rollout_steps=8)

  def with_random_steps(n):
    return cfg.replace(model_config=cfg.model_config.overridden({"params": {"random_steps": n}}))

  assert any("random_steps" in p for p in with_random_steps(5).problems())
  assert with_random_steps(4).problems() == []
