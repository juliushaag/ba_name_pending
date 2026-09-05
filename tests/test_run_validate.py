"""barl_jax.config.run.validate_run -- the run-level half of preflight
validation (barl_jax.config.tree.validate covers the model half). Each case
is a run config invalid in exactly one way, checked in isolation against a
baseline that is otherwise valid."""

from barl_jax.config.run import run_defaults, validate_run


def base(**overrides) -> dict:
  return {**run_defaults(), "model": "SAC", "env": "HalfCheetah", **overrides}


def test_valid_defaults_pass():
  assert validate_run(base(), num_seeds=1) == []


def test_bool_is_rejected_where_int_is_declared():
  assert validate_run(base(batch_size=True), num_seeds=1)


def test_non_finite_utd_ratio_is_rejected():
  assert validate_run(base(utd_ratio=float("nan")), num_seeds=1)
  assert validate_run(base(utd_ratio=float("inf")), num_seeds=1)
  assert validate_run(base(utd_ratio=0.0), num_seeds=1)


def test_unknown_checkpoint_mode_is_rejected():
  assert validate_run(base(checkpoint_mode="restart"), num_seeds=1)


def test_resume_without_a_checkpoint_is_rejected():
  assert validate_run(base(checkpoint_mode="resume", checkpoint=None), num_seeds=1)


def test_resume_with_a_checkpoint_is_accepted():
  assert validate_run(base(checkpoint_mode="resume", checkpoint="/tmp/x"), num_seeds=1) == []


def test_unknown_lr_decay_is_rejected():
  assert validate_run(base(lr_decay="quadratic"), num_seeds=1)


def test_unknown_logger_kind_is_rejected():
  assert validate_run(base(logger="tensorboard"), num_seeds=1)


def test_zero_cadence_keys_are_rejected():
  """log_every/debug_every/eval_every are modulo denominators in trainer.py;
  zero there is a ZeroDivisionError, not "log nothing"."""
  for key in ("log_every", "debug_every", "eval_every", "rollout_steps",
              "num_envs", "batch_size", "buffer_size", "total_env_steps"):
    assert validate_run(base(**{key: 0}), num_seeds=1), key


def test_random_steps_and_lr_warmup_may_be_zero():
  assert validate_run(base(random_steps=0, lr_warmup_steps=0), num_seeds=1) == []


def test_eval_video_every_and_eval_videos_may_be_null():
  assert validate_run(base(eval_video_every=None, eval_videos=None), num_seeds=1) == []


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
  problems = validate_run(base(utd_ratio=-1.0), num_seeds=1)
  assert any("utd_ratio" in p and "-1.0" in p for p in problems)
