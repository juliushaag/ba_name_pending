"""Policy-independent action selection (barl_jax.execution): Default, Best-of-N
(BoN), Mean-of-N. Pure-function tests over fake candidate/scorer callables
first (the reducers' actual arithmetic, isolated from any policy), then
integration with a real actor policy (SAC) and the one policy without a
scorer (NCLQL) for the capability-rejection path."""

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import ACT_DIM, BATCH, OBS_DIM, make_env_stats

from barl_jax.execution import ExecutionSpec, act, validate_execution_spec
from barl_jax.policy import Policy

# ---------------------------------------------------------------------------
# ExecutionSpec syntax
# ---------------------------------------------------------------------------

def test_unknown_mode_is_rejected():
  with pytest.raises(ValueError, match="mode must be one of"):
    ExecutionSpec(mode="ensemble")


@pytest.mark.parametrize("n", [0, -1, 1.5, True])
def test_bad_candidate_count_is_rejected(n):
  with pytest.raises(ValueError, match="num_candidates"):
    ExecutionSpec(mode="bon", num_candidates=n)


def test_default_requires_exactly_one_candidate():
  with pytest.raises(ValueError, match="default"):
    ExecutionSpec(mode="default", num_candidates=2)


# ---------------------------------------------------------------------------
# act() over fake candidate/scorer callables -- the reducer arithmetic itself
# ---------------------------------------------------------------------------

CHUNK = 3


def _fake_infer_step(base: float):
  """A deterministic, key-dependent "policy": candidate value is
  `base + key[1]` broadcast over a (batch, chunk, act) plan, so different
  split keys produce visibly different, checkable candidates."""
  def infer_step(observation, key):
    value = base + key[1].astype(jnp.float32)
    plan = jnp.full((observation.shape[0], CHUNK, ACT_DIM), value, jnp.float32)
    return plan, {"value": jnp.full((observation.shape[0],), value, jnp.float32)}
  return infer_step


def test_default_mode_calls_infer_step_directly_with_the_caller_key():
  infer_step = _fake_infer_step(0.0)
  obs = jnp.zeros((BATCH, OBS_DIM))
  key = jax.random.PRNGKey(7)
  action, extras = act(infer_step, None, obs, key, ExecutionSpec())
  expected, expected_extras = infer_step(obs, key)
  np.testing.assert_array_equal(action, expected)
  np.testing.assert_array_equal(extras["value"], expected_extras["value"])


def test_mean_of_n_is_the_exact_arithmetic_mean():
  infer_step = _fake_infer_step(0.0)
  obs = jnp.zeros((BATCH, OBS_DIM))
  key = jax.random.PRNGKey(0)
  n = 5
  spec = ExecutionSpec(mode="mean", num_candidates=n)
  action, extras = act(infer_step, None, obs, key, spec)

  keys = jax.random.split(key, n)
  expected = jnp.mean(jnp.stack([infer_step(obs, k)[0] for k in keys]), axis=0)
  np.testing.assert_allclose(action, expected, atol=1e-6)
  assert extras == {}  # per-candidate extras do not describe the averaged action


def test_mean_of_n_only_averages_the_candidate_axis():
  """Distinct batch rows must not bleed into each other's mean."""
  def infer_step(observation, key):
    row = jnp.arange(observation.shape[0], dtype=jnp.float32) + key[1].astype(jnp.float32)
    return jnp.broadcast_to(row[:, None, None], (observation.shape[0], CHUNK, ACT_DIM)), {}

  obs = jnp.zeros((4, OBS_DIM))
  key = jax.random.PRNGKey(3)
  n = 4
  action, _ = act(infer_step, None, obs, key, ExecutionSpec(mode="mean", num_candidates=n))
  keys = jax.random.split(key, n)
  expected_per_row = jnp.mean(jnp.stack([jnp.arange(4, dtype=jnp.float32) + k[1] for k in keys]), axis=0)
  np.testing.assert_allclose(action[:, 0, 0], expected_per_row, atol=1e-6)


def test_n_equals_1_is_bit_identical_to_default_for_both_reducers():
  infer_step = _fake_infer_step(0.0)
  obs = jnp.zeros((BATCH, OBS_DIM))
  key = jax.random.PRNGKey(11)
  default_action, _ = act(infer_step, None, obs, key, ExecutionSpec())

  mean_action, _ = act(infer_step, None, obs, key, ExecutionSpec(mode="mean", num_candidates=1))
  np.testing.assert_array_equal(mean_action, default_action)

  def score_fn(observation, action_plan):
    return jnp.zeros((observation.shape[0],))
  bon_action, _ = act(infer_step, score_fn, obs, key, ExecutionSpec(mode="bon", num_candidates=1))
  np.testing.assert_array_equal(bon_action, default_action)


def _keyed_infer_step(batch: int):
  """Each candidate's plan is `jax.random.uniform(key, ())` broadcast over
  `(batch, chunk, act)` -- a value distinguishable per split key, with no
  assumption about a key's internal bit layout."""
  def infer_step(observation, key):
    idx = jax.random.uniform(key, ())
    return jnp.full((batch, CHUNK, ACT_DIM), idx, jnp.float32), {}
  return infer_step


def _candidate_values(key, n: int):
  """The exact per-candidate idx values `_keyed_infer_step` will produce,
  computed the same way `execution._candidate_keys` does, so a test can pick
  a score function that prefers a specific candidate deterministically."""
  return jnp.array([jax.random.uniform(k, ()) for k in jax.random.split(key, n)])


def test_best_of_n_picks_the_known_winner_per_batch_row():
  """Candidate c's plan encodes idx[c]; row 0's score prefers candidate 2,
  row 1's prefers candidate 0 -- a per-row winner, not one winner broadcast
  over the whole batch. score_fn receives the N*batch-folded plan in
  candidate-major order (see execution.act's BoN branch), so it un-folds to
  (n, batch, ...) to compare against a per-row target."""
  n, batch = 4, 2
  infer_step = _keyed_infer_step(batch)
  key = jax.random.PRNGKey(0)
  idx = _candidate_values(key, n)
  winners = [2, 0]
  desired = jnp.array([idx[winners[0]], idx[winners[1]]])

  def score_fn(observation, action_plan):
    grid = action_plan.reshape(n, batch, CHUNK, ACT_DIM)[:, :, 0, 0]
    return (-jnp.abs(grid - desired[None, :])).reshape(n * batch)

  obs = jnp.zeros((batch, OBS_DIM))
  action, extras = act(infer_step, score_fn, obs, key, ExecutionSpec(mode="bon", num_candidates=n))
  np.testing.assert_allclose(action[0, 0, 0], idx[winners[0]], atol=1e-6)
  np.testing.assert_allclose(action[1, 0, 0], idx[winners[1]], atol=1e-6)
  np.testing.assert_allclose(extras["selected_score"], 0.0, atol=1e-6)
  np.testing.assert_array_equal(extras["num_invalid"], jnp.zeros(batch))


def test_best_of_n_selects_an_entire_plan_not_per_timestep():
  """chunk_size > 1: BoN picks one whole candidate plan, never a different
  candidate's action at a different timestep within the same chunk."""
  n, batch = 3, 1

  def infer_step(observation, key):
    idx = jax.random.uniform(key, ())
    # Each candidate's own plan is internally inconsistent across time unless
    # the whole plan is kept together -- t=0 says its index, t=1 says its
    # negation, so a per-timestep mix would show up as a broken equality.
    plan = jnp.stack([jnp.full((ACT_DIM,), idx), jnp.full((ACT_DIM,), -idx)])
    return plan[None], {}

  def score_fn(observation, action_plan):
    return action_plan[:, 0, 0]  # prefers the candidate with the largest idx

  obs = jnp.zeros((batch, OBS_DIM))
  key = jax.random.PRNGKey(0)
  action, _ = act(infer_step, score_fn, obs, key, ExecutionSpec(mode="bon", num_candidates=n))
  assert action[0, 0, 0] == -action[0, 1, 0]  # both timesteps came from the same winning candidate


def test_best_of_n_breaks_all_invalid_ties_deterministically():
  n, batch = 4, 3
  infer_step = _keyed_infer_step(batch)

  def all_nan_score(observation, action_plan):
    return jnp.full((action_plan.shape[0],), jnp.nan)

  obs = jnp.zeros((batch, OBS_DIM))
  key = jax.random.PRNGKey(0)
  spec = ExecutionSpec(mode="bon", num_candidates=n)
  action1, extras1 = act(infer_step, all_nan_score, obs, key, spec)
  action2, extras2 = act(infer_step, all_nan_score, obs, key, spec)
  np.testing.assert_array_equal(action1, action2)  # deterministic, not arbitrary
  np.testing.assert_array_equal(extras1["num_invalid"], jnp.full((batch,), n))
  np.testing.assert_array_equal(extras2["num_invalid"], jnp.full((batch,), n))


def test_best_of_n_masks_non_finite_scores_among_valid_ones():
  n, batch = 3, 1
  infer_step = _keyed_infer_step(batch)
  key = jax.random.PRNGKey(0)
  idx = _candidate_values(key, n)
  nan_candidate = int(jnp.argmax(idx))  # the highest-idx candidate scores NaN

  def score_fn(observation, action_plan):
    value = action_plan[:, 0, 0]
    return jnp.where(value == idx[nan_candidate], jnp.nan, value)

  obs = jnp.zeros((batch, OBS_DIM))
  action, extras = act(infer_step, score_fn, obs, key, ExecutionSpec(mode="bon", num_candidates=n))
  assert float(action[0, 0, 0]) != float(idx[nan_candidate])
  assert extras["num_invalid"][0] == 1


# ---------------------------------------------------------------------------
# validate_execution_spec -- capability / action-space checks
# ---------------------------------------------------------------------------

class _FakePolicy:
  def __init__(self, *, action_discrete=False, supports_bon=False):
    self.action_discrete = action_discrete
    self.supports_bon = supports_bon


def test_mean_is_rejected_for_discrete_action_spaces():
  problems = validate_execution_spec(ExecutionSpec(mode="mean", num_candidates=4),
                                     policy=_FakePolicy(action_discrete=True))
  assert problems and "discrete" in problems[0]


def test_bon_is_rejected_without_a_scorer():
  problems = validate_execution_spec(ExecutionSpec(mode="bon", num_candidates=4),
                                     policy=_FakePolicy(supports_bon=False))
  assert problems and "score" in problems[0]


def test_default_and_mean_need_no_scorer():
  policy = _FakePolicy(supports_bon=False)
  assert validate_execution_spec(ExecutionSpec(), policy=policy) == []
  assert validate_execution_spec(ExecutionSpec(mode="mean", num_candidates=4), policy=policy) == []


# ---------------------------------------------------------------------------
# Integration: a real actor policy (SAC) and the one policy with no scorer
# ---------------------------------------------------------------------------

class FakeEnv:
  observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
  action_space = gym.spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)


def _make(name):
  return Policy.make(name, FakeEnv(), env_stats=make_env_stats(), seed=0)


def test_sac_supports_bon_and_scores_are_finite():
  policy = _make("SAC")
  assert policy.supports_bon
  infer = policy.inference_fn(ExecutionSpec(mode="bon", num_candidates=4))
  obs = jnp.zeros((BATCH, OBS_DIM))
  action, extras = infer(obs, jax.random.PRNGKey(0))
  assert action.shape == (BATCH, policy.chunk_size, ACT_DIM)
  assert bool(jnp.all(jnp.isfinite(extras["selected_score"])))


def test_sac_mean_of_n_matches_the_manual_reduction():
  policy = _make("SAC")
  spec = ExecutionSpec(mode="mean", num_candidates=4)
  infer = policy.inference_fn(spec)
  obs = jnp.zeros((BATCH, OBS_DIM))
  key = jax.random.PRNGKey(0)
  action, extras = infer(obs, key)
  assert extras == {}

  default_infer = policy.inference_fn()
  keys = jax.random.split(key, 4)
  candidates = jnp.stack([default_infer(obs, k)[0] for k in keys])
  np.testing.assert_allclose(action, jnp.mean(candidates, axis=0), atol=1e-5)


def test_sac_default_mode_is_unaffected_by_passing_an_explicit_default_spec():
  policy = _make("SAC")
  obs = jnp.zeros((BATCH, OBS_DIM))
  key = jax.random.PRNGKey(0)
  a1, e1 = policy.inference_fn()(obs, key)
  a2, e2 = policy.inference_fn(ExecutionSpec())(obs, key)
  np.testing.assert_array_equal(a1, a2)
  assert e1.keys() == e2.keys()


def test_nclql_has_no_scorer_and_bon_is_rejected_at_preflight():
  policy = _make("NCLQL")
  assert not policy.supports_bon
  problems = validate_execution_spec(ExecutionSpec(mode="bon", num_candidates=4), policy=policy)
  assert problems

  # default/mean still work without a critic-backed scorer.
  obs = jnp.zeros((BATCH, OBS_DIM))
  key = jax.random.PRNGKey(0)
  action, _ = policy.inference_fn(ExecutionSpec(mode="mean", num_candidates=3))(obs, key)
  assert action.shape == (BATCH, policy.chunk_size, ACT_DIM)
