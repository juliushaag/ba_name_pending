"""Policy-independent action selection (barl_jax.executors): Direct, Best-of-N
(BoN), Mean-of-N, Median-of-N, Top-K-of-N, Axis-Shuffle-of-N, Perturbation-of-N.
Pure-function tests over fake candidate/scorer callables first (the reducers'
actual arithmetic, isolated from any policy), then integration with a real
actor policy (SAC) and the one policy without a scorer (NCLQL) for the
capability-rejection path."""

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import ACT_DIM, BATCH, OBS_DIM, make_env_stats

from barl_jax.components import sanitize_action
from barl_jax.executors import (
  AxisShuffleExecutor,
  AxisShuffleExecutorConfig,
  BestOfNExecutor,
  BestOfNExecutorConfig,
  DirectExecutor,
  DirectExecutorConfig,
  MeanExecutor,
  MeanExecutorConfig,
  MedianExecutor,
  MedianExecutorConfig,
  PerturbExecutor,
  PerturbExecutorConfig,
  TopKExecutor,
  TopKExecutorConfig,
)
from barl_jax.policy import Policy
from barl_jax.registration import get_executor

# ---------------------------------------------------------------------------
# Executor config syntax
# ---------------------------------------------------------------------------

def test_unknown_class_name_is_rejected():
  with pytest.raises(KeyError):
    get_executor("Ensemble")


@pytest.mark.parametrize("n", [0, -1, 1.5, True])
def test_bad_candidate_count_is_rejected(n):
  with pytest.raises(ValueError, match="num_candidates"):
    BestOfNExecutorConfig(num_candidates=n)


def test_direct_config_cannot_carry_a_stray_num_candidates():
  # DirectExecutorConfig has no num_candidates field at all -- a value
  # silently unused by the selected executor is a config mistake, not a
  # no-op to allow, and it is now structurally impossible rather than a
  # runtime rejection.
  with pytest.raises(TypeError):
    DirectExecutorConfig(num_candidates=2)


@pytest.mark.parametrize("k", [0, -1, 1.5, True, 5])
def test_bad_top_k_is_rejected(k):
  with pytest.raises(ValueError, match="top_k"):
    TopKExecutorConfig(num_candidates=4, top_k=k)


def test_mean_config_cannot_carry_a_stray_top_k():
  with pytest.raises(TypeError):
    MeanExecutorConfig(num_candidates=4, top_k=2)


@pytest.mark.parametrize("sigma", [0.0, -0.1, True])
def test_bad_sigma_is_rejected(sigma):
  with pytest.raises(ValueError, match="sigma"):
    PerturbExecutorConfig(num_candidates=4, sigma=sigma)


def test_bon_config_cannot_carry_a_stray_sigma():
  with pytest.raises(TypeError):
    BestOfNExecutorConfig(num_candidates=4, sigma=0.2)


# ---------------------------------------------------------------------------
# select() over fake candidate/scorer callables -- the reducer arithmetic itself
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


def test_direct_executor_calls_infer_step_directly_with_the_caller_key():
  infer_step = _fake_infer_step(0.0)
  obs = jnp.zeros((BATCH, OBS_DIM))
  key = jax.random.PRNGKey(7)
  action, extras = DirectExecutor(DirectExecutorConfig()).select(infer_step, None, obs, key)
  expected, expected_extras = infer_step(obs, key)
  np.testing.assert_array_equal(action, expected)
  np.testing.assert_array_equal(extras["value"], expected_extras["value"])


def test_mean_of_n_is_the_exact_arithmetic_mean():
  infer_step = _fake_infer_step(0.0)
  obs = jnp.zeros((BATCH, OBS_DIM))
  key = jax.random.PRNGKey(0)
  n = 5
  action, extras = MeanExecutor(MeanExecutorConfig(num_candidates=n)).select(infer_step, None, obs, key)

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
  action, _ = MeanExecutor(MeanExecutorConfig(num_candidates=n)).select(infer_step, None, obs, key)
  keys = jax.random.split(key, n)
  expected_per_row = jnp.mean(jnp.stack([jnp.arange(4, dtype=jnp.float32) + k[1] for k in keys]), axis=0)
  np.testing.assert_allclose(action[:, 0, 0], expected_per_row, atol=1e-6)


def test_median_of_n_is_the_exact_median_and_ignores_one_wild_candidate():
  """A single outlier candidate must not drag the result the way it would
  drag a mean -- median-of-5 stays at the 4 well-behaved candidates' value
  regardless of how far the 5th one is."""
  keys = list(jax.random.split(jax.random.PRNGKey(0), 5))

  def infer_step(observation, key, _outlier_key=keys[-1]):
    value = jnp.where(jnp.all(key == _outlier_key), 1000.0, 1.0)
    return jnp.full((observation.shape[0], CHUNK, ACT_DIM), value, jnp.float32), {}

  obs = jnp.zeros((BATCH, OBS_DIM))
  key = jax.random.PRNGKey(0)
  action, extras = MedianExecutor(MedianExecutorConfig(num_candidates=5)).select(infer_step, None, obs, key)
  np.testing.assert_allclose(action, jnp.ones_like(action), atol=1e-6)
  assert extras == {}


def test_n_equals_1_is_bit_identical_to_direct_for_every_reducer():
  infer_step = _fake_infer_step(0.0)
  obs = jnp.zeros((BATCH, OBS_DIM))
  key = jax.random.PRNGKey(11)
  direct_action, _ = DirectExecutor(DirectExecutorConfig()).select(infer_step, None, obs, key)

  mean_action, _ = MeanExecutor(MeanExecutorConfig(num_candidates=1)).select(infer_step, None, obs, key)
  np.testing.assert_array_equal(mean_action, direct_action)

  median_action, _ = MedianExecutor(MedianExecutorConfig(num_candidates=1)).select(infer_step, None, obs, key)
  np.testing.assert_array_equal(median_action, direct_action)

  def score_fn(observation, action_plan):
    return jnp.zeros((observation.shape[0],))
  bon_action, _ = BestOfNExecutor(BestOfNExecutorConfig(num_candidates=1)).select(
    infer_step, score_fn, obs, key)
  np.testing.assert_array_equal(bon_action, direct_action)

  topk_action, _ = TopKExecutor(TopKExecutorConfig(num_candidates=1, top_k=1)).select(
    infer_step, score_fn, obs, key)
  np.testing.assert_array_equal(topk_action, direct_action)

  shuffle_action, _ = AxisShuffleExecutor(AxisShuffleExecutorConfig(num_candidates=1)).select(
    infer_step, None, obs, key)
  np.testing.assert_array_equal(shuffle_action, direct_action)

  perturb_action, _ = PerturbExecutor(PerturbExecutorConfig(num_candidates=1)).select(
    infer_step, score_fn, obs, key)
  np.testing.assert_array_equal(perturb_action, direct_action)


# ---------------------------------------------------------------------------
# Axis-Shuffle-of-N
# ---------------------------------------------------------------------------

def test_axis_shuffle_of_n_recombines_axes_from_independent_candidates():
  """Candidate c's plan is filled with c itself, so every entry of the
  shuffled result must equal the index of *some* candidate -- and, with
  act_dim > 1 candidates to choose from, different axes actually do pick
  different candidates rather than one winner broadcast over the whole
  action (that would be BoN, not this)."""
  n, batch = 6, 4

  def infer_step(observation, key):
    idx = jax.random.uniform(key, ())
    return jnp.full((observation.shape[0], CHUNK, ACT_DIM), idx, jnp.float32), {}

  obs = jnp.zeros((batch, OBS_DIM))
  key = jax.random.PRNGKey(0)
  candidate_values = _candidate_values(key, n)
  action, extras = AxisShuffleExecutor(AxisShuffleExecutorConfig(num_candidates=n)).select(
    infer_step, None, obs, key)

  assert extras == {}  # per-candidate extras do not describe the recombined action
  assert action.shape == (batch, CHUNK, ACT_DIM)
  # Every entry must be one of the n candidate values.
  matches = jnp.isclose(action[..., None], candidate_values, atol=1e-6)
  assert bool(jnp.all(jnp.any(matches, axis=-1)))
  # Not every entry picked the same candidate -- otherwise this is just BoN
  # with a fixed winner.
  assert len(jnp.unique(action)) > 1


def test_axis_shuffle_of_n_only_shuffles_the_candidate_axis():
  """Distinct batch rows must not bleed into each other's axis selection:
  candidate c's plan is `row_index + offset_c`, so whichever candidate row r's
  selection actually landed on, its value minus r must be exactly that
  candidate's offset -- never another row's arange term mixed with a
  candidate's offset that wasn't actually drawn for this row."""
  n, batch = 5, 3

  def infer_step(observation, key):
    row = jnp.arange(observation.shape[0], dtype=jnp.float32)
    offset = key[1].astype(jnp.float32)
    return jnp.broadcast_to((row + offset)[:, None, None], (observation.shape[0], CHUNK, ACT_DIM)), {}

  obs = jnp.zeros((batch, OBS_DIM))
  key = jax.random.PRNGKey(5)
  offsets = jnp.array([k[1] for k in jax.random.split(key, n)], dtype=jnp.float32)
  action, _ = AxisShuffleExecutor(AxisShuffleExecutorConfig(num_candidates=n)).select(
    infer_step, None, obs, key)
  drawn_offset = action[:, 0, 0] - jnp.arange(batch, dtype=jnp.float32)
  matches = jnp.isclose(drawn_offset[:, None], offsets[None, :], atol=1e-4)
  assert bool(jnp.all(jnp.any(matches, axis=-1)))


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
  computed the same way `executors.base._candidate_keys` does, so a test can
  pick a score function that prefers a specific candidate deterministically."""
  return jnp.array([jax.random.uniform(k, ()) for k in jax.random.split(key, n)])


# ---------------------------------------------------------------------------
# Top-K-of-N
# ---------------------------------------------------------------------------

def test_topk_of_n_averages_the_k_best_scoring_candidates():
  """Row 0 ranks candidates 3 > 1 > 2 > 0, row 1 ranks 0 > 2 > 1 > 3 --
  top_k=2 must average exactly each row's top two, not the single best
  (that's BoN) and not all n (that's mean)."""
  n, batch = 4, 2
  infer_step = _keyed_infer_step(batch)
  key = jax.random.PRNGKey(0)
  idx = _candidate_values(key, n)
  scores_by_row = jnp.array([[0.0, 2.0, 1.0, 3.0],
                             [3.0, 0.0, 2.0, 1.0]])

  def score_fn(observation, action_plan):
    return scores_by_row.T.reshape(n * batch)  # (n, batch) -> candidate-major flat

  obs = jnp.zeros((batch, OBS_DIM))
  action, extras = TopKExecutor(TopKExecutorConfig(num_candidates=n, top_k=2)).select(
    infer_step, score_fn, obs, key)
  expected = jnp.array([jnp.mean(idx[jnp.array([3, 1])]), jnp.mean(idx[jnp.array([0, 2])])])
  np.testing.assert_allclose(action[:, 0, 0], expected, atol=1e-6)
  assert extras == {}  # averaged across 2 candidates -- no single one's extras describe it


def test_topk_with_k_equal_1_matches_best_of_n():
  n, batch = 4, 1
  infer_step = _keyed_infer_step(batch)
  key = jax.random.PRNGKey(1)
  idx = _candidate_values(key, n)

  def score_fn(observation, action_plan):
    grid = action_plan[:, 0, 0]
    return -jnp.abs(grid - idx[2])  # prefers candidate 2

  obs = jnp.zeros((batch, OBS_DIM))
  bon_action, _ = BestOfNExecutor(BestOfNExecutorConfig(num_candidates=n)).select(
    infer_step, score_fn, obs, key)
  topk_action, _ = TopKExecutor(TopKExecutorConfig(num_candidates=n, top_k=1)).select(
    infer_step, score_fn, obs, key)
  np.testing.assert_array_equal(topk_action, bon_action)


def test_topk_with_k_equal_n_matches_mean():
  n, batch = 4, 1
  infer_step = _keyed_infer_step(batch)
  key = jax.random.PRNGKey(2)

  def score_fn(observation, action_plan):
    return jnp.zeros((action_plan.shape[0],))

  obs = jnp.zeros((batch, OBS_DIM))
  mean_action, _ = MeanExecutor(MeanExecutorConfig(num_candidates=n)).select(infer_step, None, obs, key)
  topk_action, _ = TopKExecutor(TopKExecutorConfig(num_candidates=n, top_k=n)).select(
    infer_step, score_fn, obs, key)
  np.testing.assert_allclose(topk_action, mean_action, atol=1e-6)


def test_topk_of_n_excludes_non_finite_scores():
  n, batch = 3, 1
  infer_step = _keyed_infer_step(batch)
  key = jax.random.PRNGKey(0)
  idx = _candidate_values(key, n)
  nan_candidate = 1

  def score_fn(observation, action_plan):
    scores = jnp.arange(n, dtype=jnp.float32)
    return jnp.where(jnp.arange(n) == nan_candidate, jnp.nan, scores)

  obs = jnp.zeros((batch, OBS_DIM))
  action, _ = TopKExecutor(TopKExecutorConfig(num_candidates=n, top_k=2)).select(
    infer_step, score_fn, obs, key)
  expected = jnp.mean(idx[jnp.array([0, 2])])  # the two finite candidates
  np.testing.assert_allclose(action[0, 0, 0], expected, atol=1e-6)


def test_best_of_n_picks_the_known_winner_per_batch_row():
  """Candidate c's plan encodes idx[c]; row 0's score prefers candidate 2,
  row 1's prefers candidate 0 -- a per-row winner, not one winner broadcast
  over the whole batch. score_fn receives the N*batch-folded plan in
  candidate-major order (see BestOfNExecutor.select), so it un-folds to
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
  action, extras = BestOfNExecutor(BestOfNExecutorConfig(num_candidates=n)).select(
    infer_step, score_fn, obs, key)
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
  action, _ = BestOfNExecutor(BestOfNExecutorConfig(num_candidates=n)).select(
    infer_step, score_fn, obs, key)
  assert action[0, 0, 0] == -action[0, 1, 0]  # both timesteps came from the same winning candidate


def test_best_of_n_breaks_all_invalid_ties_deterministically():
  n, batch = 4, 3
  infer_step = _keyed_infer_step(batch)

  def all_nan_score(observation, action_plan):
    return jnp.full((action_plan.shape[0],), jnp.nan)

  obs = jnp.zeros((batch, OBS_DIM))
  key = jax.random.PRNGKey(0)
  executor = BestOfNExecutor(BestOfNExecutorConfig(num_candidates=n))
  action1, extras1 = executor.select(infer_step, all_nan_score, obs, key)
  action2, extras2 = executor.select(infer_step, all_nan_score, obs, key)
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
  action, extras = BestOfNExecutor(BestOfNExecutorConfig(num_candidates=n)).select(
    infer_step, score_fn, obs, key)
  assert float(action[0, 0, 0]) != float(idx[nan_candidate])
  assert extras["num_invalid"][0] == 1


# ---------------------------------------------------------------------------
# Perturbation-of-N
# ---------------------------------------------------------------------------

def test_perturbation_of_n_matches_the_manual_reduction():
  n, batch = 6, 3
  infer_step = _fake_infer_step(0.0)
  obs = jnp.zeros((batch, OBS_DIM))
  key = jax.random.PRNGKey(4)
  sigma = 0.3

  def score_fn(observation, action_plan):
    # Prefers whichever candidate lands closest to a fixed positive target --
    # picks out a genuinely perturbed candidate, not necessarily the base.
    return -jnp.abs(action_plan[:, 0, 0] - 0.5)

  action, extras = PerturbExecutor(PerturbExecutorConfig(num_candidates=n, sigma=sigma)).select(
    infer_step, score_fn, obs, key)

  base_action, _ = infer_step(obs, key)
  noise_key = jax.random.fold_in(key, 2)
  noise = sigma * jax.random.normal(noise_key, (n - 1,) + base_action.shape, base_action.dtype)
  perturbed = sanitize_action(base_action[None] + noise, xp=jnp)
  candidates = jnp.concatenate([base_action[None], perturbed], axis=0)
  scores = jnp.stack([score_fn(obs, candidates[c]) for c in range(n)])
  winner = jnp.argmax(scores, axis=0)
  expected = jnp.take_along_axis(candidates, winner[None, :, None, None], axis=0)[0]
  expected_score = jnp.take_along_axis(scores, winner[None, :], axis=0)[0]

  np.testing.assert_allclose(action, expected, atol=1e-6)
  np.testing.assert_allclose(extras["selected_score"], expected_score, atol=1e-6)
  np.testing.assert_array_equal(extras["num_invalid"], jnp.zeros(batch))


def test_perturbation_of_n_never_scores_worse_than_the_base_action():
  """The base action is always in the candidate pool, so a winner picked by
  argmax can never score worse than not perturbing at all."""
  n, batch = 8, 2

  def infer_step(observation, key):
    return jnp.zeros((observation.shape[0], CHUNK, ACT_DIM), jnp.float32), {}

  obs = jnp.zeros((batch, OBS_DIM))
  key = jax.random.PRNGKey(9)

  def score_fn(observation, action_plan):
    return -jnp.abs(action_plan[:, 0, 0])  # 0.0 scores 0 -- the max possible

  action, _ = PerturbExecutor(PerturbExecutorConfig(num_candidates=n, sigma=0.5)).select(
    infer_step, score_fn, obs, key)
  base_action, _ = infer_step(obs, key)
  np.testing.assert_allclose(action, base_action, atol=1e-6)


def test_perturbation_of_n_clips_to_the_valid_action_range():
  n, batch = 20, 1

  def infer_step(observation, key):
    return jnp.full((observation.shape[0], CHUNK, ACT_DIM), 0.9, jnp.float32), {}

  def score_fn(observation, action_plan):
    return action_plan[:, 0, 0]  # prefers the largest value -- pushes toward the clip boundary

  obs = jnp.zeros((batch, OBS_DIM))
  key = jax.random.PRNGKey(1)
  action, _ = PerturbExecutor(PerturbExecutorConfig(num_candidates=n, sigma=2.0)).select(
    infer_step, score_fn, obs, key)
  assert bool(jnp.all(action <= 1.0)) and bool(jnp.all(action >= -1.0))


# ---------------------------------------------------------------------------
# ActionExecutor.validate -- capability / action-space checks
# ---------------------------------------------------------------------------

class _FakePolicy:
  def __init__(self, *, action_discrete=False, supports_bon=False):
    self.action_discrete = action_discrete
    self.supports_bon = supports_bon


@pytest.mark.parametrize("executor_cls,config_cls", [
  (MeanExecutor, MeanExecutorConfig),
  (MedianExecutor, MedianExecutorConfig),
  (PerturbExecutor, PerturbExecutorConfig),
])
def test_elementwise_blends_are_rejected_for_discrete_action_spaces(executor_cls, config_cls):
  policy = _FakePolicy(action_discrete=True, supports_bon=True)
  problems = executor_cls(config_cls(num_candidates=4)).validate(policy=policy)
  assert problems and "discrete" in problems[0]


def test_topk_is_rejected_for_discrete_action_spaces_only_above_k_equals_1():
  policy = _FakePolicy(action_discrete=True, supports_bon=True)
  # top_k=1 never blends -- it is a single winner, like BoN.
  assert TopKExecutor(TopKExecutorConfig(num_candidates=4, top_k=1)).validate(policy=policy) == []
  problems = TopKExecutor(TopKExecutorConfig(num_candidates=4, top_k=2)).validate(policy=policy)
  assert problems and "discrete" in problems[0]


@pytest.mark.parametrize("executor_cls,config", [
  (BestOfNExecutor, BestOfNExecutorConfig(num_candidates=4)),
  (TopKExecutor, TopKExecutorConfig(num_candidates=4, top_k=2)),
  (PerturbExecutor, PerturbExecutorConfig(num_candidates=4)),
])
def test_score_guided_modes_are_rejected_without_a_scorer(executor_cls, config):
  problems = executor_cls(config).validate(policy=_FakePolicy())
  assert problems and "score" in problems[0]


def test_direct_and_blind_reducers_need_no_scorer():
  policy = _FakePolicy(supports_bon=False)
  assert DirectExecutor(DirectExecutorConfig()).validate(policy=policy) == []
  assert MeanExecutor(MeanExecutorConfig(num_candidates=4)).validate(policy=policy) == []
  assert MedianExecutor(MedianExecutorConfig(num_candidates=4)).validate(policy=policy) == []
  assert AxisShuffleExecutor(AxisShuffleExecutorConfig(num_candidates=4)).validate(policy=policy) == []


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
  infer = policy.inference_fn(BestOfNExecutor(BestOfNExecutorConfig(num_candidates=4)))
  obs = jnp.zeros((BATCH, OBS_DIM))
  action, extras = infer(obs, jax.random.PRNGKey(0))
  assert action.shape == (BATCH, policy.chunk_size, ACT_DIM)
  assert bool(jnp.all(jnp.isfinite(extras["selected_score"])))


def test_sac_mean_of_n_matches_the_manual_reduction():
  policy = _make("SAC")
  infer = policy.inference_fn(MeanExecutor(MeanExecutorConfig(num_candidates=4)))
  obs = jnp.zeros((BATCH, OBS_DIM))
  key = jax.random.PRNGKey(0)
  action, extras = infer(obs, key)
  assert extras == {}

  default_infer = policy.inference_fn()
  keys = jax.random.split(key, 4)
  candidates = jnp.stack([default_infer(obs, k)[0] for k in keys])
  np.testing.assert_allclose(action, jnp.mean(candidates, axis=0), atol=1e-5)


def test_sac_median_of_n_matches_the_manual_reduction():
  policy = _make("SAC")
  infer = policy.inference_fn(MedianExecutor(MedianExecutorConfig(num_candidates=5)))
  obs = jnp.zeros((BATCH, OBS_DIM))
  key = jax.random.PRNGKey(0)
  action, extras = infer(obs, key)
  assert extras == {}

  default_infer = policy.inference_fn()
  candidates = jnp.stack([default_infer(obs, k)[0] for k in jax.random.split(key, 5)])
  np.testing.assert_allclose(action, jnp.median(candidates, axis=0), atol=1e-5)


def test_sac_topk_of_n_scores_are_finite_and_shape_is_right():
  """Not a whitebox reconstruction like the other SAC tests here: the critic
  uses batch renormalization (`crossq_mlp`, norm='batchrenorm'),
  whose output depends on how many rows share one forward pass -- scoring the
  same candidates outside of `TopKExecutor.select`'s own N*batch fold does not
  reproduce the same numbers, only a policy-blind reducer (mean/median/
  axis_shuffle, already covered above) can be verified that way against a
  real critic."""
  policy = _make("SAC")
  infer = policy.inference_fn(TopKExecutor(TopKExecutorConfig(num_candidates=5, top_k=2)))
  obs = jnp.zeros((BATCH, OBS_DIM))
  action, extras = infer(obs, jax.random.PRNGKey(0))
  assert action.shape == (BATCH, policy.chunk_size, ACT_DIM)
  assert extras == {}


def test_sac_perturbation_of_n_stays_in_range_and_scores_are_finite():
  policy = _make("SAC")
  infer = policy.inference_fn(PerturbExecutor(PerturbExecutorConfig(num_candidates=8, sigma=0.2)))
  obs = jnp.zeros((BATCH, OBS_DIM))
  action, extras = infer(obs, jax.random.PRNGKey(0))
  assert action.shape == (BATCH, policy.chunk_size, ACT_DIM)
  assert bool(jnp.all(jnp.isfinite(extras["selected_score"])))
  assert bool(jnp.all(action >= -1.0)) and bool(jnp.all(action <= 1.0))


def test_sac_axis_shuffle_of_n_matches_the_manual_reduction():
  policy = _make("SAC")
  infer = policy.inference_fn(AxisShuffleExecutor(AxisShuffleExecutorConfig(num_candidates=4)))
  obs = jnp.zeros((BATCH, OBS_DIM))
  key = jax.random.PRNGKey(0)
  action, extras = infer(obs, key)
  assert extras == {}

  default_infer = policy.inference_fn()
  candidates = jnp.stack([default_infer(obs, k)[0]
                          for k in jax.random.split(key, 4)])
  winner = jax.random.randint(jax.random.fold_in(key, 1), action.shape, 0, 4)
  expected = jnp.take_along_axis(candidates, winner[None], axis=0)[0]
  np.testing.assert_allclose(action, expected, atol=1e-6)


def test_sac_default_mode_is_unaffected_by_passing_an_explicit_direct_executor():
  policy = _make("SAC")
  obs = jnp.zeros((BATCH, OBS_DIM))
  key = jax.random.PRNGKey(0)
  a1, e1 = policy.inference_fn()(obs, key)
  a2, e2 = policy.inference_fn(DirectExecutor(DirectExecutorConfig()))(obs, key)
  np.testing.assert_array_equal(a1, a2)
  assert e1.keys() == e2.keys()


def test_nclql_has_no_scorer_and_score_guided_modes_are_rejected_at_preflight():
  policy = _make("NCLQL")
  assert not policy.supports_bon
  for executor in (BestOfNExecutor(BestOfNExecutorConfig(num_candidates=4)),
                   TopKExecutor(TopKExecutorConfig(num_candidates=4, top_k=2)),
                   PerturbExecutor(PerturbExecutorConfig(num_candidates=4))):
    problems = executor.validate(policy=policy)
    assert problems, type(executor).__name__

  # direct/mean/median/axis_shuffle still work without a critic-backed scorer.
  obs = jnp.zeros((BATCH, OBS_DIM))
  key = jax.random.PRNGKey(0)
  for executor in (MeanExecutor(MeanExecutorConfig(num_candidates=3)),
                   MedianExecutor(MedianExecutorConfig(num_candidates=3)),
                   AxisShuffleExecutor(AxisShuffleExecutorConfig(num_candidates=3))):
    action, _ = policy.inference_fn(executor)(obs, key)
    assert action.shape == (BATCH, policy.chunk_size, ACT_DIM), type(executor).__name__
