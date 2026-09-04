"""The `Policy` contract, over every model in `config/models/`.

Each of the five is a different algorithm, but every one has to satisfy the
same bargain with the training loop: `train_step` returns finite losses keyed
exactly by `create_optimizer()`, and every trainable leaf has an owner in
`optimizer_labels()`. Nothing here asserts a specific numerical behaviour --
that is what the per-algorithm tests own -- only that the contract holds.

This is also SDAC's only behavioural coverage: it is never otherwise
constructed or stepped by the suite.
"""

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from barl_jax.config_tree import ConfigTree
from barl_jax.metrics import PRIMARY
from barl_jax.policy import TRAINABLE, Policy
from barl_jax.train import train_step

from conftest import ACT_DIM, BATCH, OBS_DIM, make_env_stats

MODELS = sorted(ConfigTree().available_models())


class FakeEnv:
  observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
  action_space = gym.spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)


def make(name: str) -> Policy:
  return Policy.make(name, FakeEnv(), env_stats=make_env_stats(), seed=0)


def batch():
  rng = np.random.default_rng(0)
  f = lambda *s: jnp.asarray(rng.normal(size=s), jnp.float32)
  return {"obs": f(BATCH, OBS_DIM), "next_obs": f(BATCH, OBS_DIM),
          "act": jnp.clip(f(BATCH, ACT_DIM), -1, 1),
          "rew": f(BATCH), "term": jnp.zeros((BATCH,), jnp.float32)}


@pytest.mark.parametrize("name", MODELS)
def test_train_step_losses_match_the_optimizer_keys(name):
  policy = make(name)
  optimizer = nnx.Optimizer(policy, policy.build_optimizer(lr_schedule=None), wrt=TRAINABLE)
  graphdef, state = nnx.split((policy, optimizer))

  data = batch()
  for _ in range(4):
    state, losses, _, _ = train_step(graphdef, state, data, ())
    assert set(losses) == set(policy.create_optimizer())
    assert all(np.isfinite(np.asarray(v)) for v in losses.values())


@pytest.mark.parametrize("name", MODELS)
def test_inference_returns_a_singleton_chunk_by_default(name):
  policy = make(name)
  act, _ = policy.inference_fn()(jnp.zeros((BATCH, OBS_DIM), jnp.float32), jax.random.PRNGKey(0))
  assert act.shape == (BATCH, 1, ACT_DIM)


@pytest.mark.parametrize("name", MODELS)
def test_optimizer_labels_cover_every_trainable_leaf(name):
  policy = make(name)
  labels = policy.optimizer_labels()
  for path, _ in nnx.to_flat_state(nnx.state(policy, TRAINABLE)):
    owner = labels
    for p in path:
      owner = owner[p]
    assert owner is not None, f"{'.'.join(map(str, path))} has no optimizer label"

# One captured snapshot per model, from the current (pre-refactor) code. A
# change here is a wandb series rename -- approve the diff before updating it,
# rather than treating a mismatch as a bug to silence.
METRIC_KEYS = {
  "AMDP": ['actor/action/dim_std_mean', 'actor/action/frac_at_limit', 'actor/action/norm_mean', 'actor/entropy/edge_sat', 'actor/entropy/max', 'actor/entropy/mean', 'actor/entropy/min', 'actor/entropy/std', 'actor/grad_norm', 'actor/loss', 'actor/score_clipped_frac', 'actor/score_loss', 'actor/score_target/edge_sat', 'actor/score_target/max', 'actor/score_target/mean', 'actor/score_target/min', 'actor/score_target/std', 'actor/u/dim_std_mean', 'actor/u/norm_mean', 'batch/action/edge_sat', 'batch/action/max', 'batch/action/mean', 'batch/action/min', 'batch/action/std', 'batch/reward/edge_sat', 'batch/reward/max', 'batch/reward/mean', 'batch/reward/min', 'batch/reward/std', 'critic/critic/edge_sat', 'critic/critic/max', 'critic/critic/mean', 'critic/critic/min', 'critic/critic/std', 'critic/ensemble/disagreement', 'critic/ensemble/pessimism_gap', 'critic/grad_norm', 'critic/loss', 'critic/q_target/edge_sat', 'critic/q_target/max', 'critic/q_target/mean', 'critic/q_target/min', 'critic/q_target/std', 'critic/return_std/edge_sat', 'critic/return_std/max', 'critic/return_std/mean', 'critic/return_std/min', 'critic/return_std/std', 'critic/score/dim_std_mean', 'critic/score/norm_mean', 'entropy/alpha', 'entropy/grad_norm', 'entropy/loss', 'entropy/target', 'entropy/value/edge_sat', 'entropy/value/max', 'entropy/value/mean', 'entropy/value/min', 'entropy/value/std', 'entropy/value_scaled/edge_sat', 'entropy/value_scaled/max', 'entropy/value_scaled/mean', 'entropy/value_scaled/min', 'entropy/value_scaled/std'],
  "DIME": ['actor/action/dim_std_mean', 'actor/action/frac_at_limit', 'actor/action/norm_mean', 'actor/entropy/edge_sat', 'actor/entropy/max', 'actor/entropy/mean', 'actor/entropy/min', 'actor/entropy/std', 'actor/grad_norm', 'actor/loss', 'batch/action/edge_sat', 'batch/action/max', 'batch/action/mean', 'batch/action/min', 'batch/action/std', 'batch/reward/edge_sat', 'batch/reward/max', 'batch/reward/mean', 'batch/reward/min', 'batch/reward/std', 'critic/critic/edge_sat', 'critic/critic/max', 'critic/critic/mean', 'critic/critic/min', 'critic/critic/std', 'critic/ensemble/disagreement', 'critic/ensemble/pessimism_gap', 'critic/grad_norm', 'critic/loss', 'critic/q_target/edge_sat', 'critic/q_target/max', 'critic/q_target/mean', 'critic/q_target/min', 'critic/q_target/std', 'critic/return_std/edge_sat', 'critic/return_std/max', 'critic/return_std/mean', 'critic/return_std/min', 'critic/return_std/std', 'entropy/alpha', 'entropy/grad_norm', 'entropy/loss', 'entropy/target', 'entropy/value/edge_sat', 'entropy/value/max', 'entropy/value/mean', 'entropy/value/min', 'entropy/value/std', 'entropy/value_scaled/edge_sat', 'entropy/value_scaled/max', 'entropy/value_scaled/mean', 'entropy/value_scaled/min', 'entropy/value_scaled/std'],
  "NCLQL": ['batch/action/edge_sat', 'batch/action/max', 'batch/action/mean', 'batch/action/min', 'batch/action/std', 'batch/reward/edge_sat', 'batch/reward/max', 'batch/reward/mean', 'batch/reward/min', 'batch/reward/std', 'critic/critic/edge_sat', 'critic/critic/max', 'critic/critic/mean', 'critic/critic/min', 'critic/critic/std', 'critic/ensemble/disagreement', 'critic/ensemble/pessimism_gap', 'critic/grad_norm', 'critic/loss', 'critic/q_target/edge_sat', 'critic/q_target/max', 'critic/q_target/mean', 'critic/q_target/min', 'critic/q_target/std', 'critic/return_std/edge_sat', 'critic/return_std/max', 'critic/return_std/mean', 'critic/return_std/min', 'critic/return_std/std', 'critic/smoothing_loss', 'critic/td_loss', 'nclql/action/dim_std_mean', 'nclql/action/frac_at_limit', 'nclql/action/norm_mean', 'nclql/score_norm'],
  "SAC": ['actor/action/dim_std_mean', 'actor/action/frac_at_limit', 'actor/action/norm_mean', 'actor/entropy/edge_sat', 'actor/entropy/max', 'actor/entropy/mean', 'actor/entropy/min', 'actor/entropy/std', 'actor/grad_norm', 'actor/loss', 'batch/action/edge_sat', 'batch/action/max', 'batch/action/mean', 'batch/action/min', 'batch/action/std', 'batch/reward/edge_sat', 'batch/reward/max', 'batch/reward/mean', 'batch/reward/min', 'batch/reward/std', 'critic/critic/edge_sat', 'critic/critic/max', 'critic/critic/mean', 'critic/critic/min', 'critic/critic/std', 'critic/ensemble/disagreement', 'critic/ensemble/pessimism_gap', 'critic/grad_norm', 'critic/loss', 'critic/q_target/edge_sat', 'critic/q_target/max', 'critic/q_target/mean', 'critic/q_target/min', 'critic/q_target/std', 'critic/return_std/edge_sat', 'critic/return_std/max', 'critic/return_std/mean', 'critic/return_std/min', 'critic/return_std/std', 'entropy/alpha', 'entropy/grad_norm', 'entropy/loss', 'entropy/target', 'entropy/value/edge_sat', 'entropy/value/max', 'entropy/value/mean', 'entropy/value/min', 'entropy/value/std', 'entropy/value_scaled/edge_sat', 'entropy/value_scaled/max', 'entropy/value_scaled/mean', 'entropy/value_scaled/min', 'entropy/value_scaled/std'],
  # SDAC's keys changed under the actor-template refactor (REFACTOR.md): the
  # actor now owns two optimizer keys ("actor", "explore"), so metrics_key's
  # `shared` branch splits actor/loss -> actor/loss/{actor,explore} the same
  # way grad_norm splits; sdac/explore_std and sdac/noise_entropy moved to
  # actor/explore_noise_std and actor/noise_entropy_value (renamed to avoid
  # colliding with the actor's own explore_std property / noise_entropy()
  # method via Recorder's setattr); and actor/entropy, actor/action are now
  # recorded by the shared RLPolicy.train_step like every other algorithm's.
  "SDAC": ['actor/action/dim_std_mean', 'actor/action/frac_at_limit', 'actor/action/norm_mean', 'actor/cand_saturation', 'actor/entropy/edge_sat', 'actor/entropy/max', 'actor/entropy/mean', 'actor/entropy/min', 'actor/entropy/std', 'actor/eps_pred/dim_std_mean', 'actor/eps_pred/norm_mean', 'actor/eps_target/dim_std_mean', 'actor/eps_target/norm_mean', 'actor/explore_noise_std', 'actor/grad_norm/actor', 'actor/grad_norm/explore', 'actor/loss/actor', 'actor/loss/explore', 'actor/noise_entropy_value', 'actor/q_scale', 'actor/signal_ratio', 'actor/tilt_ratio', 'actor/weight_ess/edge_sat', 'actor/weight_ess/max', 'actor/weight_ess/mean', 'actor/weight_ess/min', 'actor/weight_ess/std', 'batch/action/edge_sat', 'batch/action/max', 'batch/action/mean', 'batch/action/min', 'batch/action/std', 'batch/reward/edge_sat', 'batch/reward/max', 'batch/reward/mean', 'batch/reward/min', 'batch/reward/std', 'critic/critic/edge_sat', 'critic/critic/max', 'critic/critic/mean', 'critic/critic/min', 'critic/critic/std', 'critic/ensemble/disagreement', 'critic/ensemble/pessimism_gap', 'critic/grad_norm', 'critic/loss', 'critic/q_target/edge_sat', 'critic/q_target/max', 'critic/q_target/mean', 'critic/q_target/min', 'critic/q_target/std', 'critic/return_std/edge_sat', 'critic/return_std/max', 'critic/return_std/mean', 'critic/return_std/min', 'critic/return_std/std', 'entropy/alpha', 'entropy/grad_norm', 'entropy/loss', 'entropy/target', 'entropy/value/edge_sat', 'entropy/value/max', 'entropy/value/mean', 'entropy/value/min', 'entropy/value/std', 'entropy/value_scaled/edge_sat', 'entropy/value_scaled/max', 'entropy/value_scaled/mean', 'entropy/value_scaled/min', 'entropy/value_scaled/std'],
}


@pytest.mark.parametrize("name", MODELS)
def test_metric_keys_are_pinned(name):
  policy = make(name)
  optimizer = nnx.Optimizer(policy, policy.build_optimizer(lr_schedule=None), wrt=TRAINABLE)
  graphdef, state = nnx.split((policy, optimizer))

  _, _, (primary, _), _ = train_step(graphdef, state, batch(), (PRIMARY,), jnp.asarray(0, jnp.int32))
  assert sorted(primary) == METRIC_KEYS[name]


def test_rl_policy_gate_excludes_noise_conditioned_critics():
  """`RLPolicy`'s critic slot gates on `QCriticBase`, which a noise-conditioned
  critic satisfies -- but `NoiseConditionedCritic.loss` is a wider signature
  than the template's one-backward call can drive. The refusal lives on the
  `noise_conditioned` marker (`rl.py` asserts it off), so pin that marker: a
  normal critic must keep it off and every NCLQL critic on, or the pairing
  silently passes the gate."""
  from barl_jax.networks.registry import get_component

  assert get_component("C51Critic").noise_conditioned is False
  for name in ("NCLQLCritic", "NCLQLC51Critic", "NCLQLHLGaussCritic"):
    assert get_component(name).noise_conditioned is True


def test_resolved_style_surface_is_the_surface_that_gets_built():
  policy = make("SDAC")
  resolved = ConfigTree().resolve_slot("SDAC", "actor", None)[1]
  assert resolved["activation"].lower() == "mish"
  assert policy.actor.activation.lower() == resolved["activation"].lower()
  assert policy.actor.field.net.act is jax.nn.mish


def test_unknown_component_override_is_rejected_at_construction():
  import barl_jax.model_config as model_config

  previous = model_config.get_overrides()
  try:
    model_config.set_overrides({"components.actor.activaton": "relu"})
    with pytest.raises(KeyError, match="unused config keys.*activaton"):
      make("SAC")
  finally:
    model_config.set_overrides(previous)


def test_duplicate_optimizer_keys_are_rejected(monkeypatch):
  import optax
  from barl_jax.networks.actors.stochastic import StochasticActor
  from barl_jax.networks.critic import C51Critic

  duplicate = lambda self: {"shared": optax.identity()}
  monkeypatch.setattr(StochasticActor, "create_optimizer", duplicate)
  monkeypatch.setattr(C51Critic, "create_optimizer", duplicate)
  with pytest.raises(KeyError, match="optimizer key 'shared'.*more than one"):
    make("SAC")
