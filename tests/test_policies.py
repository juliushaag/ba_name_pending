"""The `Policy` contract, over every model in `config/models/`.

Each of the five is a different algorithm, but every one has to satisfy the
same bargain with the training loop: `train_step` returns finite losses keyed
exactly by `create_optimizer()`, and every trainable leaf has an owner in
`optimizer_labels()`. Nothing here asserts a specific numerical behaviour --
that is what the per-algorithm tests own -- only that the contract holds.
"""

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import ACT_DIM, BATCH, OBS_DIM, make_env_stats
from flax import nnx

from barl_jax.components import TRAINABLE
from barl_jax.config.run import RunConfig
from barl_jax.metrics import PRIMARY
from barl_jax.optimization import train_step
from barl_jax.policies.rl import MaxEntRLPolicy
from barl_jax.policy import Policy

MODELS = sorted(Policy.available())
MAXENT_MODELS = [n for n in MODELS
                 if issubclass(Policy.class_for(RunConfig.defaults(model=n, env="HalfCheetah").policy), MaxEntRLPolicy)
                 and n != "SDAC"]  # SDAC's alpha is scalar-only (config_problems)


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
          "chunk_return": f(BATCH), "term": jnp.zeros((BATCH,), jnp.float32),
          "horizon": jnp.ones((BATCH,), jnp.int32)}


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
def test_actor_loss_gradient_does_not_reach_the_critic(name):
  """`frozen(critic)` inside every actor's `loss()` has to actually stop the
  gradient, not just read the current price -- if it leaked, the actor's own
  objective would also be training the critic's heads on whatever the actor
  happens to sample, silently changing what the critic regresses to. NCLQL's
  LangevinActor has no `loss()` -- it climbs the live critic's action
  gradients instead of an optimized objective, so there is no seam this
  applies to."""
  policy = make(name)
  if not hasattr(policy, "actor") or not hasattr(policy.actor, "loss"):
    pytest.skip(f"{name} has no actor loss")
  data = batch()

  def actor_loss(critic):
    losses = policy.actor.loss(data["obs"], critic, policy.alpha(), key=jax.random.PRNGKey(0))
    return sum(losses.values())

  grads = nnx.grad(actor_loss, argnums=nnx.DiffState(0, TRAINABLE))(policy.critic)
  leaves = jax.tree.leaves(nnx.state(grads))
  assert leaves and all(bool(jnp.all(g == 0)) for g in leaves), \
    "actor.loss's gradient reached a critic param it should have been frozen from"


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


@pytest.mark.parametrize("name", MAXENT_MODELS)
@pytest.mark.parametrize("chunk_size", [1, 4])
def test_per_action_alpha_gives_one_alpha_per_chunk_position(name, chunk_size):
  """`alpha_granularity: per_action` trains one Lagrange multiplier per chunk
  position instead of one for the whole plan. Every MaxEnt actor's entropy is
  already decomposed at that resolution (`Component.chunk_sum`), so this has
  to hold for every one of them, at chunk_size 1 (where it must degenerate to
  the same shape `per_chunk` uses) and above."""
  run = RunConfig.from_dict({
    "model": name, "env": "HalfCheetah",
    "model_overrides": {"params": {"alpha_granularity": "per_action", "chunk_size": chunk_size}},
  })
  policy = Policy.make(name, FakeEnv(), run.model_config, env_stats=make_env_stats(), seed=0)
  assert policy.chunk_size == chunk_size
  assert policy.alpha().shape == (chunk_size,)

  optimizer = nnx.Optimizer(policy, policy.build_optimizer(lr_schedule=None), wrt=TRAINABLE)
  graphdef, state = nnx.split((policy, optimizer))

  data = {**batch(), "act": jnp.clip(jax.random.normal(jax.random.PRNGKey(2), (BATCH, chunk_size * ACT_DIM)), -1, 1),
          "horizon": jnp.full((BATCH,), chunk_size, jnp.int32)}
  for _ in range(4):
    state, losses, _, _ = train_step(graphdef, state, data, ())
    assert set(losses) == set(policy.create_optimizer())
    assert all(np.isfinite(np.asarray(v)) for v in losses.values())


# One captured snapshot per model, from the current (pre-refactor) code. A
# change here is a wandb series rename -- approve the diff before updating it,
# rather than treating a mismatch as a bug to silence.
METRIC_KEYS = {
  "AMDP": ['actor/action/dim_std_mean', 'actor/action/frac_at_limit', 'actor/action/norm_mean', 'actor/entropy/edge_sat', 'actor/entropy/max', 'actor/entropy/mean', 'actor/entropy/min', 'actor/entropy/std', 'actor/grad_norm', 'actor/loss', 'actor/score_clipped_frac', 'actor/score_loss', 'actor/score_target/edge_sat', 'actor/score_target/max', 'actor/score_target/mean', 'actor/score_target/min', 'actor/score_target/std', 'actor/u/dim_std_mean', 'actor/u/norm_mean', 'batch/action/edge_sat', 'batch/action/max', 'batch/action/mean', 'batch/action/min', 'batch/action/std', 'batch/reward/edge_sat', 'batch/reward/max', 'batch/reward/mean', 'batch/reward/min', 'batch/reward/std', 'critic/critic/edge_sat', 'critic/critic/max', 'critic/critic/mean', 'critic/critic/min', 'critic/critic/std', 'critic/ensemble/disagreement', 'critic/ensemble/pessimism_gap', 'critic/grad_norm', 'critic/loss', 'critic/q_target/edge_sat', 'critic/q_target/max', 'critic/q_target/mean', 'critic/q_target/min', 'critic/q_target/std', 'critic/return_std/edge_sat', 'critic/return_std/max', 'critic/return_std/mean', 'critic/return_std/min', 'critic/return_std/std', 'critic/score/dim_std_mean', 'critic/score/norm_mean', 'entropy/alpha', 'entropy/grad_norm', 'entropy/loss', 'entropy/target', 'entropy/value/edge_sat', 'entropy/value/max', 'entropy/value/mean', 'entropy/value/min', 'entropy/value/std', 'entropy/value_scaled/edge_sat', 'entropy/value_scaled/max', 'entropy/value_scaled/mean', 'entropy/value_scaled/min', 'entropy/value_scaled/std'],
  "DIME": ['actor/action/dim_std_mean', 'actor/action/frac_at_limit', 'actor/action/norm_mean', 'actor/entropy/edge_sat', 'actor/entropy/max', 'actor/entropy/mean', 'actor/entropy/min', 'actor/entropy/std', 'actor/grad_norm', 'actor/loss', 'batch/action/edge_sat', 'batch/action/max', 'batch/action/mean', 'batch/action/min', 'batch/action/std', 'batch/reward/edge_sat', 'batch/reward/max', 'batch/reward/mean', 'batch/reward/min', 'batch/reward/std', 'critic/critic/edge_sat', 'critic/critic/max', 'critic/critic/mean', 'critic/critic/min', 'critic/critic/std', 'critic/ensemble/disagreement', 'critic/ensemble/pessimism_gap', 'critic/grad_norm', 'critic/loss', 'critic/q_target/edge_sat', 'critic/q_target/max', 'critic/q_target/mean', 'critic/q_target/min', 'critic/q_target/std', 'critic/return_std/edge_sat', 'critic/return_std/max', 'critic/return_std/mean', 'critic/return_std/min', 'critic/return_std/std', 'entropy/alpha', 'entropy/grad_norm', 'entropy/loss', 'entropy/target', 'entropy/value/edge_sat', 'entropy/value/max', 'entropy/value/mean', 'entropy/value/min', 'entropy/value/std', 'entropy/value_scaled/edge_sat', 'entropy/value_scaled/max', 'entropy/value_scaled/mean', 'entropy/value_scaled/min', 'entropy/value_scaled/std'],
  "NCLQL": ['actor/action/dim_std_mean', 'actor/action/frac_at_limit', 'actor/action/norm_mean', 'actor/score_norm', 'batch/action/edge_sat', 'batch/action/max', 'batch/action/mean', 'batch/action/min', 'batch/action/std', 'batch/reward/edge_sat', 'batch/reward/max', 'batch/reward/mean', 'batch/reward/min', 'batch/reward/std', 'critic/clean_td_loss', 'critic/critic/edge_sat', 'critic/critic/max', 'critic/critic/mean', 'critic/critic/min', 'critic/critic/std', 'critic/ensemble/disagreement', 'critic/ensemble/pessimism_gap', 'critic/grad_norm', 'critic/loss', 'critic/q_target/edge_sat', 'critic/q_target/max', 'critic/q_target/mean', 'critic/q_target/min', 'critic/q_target/std', 'critic/return_std/edge_sat', 'critic/return_std/max', 'critic/return_std/mean', 'critic/return_std/min', 'critic/return_std/std', 'critic/smoothing_loss'],
  "SAC": ['actor/action/dim_std_mean', 'actor/action/frac_at_limit', 'actor/action/norm_mean', 'actor/entropy/edge_sat', 'actor/entropy/max', 'actor/entropy/mean', 'actor/entropy/min', 'actor/entropy/std', 'actor/grad_norm', 'actor/loss', 'batch/action/edge_sat', 'batch/action/max', 'batch/action/mean', 'batch/action/min', 'batch/action/std', 'batch/reward/edge_sat', 'batch/reward/max', 'batch/reward/mean', 'batch/reward/min', 'batch/reward/std', 'critic/critic/edge_sat', 'critic/critic/max', 'critic/critic/mean', 'critic/critic/min', 'critic/critic/std', 'critic/ensemble/disagreement', 'critic/ensemble/pessimism_gap', 'critic/grad_norm', 'critic/loss', 'critic/q_target/edge_sat', 'critic/q_target/max', 'critic/q_target/mean', 'critic/q_target/min', 'critic/q_target/std', 'critic/return_std/edge_sat', 'critic/return_std/max', 'critic/return_std/mean', 'critic/return_std/min', 'critic/return_std/std', 'entropy/alpha', 'entropy/grad_norm', 'entropy/loss', 'entropy/target', 'entropy/value/edge_sat', 'entropy/value/max', 'entropy/value/mean', 'entropy/value/min', 'entropy/value/std', 'entropy/value_scaled/edge_sat', 'entropy/value_scaled/max', 'entropy/value_scaled/mean', 'entropy/value_scaled/min', 'entropy/value_scaled/std'],
  "SDAC": ['actor/action/dim_std_mean', 'actor/action/frac_at_limit', 'actor/action/norm_mean', 'actor/eps_pred/dim_std_mean', 'actor/eps_pred/norm_mean', 'actor/grad_norm', 'actor/loss', 'actor/weight_ess/edge_sat', 'actor/weight_ess/max', 'actor/weight_ess/mean', 'actor/weight_ess/min', 'actor/weight_ess/std', 'actor/weight_max/edge_sat', 'actor/weight_max/max', 'actor/weight_max/mean', 'actor/weight_max/min', 'actor/weight_max/std', 'batch/action/edge_sat', 'batch/action/max', 'batch/action/mean', 'batch/action/min', 'batch/action/std', 'batch/reward/edge_sat', 'batch/reward/max', 'batch/reward/mean', 'batch/reward/min', 'batch/reward/std', 'critic/critic/edge_sat', 'critic/critic/max', 'critic/critic/mean', 'critic/critic/min', 'critic/critic/std', 'critic/ensemble/disagreement', 'critic/ensemble/pessimism_gap', 'critic/grad_norm', 'critic/loss', 'critic/q_target/edge_sat', 'critic/q_target/max', 'critic/q_target/mean', 'critic/q_target/min', 'critic/q_target/std', 'critic/return_std/edge_sat', 'critic/return_std/max', 'critic/return_std/mean', 'critic/return_std/min', 'critic/return_std/std', 'entropy/alpha', 'entropy/grad_norm', 'entropy/loss', 'entropy/target', 'entropy/value/edge_sat', 'entropy/value/max', 'entropy/value/mean', 'entropy/value/min', 'entropy/value/std', 'entropy/value_scaled/edge_sat', 'entropy/value_scaled/max', 'entropy/value_scaled/mean', 'entropy/value_scaled/min', 'entropy/value_scaled/std'],
  "TD3": ['actor/action/dim_std_mean', 'actor/action/frac_at_limit', 'actor/action/norm_mean', 'actor/grad_norm', 'actor/loss', 'batch/action/edge_sat', 'batch/action/max', 'batch/action/mean', 'batch/action/min', 'batch/action/std', 'batch/reward/edge_sat', 'batch/reward/max', 'batch/reward/mean', 'batch/reward/min', 'batch/reward/std', 'critic/critic/edge_sat', 'critic/critic/max', 'critic/critic/mean', 'critic/critic/min', 'critic/critic/std', 'critic/ensemble/disagreement', 'critic/ensemble/pessimism_gap', 'critic/grad_norm', 'critic/loss', 'critic/q_target/edge_sat', 'critic/q_target/max', 'critic/q_target/mean', 'critic/q_target/min', 'critic/q_target/std'],
}


@pytest.mark.parametrize("name", MODELS)
def test_metric_keys_are_pinned(name):
  policy = make(name)
  optimizer = nnx.Optimizer(policy, policy.build_optimizer(lr_schedule=None), wrt=TRAINABLE)
  graphdef, state = nnx.split((policy, optimizer))

  _, _, (primary, _), _ = train_step(graphdef, state, batch(), (PRIMARY,), jnp.asarray(0, jnp.int32))
  assert sorted(primary) == METRIC_KEYS[name]


def test_critics_declare_supported_objectives():
  from barl_jax.critics.presets import nclql
  from barl_jax.registration import get_component
  for name in ("DirectCritic", "EnsembleCritic", "HLGaussCritic", "C51Critic"):
    cls = get_component(name)
    assert cls.learning_objective_for(cls.config_type()) == "td"
    assert cls.learning_objective_for(nclql(cls.config_type())) == "noise_td"


def test_resolved_style_surface_is_the_surface_that_gets_built():
  policy = make("SDAC")
  activation = policy.config.components.actor.mlp.activation
  assert policy.actor.field.net.act is getattr(jax.nn, activation.lower())


def test_unknown_component_override_is_rejected_at_construction():
  with pytest.raises((KeyError, TypeError), match="activaton"):
    RunConfig.from_dict({
      "model": "SAC", "env": "HalfCheetah",
      "model_overrides": {"components": {"actor": {"activaton": "relu"}}},
    })


def test_duplicate_optimizer_keys_are_rejected(monkeypatch):
  import optax

  from barl_jax.actors.sac import StochasticActor
  from barl_jax.critics import C51Critic

  duplicate = lambda self: {"shared": optax.identity()}
  monkeypatch.setattr(StochasticActor, "create_optimizer", duplicate)
  monkeypatch.setattr(C51Critic, "create_optimizer", duplicate)
  with pytest.raises(KeyError, match="optimizer key 'shared'.*more than one"):
    make("SAC")
