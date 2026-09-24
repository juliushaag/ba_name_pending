"""Focused contracts for the flow matching (FloQ) critic.

The generic critic suite checks the shared Bellman interface.  These checks
exercise the extra path in which a scalar Q value is produced by repeatedly
integrating a velocity field, and then used by both SAC and AMDP.
"""

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import BATCH, OBS_DIM, build, make_env_stats
from flax import nnx

from barl_jax.components import TRAINABLE
from barl_jax.config.run import RunConfig
from barl_jax.metrics import DEBUG, PRIMARY
from barl_jax.optimization import train_step
from barl_jax.policy import Policy


class _BoxEnv:
  observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
  action_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)


def _critic(*, seed=0, **values):
  return build("FloqCritic", seed=seed, env_stats=make_env_stats(), **values)


def _batch():
  key = jax.random.PRNGKey(90)
  obs_key, next_key, act_key, ret_key = jax.random.split(key, 4)
  return {
    "obs": jax.random.normal(obs_key, (BATCH, OBS_DIM)),
    "next_obs": jax.random.normal(next_key, (BATCH, OBS_DIM)),
    "act": jnp.tanh(jax.random.normal(act_key, (BATCH, 3))),
    "chunk_return": jax.random.normal(ret_key, (BATCH,)),
    "term": jnp.zeros((BATCH,)),
    "horizon": jnp.ones((BATCH,), dtype=jnp.int32),
  }


def test_floq_value_is_finite_and_has_a_jittable_iterative_path():
  critic = _critic()
  data = _batch()
  eager = critic.value(data["obs"], data["act"])
  assert eager.shape == (BATCH, 1)
  assert bool(jnp.all(jnp.isfinite(eager)))

  graphdef, state = nnx.split(critic)

  @jax.jit
  def run(graphdef, state, obs, action):
    return nnx.merge(graphdef, state).value(obs, action)

  compiled = run(graphdef, state, data["obs"], data["act"])
  np.testing.assert_allclose(compiled, eager, rtol=1e-5, atol=1e-5)


def test_floq_flow_matching_loss_updates_velocity_parameters_and_is_key_reproducible():
  critic = _critic(flow_steps=2, noise_samples=3)
  data = _batch()

  def loss_fn(c):
    losses = c.td_loss(
      data["obs"], data["act"], data["next_obs"], data["act"],
      data["chunk_return"], data["term"], data["horizon"],
      key=jax.random.PRNGKey(7))
    assert set(losses) == {"critic"}
    return losses["critic"]

  first = loss_fn(critic)
  second = loss_fn(critic)
  np.testing.assert_allclose(first, second, rtol=1e-6, atol=1e-6)
  assert bool(jnp.isfinite(first))

  grads = nnx.grad(loss_fn, argnums=nnx.DiffState(0, TRAINABLE))(critic)
  leaves = jax.tree.leaves(nnx.state(grads))
  assert leaves and all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)
  assert any(bool(jnp.any(g != 0)) for g in leaves), "flow matching loss did not train the field"


@pytest.mark.parametrize("model", ["SAC", "AMDP"])
def test_sac_and_amdp_accept_floq_critic_and_take_a_training_step(model):
  run = RunConfig.from_dict({
    "model": model, "env": "HalfCheetah",
    "model_overrides": {
      "params": {"chunk_size": 2},
      "components": {
        "actor": {"latent_dim": 8},
        "critic": {"class": "FloqCritic", "latent_dim": 8,
                   "flow_steps": 2, "noise_samples": 2},
      },
    },
  })
  policy = Policy.make(model, _BoxEnv(), run.model_config,
                       env_stats=make_env_stats(), seed=0)
  assert type(policy.critic).__name__ == "FloqCritic"

  derived = policy.derived_config()
  assert derived["critic.v_min"] == pytest.approx(-10.0)
  assert derived["critic.v_max"] == pytest.approx(10.0)
  optimizer = nnx.Optimizer(policy, policy.build_optimizer(lr_schedule=None), wrt=TRAINABLE)
  graphdef, state = nnx.split((policy, optimizer))
  data = _batch()
  data["act"] = jnp.tile(data["act"], (1, 2))
  data["horizon"] = jnp.full_like(data["horizon"], 2)
  state, losses, _, _ = train_step(graphdef, state, data, (PRIMARY, DEBUG), jnp.asarray(0))
  assert losses and all(bool(jnp.all(jnp.isfinite(v))) for v in losses.values())



def test_floq_config_rejects_nonpositive_flow_steps():
  with pytest.raises((ValueError, TypeError), match="(positive|>=|flow|step)"):
    _critic(flow_steps=0)


def test_floq_rng_state_roundtrips_for_checkpoint_continuation():
  critic = _critic(flow_steps=1, noise_samples=2)
  data = _batch()
  args = (data["obs"], data["act"], data["next_obs"], data["act"],
          data["chunk_return"], data["term"], data["horizon"])
  key = jax.random.PRNGKey(123)
  critic.td_loss(*args, key=key)["critic"]
  graphdef, state = nnx.split(critic)
  restored = nnx.merge(graphdef, state)
  current = critic.td_loss(*args, key=key)["critic"]
  resumed = restored.td_loss(*args, key=key)["critic"]
  np.testing.assert_array_equal(current, resumed)

class _ConstantField(nnx.Module):
  def __init__(self, value, n_models):
    self.value = float(value)
    self.n_models = n_models

  def __call__(self, obs, action, z, time):
    del obs, time
    return (jnp.full((self.n_models, *z.shape), self.value, dtype=z.dtype) if z.ndim == 2 else jnp.full_like(z, self.value))


def test_floq_euler_chain_matches_constant_and_affine_python_references():
  obs = jnp.zeros((2, OBS_DIM), jnp.float32)
  action = jnp.zeros((2, 3), jnp.float32)
  starts = jnp.array([[[-0.3], [0.2]], [[0.4], [0.9]]], dtype=jnp.float32)

  for steps in (1, 2, 4):
    critic = _critic(flow_steps=steps)
    out = critic._integrate(_ConstantField(0.37, critic.n_models), obs, action, starts)
    np.testing.assert_allclose(out, jnp.broadcast_to(starts[None, ...] + 0.37, out.shape), rtol=1e-6, atol=1e-6)

    def affine(_obs, _action, z, _time):
      value = 0.2 * z + 0.1
      return jnp.broadcast_to(value, (critic.n_models, *value.shape)) if z.ndim == 2 else value

    expected = np.array(starts)
    for _ in range(steps):
      expected += (0.2 * expected + 0.1) / steps
    np.testing.assert_allclose(
      critic._integrate(affine, obs, action, starts), jnp.broadcast_to(expected[None, ...], (critic.n_models, *expected.shape)),
      rtol=1e-6, atol=1e-6)



def test_floq_value_has_a_finite_nonzero_action_gradient():
  critic = _critic(flow_steps=2)
  data = _batch()
  action = data["act"]
  grad = nnx.grad(lambda c, a: c.value(data["obs"], a).sum(), argnums=1)(critic, action)
  assert grad.shape == action.shape
  assert bool(jnp.all(jnp.isfinite(grad)))
  assert bool(jnp.any(jnp.abs(grad) > 1e-8))


def test_floq_target_copy_tracks_live_flow_but_keeps_live_parameters_intact():
  critic = _critic()
  assert critic.has_target
  before_target = jax.tree.map(jnp.copy, nnx.state(critic.net_target, nnx.Param))
  live_before = jax.tree.map(jnp.copy, nnx.state(critic.net, nnx.Param))
  nnx.update(critic.net,
             jax.tree.map(lambda x: x + 0.5, nnx.state(critic.net, nnx.Param)))
  live_after = nnx.state(critic.net, nnx.Param)
  critic.after_step()
  target_after = nnx.state(critic.net_target, nnx.Param)
  assert any(not np.array_equal(a, b)
             for a, b in zip(jax.tree.leaves(before_target), jax.tree.leaves(target_after)))
  assert all(np.all(np.isfinite(g)) for g in jax.tree.leaves(target_after))
  for got, expected in zip(jax.tree.leaves(live_after), jax.tree.leaves(live_before)):
    assert np.any(np.asarray(got) != np.asarray(expected))

class _AffineEnsemble(nnx.Module):
  def __init__(self, gain, bias, n_models):
    self.gain = nnx.Param(jnp.asarray(gain, jnp.float32))
    self.bias = nnx.Param(jnp.asarray(bias, jnp.float32))
    self.n_models = n_models

  def __call__(self, obs, action, z, time):
    del obs, time
    value = self.gain[...] * z + self.bias[...] + 0.05 * action[..., :1]
    return jnp.broadcast_to(value, (self.n_models, *value.shape)) if z.ndim == 2 else value


def test_floq_td_loss_matches_sampled_flow_matching_equation():
  critic = _critic(flow_steps=1, noise_samples=2)
  live = _AffineEnsemble(0.2, 0.1, critic.n_models)
  target_field = _AffineEnsemble(-0.1, 0.04, critic.n_models)
  critic.net = live
  critic.net_target = target_field
  data = _batch()
  entropy_bonus = jnp.full((BATCH, 1), 0.3)
  key = jax.random.PRNGKey(31)
  got = critic.td_loss(
    data["obs"], data["act"], data["next_obs"], data["act"],
    data["chunk_return"], data["term"], data["horizon"],
    entropy_bonus=entropy_bonus, key=key)["critic"]

  target_key, start_key, time_key = jax.random.split(key, 3)
  noise_samples = critic.config.noise_samples
  target_starts = jax.random.uniform(
    target_key, (BATCH, noise_samples, 1),
    minval=critic.noise_min, maxval=critic.noise_max)
  target_endpoint = target_starts + (-0.1 * target_starts + 0.04 + 0.05 * data["act"][:, None, :1])
  target_flow = target_endpoint.mean(axis=1)[None, ...]
  bootstrap = target_flow.mean(axis=0)
  y = data["chunk_return"].reshape(-1, 1) + critic.config.gamma * (bootstrap + entropy_bonus)

  starts = jax.random.uniform(
    start_key, (BATCH, noise_samples, 1),
    minval=critic.noise_min, maxval=critic.noise_max)
  z0 = starts.reshape(-1, 1)
  t = jax.random.uniform(time_key, z0.shape)
  y_rep = jnp.repeat(y, noise_samples, axis=0)
  zt = (1.0 - t) * z0 + t * y_rep
  pred = 0.2 * zt + 0.1 + 0.05 * jnp.repeat(data["act"], noise_samples, axis=0)[:, :1]
  expected = ((pred - (y_rep - z0)) ** 2).mean()
  next_action_grad = nnx.grad(lambda c, next_action: c.td_loss(
    data["obs"], data["act"], data["next_obs"], next_action,
    data["chunk_return"], data["term"], data["horizon"],
    entropy_bonus=entropy_bonus, key=key)["critic"], argnums=1)(critic, data["act"])
  assert bool(jnp.all(next_action_grad == 0))
  grads = nnx.grad(lambda c: c.td_loss(
    data["obs"], data["act"], data["next_obs"], data["act"],
    data["chunk_return"], data["term"], data["horizon"],
    entropy_bonus=entropy_bonus, key=key)["critic"],
    argnums=nnx.DiffState(0, nnx.Param))(critic)
  grad_state = nnx.state(grads, nnx.Param)
  assert all(bool(jnp.all(g == 0)) for g in jax.tree.leaves(grad_state["net_target"]))
  assert any(bool(jnp.any(g != 0)) for g in jax.tree.leaves(grad_state["net"]))
  np.testing.assert_allclose(got, expected, rtol=2e-5, atol=2e-6)
