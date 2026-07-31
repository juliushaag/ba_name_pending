"""Env-derived facts every component is handed at construction time, as a
plain dict so adding a key costs nothing. Built once per run by
Environment.get_env_stats(), stored on the Policy and injected into every slot
by Policy.build_component(). Deliberately *not* part of the models.yaml /
components.yaml layering -- this is a function of the environment, not a
tunable; a component that wants it tunable takes a normal kwarg that falls
back to its EnvStats entry (see C51Critic's v_min/v_max).

The `live` sub-dict is the one mutable part: scripts/online_learning.py
refreshes it from ReplayBuffer.rollout()'s stats after every rollout, and since
every component holds a reference to the same object, they all see it."""

import gymnasium as gym
import numpy as np


class EnvStats(dict):

  @classmethod
  def from_env(cls, env: gym.Env, **declared) -> "EnvStats":
    obs_space, act_space = env.observation_space, env.action_space
    discrete = isinstance(act_space, gym.spaces.Discrete)

    stats = cls(
      obs_shape    = tuple(obs_space.shape),
      obs_dim      = int(np.prod(obs_space.shape)),
      obs_low      = np.asarray(obs_space.low,  dtype=np.float32),
      obs_high     = np.asarray(obs_space.high, dtype=np.float32),
      act_discrete = discrete,
      act_dim      = int(act_space.n) if discrete else int(np.prod(act_space.shape)),
      act_low      = None if discrete else np.asarray(act_space.low,  dtype=np.float32),
      act_high     = None if discrete else np.asarray(act_space.high, dtype=np.float32),
      max_episode_steps = None,
      return_min   = None,
      return_max   = None,
      live         = dict(),
    )
    stats.update(declared)
    return stats

  def derive(self, **overrides) -> "EnvStats":
    """Shallow copy with some keys replaced, for a component that feeds its
    inner network something other than the raw env dims (see NC-LQL's
    level-embedding-widened action vector). `live` is shared by
    reference, so the copy still tracks each rollout."""
    return EnvStats({**self, **overrides})

  def static(self) -> dict:
    """Everything but the per-rollout `live` section -- what's worth logging
    once as run config."""
    return {k: v for k, v in self.items() if k != "live"}
