"""The shared off-policy actor-critic template.

SAC, AMDP and DIME's `train_step`s were, at bottom, one function: a
stop-gradiented bootstrap draw at `next_obs`, a critic Bellman target with an
optional entropy bonus, an actor loss gated at some cadence, and an entropy
dual fed the bootstrap's entropy. What differed was four things -- obs
normalisation, the gate's cadence, the name of the actor's generator, and
whether the dual is fed anything at all -- and none of those is an algorithm.

`RLPolicy` states that shared step once; `MaxEntRLPolicy` adds the dual. An
algorithm becomes a `config/models/*.yaml` entry naming a `policy:`, an
`actor:` class and a `critic:` class -- not a Python subclass -- which is what
lets SAC, AMDP, DIME and SDAC (once its actor owns best-of-N and exploration,
see `actors`) share these two classes entirely."""

import gymnasium as gym
import jax
import jax.numpy as jnp

from barl_jax.hparam import Config, HParam
from barl_jax.networks.critic import QCriticBase
from barl_jax.networks.regulator import AutotuneRegulator
from barl_jax.policy import Actor, Policy, gated_losses, register_policy


@register_policy
class RLPolicy(Policy):
  """Actor + critic, no entropy dual -- `alpha`/`target_shift`/`dual_loss` are
  the three seams `MaxEntRLPolicy` fills in. On its own this is TD3-shaped:
  nothing in this repo's config points a model at `RLPolicy` directly, but it
  is the layout every actor-critic algorithm here shares, and an actor that
  adds no entropy term to its own objective needs no dual at all."""

  SLOTS = ("critic", "actor")
  policy_delay = HParam(1, descr="grad steps between actor/dual updates (critic steps every step)")
  normalize_obs = Config(True)

  def __init__(self, env: gym.Env, **kwargs):
    super().__init__(env, **kwargs)

    assert isinstance(env.action_space, gym.spaces.Box)

    chunked = self.env_stats.derive(act_dim=self.env_stats["chunked_act_dim"])
    self.critic: QCriticBase = self.build_component("critic", QCriticBase,
                                                    env_stats=chunked)
    # The build gate is QCriticBase, which a noise-conditioned critic satisfies
    # -- but its `loss` is a wider signature this template's one-backward
    # `critic.loss(..., shift=...)` call cannot drive. Refuse the pairing here
    # rather than letting it pass the gate and fail at the first train_step.
    assert not getattr(self.critic, "noise_conditioned", False), (
      f"{self.model_name}: {type(self.critic).__name__} is noise-conditioned and trains "
      f"through its own wider loss -- it belongs under the NCLQL policy, not this template.")
    self.actor: Actor        = self.build_component("actor", Actor)
    # No cadence/optimizer-key guard here: `self.optimizer_cadence()` is a
    # virtual call, and a MaxEntRLPolicy subclass hasn't built `self.entropy`
    # yet at this point in `super().__init__()` -- the check belongs at the
    # end of whichever `__init__` runs last (MaxEntRLPolicy's, today).

  def _obs(self, x):
    return self.normalize_observations(x) if self.normalize_obs else x

  def optimizer_cadence(self) -> dict:
    return {"actor": self.policy_delay}

  # ---- the three seams MaxEntRLPolicy fills in -----------------------------

  def alpha(self):
    return jnp.asarray(1.0)

  def target_shift(self, alpha, entropy):
    return 0.0

  def dual_loss(self, entropy, losses) -> None:
    pass

  # ---------------------------------------------------------------------

  def train_step(self, batch):
    obs      = self._obs(batch["obs"])
    next_obs = self._obs(batch["next_obs"])
    action   = batch["act"]
    reward   = batch["rew"]
    term     = batch["term"]
    horizon  = batch.get("horizon")

    losses = {}
    alpha  = self.alpha()

    # Every draw this step descends from one key, split here: the bootstrap
    # sample and the actor losses must not share a stream, and a step's
    # randomness should be readable off train_step rather than hidden in each
    # component's `rngs`. keys[2] is retired -- the explore term shares the
    # actor's stream now (one best-of-N) -- but the split stays 3-way so the
    # other two streams are unchanged.
    keys = jax.random.split(self.rngs.noise(), 3)

    next_action, entropy = jax.lax.stop_gradient(
      self.actor.generate_action(next_obs, critic=self.critic, key=keys[0]))
    next_action = next_action.reshape(next_action.shape[0], -1)

    self.critic.loss(obs, action, next_obs, next_action, reward, term,
                     losses, shift=self.target_shift(alpha, entropy), horizon=horizon)

    # Skipped outright on the steps `policy_delay` would mask the update on
    # anyway -- the losses price through a frozen critic and their targets are
    # stop_gradient'd, so nothing here reaches the critic's parameters and the
    # skip is parameter-identical (see `gated_losses`).
    losses.update(gated_losses(
      self, self.policy_delay,
      lambda actor, critic: actor.loss(obs, critic, alpha, key=keys[1]),
      tuple(sorted(self.actor.create_optimizer())), self.actor, self.critic))
    self.dual_loss(entropy, losses)

    self.actor.metrics.vec("action", next_action.reshape(
      next_action.shape[0], self.actor.chunk_size, self.actor.action_dim), limit=1.0)
    self.actor.metrics.dist("entropy", entropy)

    return losses

  def inference_step(self, observation, step_info=None, key=None):
    obs = self._obs(observation)
    return self.action_chunk(self.actor.sample(obs, critic=self.critic, key=key)), {}


@register_policy
class MaxEntRLPolicy(RLPolicy):
  """`RLPolicy` plus an automatic entropy dual -- SAC, AMDP, DIME and SDAC all
  resolve here. `entropy_in_target` is the one real behavioural fork: SDAC's
  Bellman target carries no entropy bonus (its `entropy` is a scalar over the
  exploration parameter, not the per-sample quantity the target wants), so it
  sets this false and the dual trains purely off the actor's DSM loss."""

  SLOTS = ("critic", "actor", "entropy")
  entropy_scale     = HParam(-1.0)
  entropy_in_target = Config(True)

  def __init__(self, env: gym.Env, **kwargs):
    super().__init__(env, **kwargs)

    self.entropy: AutotuneRegulator = self.build_component(
      "entropy", AutotuneRegulator,
      target=self.chunk_size * self.action_dim * self.entropy_scale)
    assert set(self.optimizer_cadence()) <= set(self.create_optimizer()), (
      f"{self.model_name}: optimizer_cadence names a key create_optimizer() "
      f"doesn't -- build_optimizer's cadence.get(key, 1) would drop it silently.")

  def optimizer_cadence(self) -> dict:
    return {**super().optimizer_cadence(), "entropy_alpha": self.policy_delay}

  def alpha(self):
    return self.entropy.value()

  def target_shift(self, alpha, entropy):
    if not self.entropy_in_target:
      return 0.0
    return jax.lax.stop_gradient(alpha * entropy)

  def dual_loss(self, entropy, losses) -> None:
    self.entropy.loss(entropy, losses)
