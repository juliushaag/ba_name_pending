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
see `networks/actors`) share these two classes entirely."""

import gymnasium as gym
import jax
import jax.numpy as jnp

from barl_jax.hparam import HParam
from barl_jax.networks.critic import QCriticBase
from barl_jax.networks.regulator import AutotuneRegulator
from barl_jax.policy import Actor, Policy, gated_loss, register_policy


@register_policy
class RLPolicy(Policy):
  """Actor + critic, no entropy dual -- `alpha`/`target_shift`/`dual_loss` are
  the three seams `MaxEntRLPolicy` fills in. On its own this is TD3-shaped:
  nothing in this repo's config points a model at `RLPolicy` directly, but it
  is the layout every actor-critic algorithm here shares, and an actor that
  adds no entropy term to its own objective needs no dual at all."""

  SLOTS = ("critic", "actor")
  policy_delay = HParam(1, descr="grad steps between actor/dual updates (critic steps every step)")
  normalize_obs = HParam(True)

  def __init__(self, env: gym.Env, **kwargs):
    super().__init__(env, **kwargs)

    assert isinstance(env.action_space, gym.spaces.Box)

    self.critic: QCriticBase = self.build_component("critic", QCriticBase)
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

    losses = {}
    alpha  = self.alpha()

    # Every draw this step descends from one key, split here: the bootstrap
    # sample, the actor loss and its extra losses (SDAC's exploration term)
    # must not share a stream, and a step's randomness should be readable off
    # train_step rather than hidden in each component's `rngs`.
    keys = jax.random.split(self.rngs.noise(), 3)

    next_action, entropy = jax.lax.stop_gradient(
      self.actor.generate_action(next_obs, critic=self.critic, key=keys[0]))

    self.critic.loss(obs, action, next_obs, next_action, reward, term,
                     losses, shift=self.target_shift(alpha, entropy))

    # Skipped outright on the steps `policy_delay` would mask the update on
    # anyway -- the loss prices through a frozen critic and its target is
    # stop_gradient'd, so nothing here reaches the critic's parameters and the
    # skip is parameter-identical (see `gated_loss`).
    losses["actor"] = gated_loss(
      self, self.policy_delay,
      lambda actor, critic: actor.loss(obs, critic, alpha, key=keys[1]),
      self.actor, self.critic)

    losses.update(self.actor.extra_losses(obs, self.critic, alpha, key=keys[2]))
    self.dual_loss(entropy, losses)

    self.actor.metrics.vec("action", next_action, limit=1.0)
    self.actor.metrics.dist("entropy", entropy)

    return losses

  def inference_step(self, observation, step_info=None, key=None):
    obs = self._obs(observation)
    return self.actor.sample(obs, critic=self.critic, key=key), {}


@register_policy
class MaxEntRLPolicy(RLPolicy):
  """`RLPolicy` plus an automatic entropy dual -- SAC, AMDP, DIME and SDAC all
  resolve here. `entropy_in_target` is the one real behavioural fork: SDAC's
  Bellman target carries no entropy bonus (its `entropy` is a scalar over the
  exploration parameter, not the per-sample quantity the target wants), so it
  sets this false and the dual trains purely off `extra_losses`."""

  SLOTS = ("critic", "actor", "entropy")
  entropy_scale     = HParam(-1.0)
  entropy_in_target = HParam(True)

  def __init__(self, env: gym.Env, **kwargs):
    super().__init__(env, **kwargs)

    self.entropy: AutotuneRegulator = self.build_component(
      "entropy", AutotuneRegulator, target=self.action_dim * self.entropy_scale)
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
