"""The shared off-policy actor-critic template.

SAC, AMDP and DIME's `train_step`s were, at bottom, one function: a
stop-gradiented bootstrap draw at `next_obs`, a critic Bellman target with an
optional entropy bonus, an actor loss gated at some cadence, and an entropy
dual fed the bootstrap's entropy. What differed was four things -- obs
normalisation, the gate's cadence, the name of the actor's generator, and
whether the dual is fed anything at all -- and none of those is an algorithm.

`RLPolicy` states that shared step once; `MaxEntRLPolicy` adds the dual. Every
algorithm reuses this template: `SAC`, `AMDP` and `DIME` (each beside its
actor, in `barl_jax.actors.*`) are thin `MaxEntRLPolicy` subclasses that pin
`actor_cls`, the actor-policy integration binding for that algorithm; `NCLQL`
and `SDAC` pin one on `RLPolicy` directly (no entropy dual). The algorithm
difference is entirely in which actor/critic/params a model variant's
`ModelConfig` selects and, for SDAC, the `actor_objective` seam, never in a
second copy of this training step."""

from dataclasses import dataclass

import gymnasium as gym
import jax
import jax.numpy as jnp

from barl_jax.components import ActionSampler
from barl_jax.critics import QCritic
from barl_jax.networks.regulator import AutotuneRegulator
from barl_jax.optimization import gated_losses
from barl_jax.policy import Policy, register_policy

ALPHA_GRANULARITIES = ("per_chunk", "per_action")


@dataclass(frozen=True)
class PolicyConfig:
  """Params every off-policy actor-critic policy owns."""

  chunk_size: int = 1
  normalize_obs: bool = True
  policy_delay: int = 1


@dataclass(frozen=True)
class MaxEntPolicyConfig(PolicyConfig):
  entropy_scale: float = -1.0
  entropy_in_target: bool = True
  alpha_granularity: str = "per_chunk"


@register_policy
class RLPolicy(Policy):
  """Actor + critic, no entropy dual -- `alpha`/`target_entropy_bonus`/`dual_loss`
  are the three seams `MaxEntRLPolicy` fills in. On its own this is TD3-shaped:
  nothing in this repo's config points a model at `RLPolicy` directly, but it
  is the layout every actor-critic algorithm here shares, and an actor that
  adds no entropy term to its own objective needs no dual at all."""

  SLOTS = ("critic", "actor")
  @classmethod
  def slot_objective(cls, slot, actor_cls=None):
    if slot == "critic" and actor_cls is not None:
      return cls.SLOT_OBJECTIVES.get(slot, actor_cls.critic_objective)
    return super().slot_objective(slot, actor_cls)

  # The actor-policy integration binding: a per-algorithm subclass (`SAC`,
  # `AMDP`, `DIME`, `SDAC`, `NCLQL`, each beside its actor implementation) sets
  # this to its concrete actor class, tightening build_component's gate from
  # the generic ActionSampler ABC. A `components.actor.class` mismatch then
  # fails at construction with a clear error naming both classes, instead of
  # silently accepting whatever the config named.
  actor_cls: type = ActionSampler

  def __init__(self, env: gym.Env, *args, **kwargs):
    super().__init__(env, *args, **kwargs)
    assert isinstance(env.action_space, gym.spaces.Box)

    chunked = self.env_stats.derive(act_dim=self.env_stats["chunked_act_dim"])
    self.critic: QCritic = self.build_component("critic", QCritic, env_stats=chunked)
    self.actor: ActionSampler = self.build_component("actor", self.actor_cls)
    # No cadence/optimizer-key guard here: `self.optimizer_cadence()` is a
    # virtual call, and a MaxEntRLPolicy subclass hasn't built `self.entropy`
    # yet at this point in `super().__init__()` -- the check belongs at the
    # end of whichever `__init__` runs last (MaxEntRLPolicy's, today).

  @property
  def policy_delay(self) -> int:
    """Grad steps between actor/dual updates (the critic steps every step)."""
    return self.params.policy_delay

  @classmethod
  def config_problems(cls, params) -> list[str]:
    return [] if params.policy_delay >= 1 else [f"policy_delay={params.policy_delay!r}: must be >= 1"]

  def optimizer_cadence(self) -> dict:
    return {key: self.policy_delay for key in self.actor.create_optimizer()}

  def actor_objective(self, actor, critic, obs, action, alpha, key):
    return actor.loss(obs, critic, alpha, key=key)

  # ---- the three seams MaxEntRLPolicy fills in -----------------------------

  def alpha(self):
    return jnp.asarray(1.0)

  def target_entropy_bonus(self, alpha, entropy):
    return 0.0

  def dual_loss(self, entropy) -> dict:
    return {}

  # ---------------------------------------------------------------------

  @property
  def discount(self):
    return self.critic.config.gamma

  def train_step(self, batch):
    obs          = self.normalize_observations(batch["obs"])
    next_obs     = self.normalize_observations(batch["next_obs"])
    
    action       = batch["act"]
    chunk_return = batch["chunk_return"]
    terminated   = batch["term"]
    horizon      = batch["horizon"]

    losses = {}
    alpha  = self.alpha()

    # Every draw this step descends from one key, split here: the bootstrap
    # sample and the actor losses must not share a stream, and a step's
    # randomness should be readable off train_step rather than hidden in each
    # component's `rngs`. keys[2] is retired -- the explore term shares the
    # actor's stream now (one best-of-N) -- but the split stays 3-way so the
    # other two streams are unchanged.
    actor_keys = tuple(sorted(self.actor.create_optimizer()))
    # Critic-only samplers retain the original bootstrap/smoothing split.
    keys = jax.random.split(self.rngs.noise(), 3 if actor_keys else 2)

    next_action, entropy = jax.lax.stop_gradient(
      self.actor.generate_action(next_obs, critic=self.critic, key=keys[0]))
    next_action = next_action.reshape(next_action.shape[0], -1)

    losses.update(self.critic.td_loss(
      obs, action, next_obs, next_action, chunk_return, terminated, horizon,
      entropy_bonus=self.target_entropy_bonus(alpha, entropy), key=keys[-1]))

    # An actor can own multiple optimizers. Compute their shared loss whenever any partition is due; mask each
    # optimizer independently according to optimizer_cadence().
    if actor_keys:
      losses.update(gated_losses(
        self,
        lambda actor, critic: self.actor_objective(actor, critic, obs, action, alpha, keys[1]),
        actor_keys, self.actor, self.critic))
    losses.update(self.dual_loss(entropy))

    self.actor.metrics.vec("action", next_action.reshape(
      next_action.shape[0], self.actor.chunk_size, self.actor.action_dim), limit=1.0)
    if entropy is not None:
      self.actor.metrics.dist("entropy", entropy)

    return losses

  def inference_step(self, observation, step_info=None, key=None):
    obs = self.normalize_observations(observation)
    return self.action_chunk(self.actor.sample(obs, critic=self.critic, key=key)), {}

  # ---- Best-of-N scoring (barl_jax.executors) --------------------------

  @property
  def supports_bon(self):
    return self.critic.supports_bon

  def score(self, observation, action_chunk):
    """`critic.value()` at the online critic, reduced per
    `ensemble.value_reduce`. Flattens the action plan the
    same way `train_step` flattens `next_action` before handing it to the
    critic."""
    obs = self.normalize_observations(observation)
    flat_action = action_chunk.reshape(action_chunk.shape[0], -1)
    return self.critic.value(obs, flat_action)


@register_policy
class MaxEntRLPolicy(RLPolicy):
  """`RLPolicy` plus an automatic entropy dual -- SAC, AMDP and DIME all
  resolve here. `entropy_in_target` toggles the Bellman target's entropy bonus.

  `alpha_granularity` is the other fork: `"per_chunk"` (default) trains one
  alpha for the whole chunked plan, exactly as before; `"per_action"` trains
  one alpha per chunk position, each held at its own per-action target entropy
  `action_dim * entropy_scale`. Every actor's `generate_action` already reports
  entropy decomposed per chunk position (`(B, chunk_size)`), so the only thing that changes between the two
  is `target`'s shape -- `AutotuneRegulator` follows it -- and whether the dual
  sees that decomposition summed to one number or as-is."""

  SLOTS = ("critic", "actor", "entropy")
  SLOT_OBJECTIVES = {"critic": "td"}

  def __init__(self, env: gym.Env, *args, **kwargs):
    super().__init__(env, *args, **kwargs)
    params = self.params
    target = (jnp.full((self.chunk_size,), self.action_dim * params.entropy_scale, jnp.float32)
              if params.alpha_granularity == "per_action"
              else self.chunk_size * self.action_dim * params.entropy_scale)
    self.entropy: AutotuneRegulator = self.build_component(
      "entropy", AutotuneRegulator, target=target)
    assert set(self.optimizer_cadence()) <= set(self.create_optimizer()), (
      f"{self.model_name}: optimizer_cadence names a key create_optimizer() "
      f"doesn't -- build_optimizer's cadence.get(key, 1) would drop it silently.")

  @classmethod
  def config_problems(cls, params) -> list[str]:
    problems = super().config_problems(params)
    if params.alpha_granularity not in ALPHA_GRANULARITIES:
      problems.append(f"alpha_granularity={params.alpha_granularity!r}: "
                      f"must be one of {ALPHA_GRANULARITIES}")
    return problems

  def optimizer_cadence(self) -> dict:
    return {**super().optimizer_cadence(),
            **{key: self.policy_delay for key in self.entropy.create_optimizer()}}

  def alpha(self):
    return self.entropy.value()

  def target_entropy_bonus(self, alpha, entropy):
    """`entropy` is `(B, chunk_size)`; summing the chunk axis after weighting by
    `alpha` (scalar or `(chunk_size,)`) restores the single `(B, 1)` bonus the
    Bellman target wants, whichever granularity produced it."""
    if not self.params.entropy_in_target:
      return 0.0
    return jax.lax.stop_gradient((alpha * entropy).sum(axis=-1, keepdims=True))

  def dual_loss(self, entropy) -> dict:
    # per_chunk's target is one number for the whole plan, so the dual must see
    # entropy summed the same way; per_action's target is per position, so the
    # per-position decomposition goes in as-is.
    value = entropy if self.params.alpha_granularity == "per_action" else entropy.sum(axis=-1, keepdims=True)
    return self.entropy.loss(value)
