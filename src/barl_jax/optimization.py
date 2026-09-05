"""Optimizer ownership, partitioning, and schedules."""
import jax.numpy as jnp
import optax
from flax import nnx

from barl_jax.components import TRAINABLE


class OptimizerMixin:
  def create_optimizer(self) -> dict:
    """Optimizer key -> optax transform, unioned over every slot. A Policy with
    a tunable of its own (an exploration std owned by no component) overrides
    this and adds its key."""
    out = {}
    for component in self.components.values():
      for key, transform in component.create_optimizer().items():
        if key in out:
          raise KeyError(f"{self.model_name}: optimizer key {key!r} is owned by more than one "
                         "component slot")
        out[key] = transform
    return out

  def optimizer_cadence(self) -> dict:
    """Optimizer key -> grad steps between its updates; absent means every step.

    TD3's delayed policy update, which dime_am carries as `policy_delay: 3`
    (`am_diff.py`: the critic steps every gradient step, `update_actor` and
    `update_temperature` only when `(n_updates + i + 1) % policy_delay == 0`).
    The point is to let the critic settle before the actor chases it, so it is a
    property of the algorithm and belongs to the Policy rather than to a slot."""
    return {}

  def optimizer_labels(self) -> dict:
    """The `param path -> optimizer key` tree optax.multi_transform partitions
    on. A slot's name *is* its attribute name (see `components`), so a
    parameter's owner is just the head of its path -- no identity matching
    against the instance dict, and nothing to keep in sync.

    A component owning more than one key overrides `param_labels(subpath)` to
    say which of its sub-attributes maps to which."""
    owners = self.optimizer_slots()
    default_key = {}
    for key, slot in owners.items():
      default_key.setdefault(slot, key)

    components = self.components
    labels = []
    for path, _ in nnx.to_flat_state(nnx.state(self, TRAINABLE)):
      slot = str(path[0])
      component = components.get(slot)
      key = None
      if component is not None:
        split = getattr(component, "param_labels", None)
        key = (split(path[1:]) if split else None) or default_key.get(slot)
      labels.append((path, key or self._fallback_label(path)))
    return nnx.from_flat_state(labels)


  def _fallback_label(self, path) -> str:
    key = self.policy_params.get(str(path[0]))
    if key is not None:
      return key
    raise KeyError(
      f"{self.model_name}: parameter {'.'.join(map(str, path))} belongs to no "
      f"component slot, so no optimizer key owns it. Either build it through "
      f"build_component(), or declare it in `policy_params`."
    )

  def build_optimizer(self, *, lr_schedule):
    """One optax.multi_transform over the whole param tree. multi_transform
    applies each transform to its own partition, so each component's own clip
    and Adam see only that component's gradients -- exactly what the torch
    loop's clip_grad_norm_(params_of_key_k, ...) did, in one backward pass
    instead of one per key.

    The clip itself lives in the component (networks/optim.adam, spent from
    config/presets/optim.yaml), not here: `global_norm` and `element` are
    different operations rather than two strengths of one, and which of them a
    net wants is a property of that net's gradients."""
    cadence = self.optimizer_cadence()

    declared = self.create_optimizer()
    unknown = set(cadence) - set(declared)
    if unknown:
      raise ValueError(f"Optimizer cadence names unknown keys: {sorted(unknown)}")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 1
           for value in cadence.values()):
      raise ValueError("Optimizer cadence must contain positive integers")
    transforms = {}
    for key, tx in declared.items():
      every = int(cadence.get(key, 1))
      if every > 1:
        # Masks the *update* and leaves the inner state alone, so a skipped step
        # does not decay Adam's moments -- which zeroing the gradient instead
        # would not give: Adam with g=0 still moves on accumulated momentum.
        # Wrapped around `tx` alone rather than the whole chain, so the LR
        # schedule outside it still advances once per grad step.
        tx = optax.conditionally_mask(tx, lambda step, k=every: (step + 1) % k == 0)
      transforms[key] = (tx if lr_schedule is None else
                         optax.chain(tx, optax.scale_by_schedule(lr_schedule)))
    return optax.multi_transform(transforms, self.optimizer_labels())

  DECAYS = {
    # Quarter-sine 1 -> 0: flat for the first part of the run, steepest at the end.
    "sine":     lambda progress: jnp.sin((1.0 - progress) * jnp.pi / 2),
    "linear":   lambda progress: 1.0 - progress,
    "cosine":   lambda progress: 0.5 * (1.0 + jnp.cos(progress * jnp.pi)),
    "constant": lambda progress: jnp.ones_like(progress),
  }

  @staticmethod
  def lr_schedule(warmup_steps: int, total_steps: int, decay: str):
    """Multiplicative LR factor: linear 0 -> 1 ramp over `warmup_steps`, then
    `decay` from 1 -> 0 by `total_steps` (flat past that, in case `total_steps` --
    an estimate -- undershoots the run's actual grad-step count). `decay` is the
    run config's `lr_decay`; it used to be hardcoded to "sine" here, which meant
    the schedule's shape was the one part of the optimizer no run recorded."""
    if decay not in OptimizerMixin.DECAYS:
      raise KeyError(f"lr_decay {decay!r} is not one of {', '.join(OptimizerMixin.DECAYS)}")
    decay_fn = OptimizerMixin.DECAYS[decay]

    def schedule(step):
      step = jnp.asarray(step, jnp.float32)
      warm = step / max(1, warmup_steps)
      progress = jnp.clip((step - warmup_steps) / max(1, total_steps - warmup_steps), 0.0, 1.0)
      decayed = decay_fn(progress)
      return jnp.where(step < warmup_steps, warm, decayed) if warmup_steps > 0 else decayed
    return schedule

