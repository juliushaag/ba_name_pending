"""Optimizer ownership, partitioning, gradient steps, and schedules."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from barl_jax.components import TRAINABLE
from barl_jax.metrics import DEBUG, PRIMARY, DebugMetric, Metric, armed, collect, recording


class OptimizerMixin:
  # Parameters a Policy owns directly rather than through a slot, mapped to
  # the optimizer key that trains them: `{"log_std": "explore"}`. Without an
  # entry a stray parameter is a hard error rather than a silently untrained
  # one (torch's add_param_group had the same job, minus the check).
  policy_params: dict[str, str] = {}

  def optimizer_slots(self) -> dict[str, str]:
    """Optimizer key -> owning component slot, by asking each slot which keys
    it creates. Keys owned by no component map to this algorithm's own
    namespace. Building the transforms is free -- optax allocates no state
    until init()."""
    cache = self._optimizer_slots_cache.value
    if cache["value"] is None:
      owners = {}
      for slot, component in self.components.items():
        for key in component.create_optimizer():
          owners[key] = slot
      cache["value"] = owners
    return cache["value"]

  def metrics_key(self, optimizer_key: str, field: str) -> str:
    """`critic/loss` when the slot owns one optimizer key, `actor/loss/flow`
    when it owns several -- so the common case stays flat."""
    owners = self.optimizer_slots()
    if optimizer_key not in owners:  # no component owns it -- always name it
      return f"{self.algo_name()}/{field}/{optimizer_key}"
    slot   = owners[optimizer_key]
    shared = sum(1 for o in owners.values() if o == slot) > 1
    return f"{slot}/{field}/{optimizer_key}" if shared else f"{slot}/{field}"

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

    The clip itself lives in the component (optimization.adam, spent from
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
        tx = optax.conditionally_mask(tx, lambda step, k=every: update_due(step, k))
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


# ---------------------------------------------------------------------------
# Loss groups and optimizer masks share cadence declarations and predicates.
# Diagnostic evaluation may run off-cadence, but must not change training state.
# ---------------------------------------------------------------------------

_GRAD_STEP = ContextVar("gradient_step", default=None)


@contextmanager
def gating(step):
  """Keep the shared step unbatched under seed vmap, scoped to this trace.

  Reading the per-policy counter under vmap batches the predicate and turns
  cond into select (executing both branches). The caller supplies in_axes=None.
  ContextVar also isolates concurrent/nested traces instead of sharing a global.
  """
  token = _GRAD_STEP.set(step)
  try:
    yield
  finally:
    _GRAD_STEP.reset(token)


def update_due(step, every: int):
  """A zero-based gradient step updates every kth time, starting at k - 1."""
  return True if every == 1 else (step + 1) % every == 0


def gated_losses(policy, fn, keys, *modules):
  """Evaluate a loss group whenever any of its optimizer keys is due.

  Cadences come from optimizer_cadence(), the same source as optimizer masks.
  Each loss must reach only its declared optimizer partitions; modules must
  include all mutable NNX state touched by fn. Optimizers still mask each key
  independently when a group contains different cadences.

  Logged steps evaluate real diagnostic losses even when no key is due, but
  roll back non-metric NNX state (including batch statistics and RNG counters).
  Unlogged skipped steps execute no loss computation and return scalar zeros.
  """
  keys = tuple(keys)
  if not keys or len(set(keys)) != len(keys):
    raise ValueError("A gated loss group needs distinct optimizer keys")
  unknown = set(keys) - set(policy.create_optimizer())
  if unknown:
    raise ValueError(f"Gated loss names unknown optimizer keys: {sorted(unknown)}")
  cadence = policy.optimizer_cadence()
  periods = tuple(cadence.get(key, 1) for key in keys)
  if any(not isinstance(k, int) or isinstance(k, bool) or k < 1 for k in periods):
    raise ValueError("Optimizer cadence must contain positive integers")
  step = _GRAD_STEP.get()
  step = policy.train_steps[...] if step is None else step
  due = True if 1 in periods else jnp.any(jnp.stack([update_due(step, k) for k in periods]))

  def evaluate(*modules):
    losses = fn(*modules)
    if set(losses) != set(keys):
      raise ValueError(f"Gated loss keys {sorted(losses)} differ from declared keys {sorted(keys)}")
    return losses

  if due is True:
    return evaluate(*modules)
  if armed():
    # nnx.state alone aliases live Variables: clone their holders before fn
    # mutates them. Immutable array buffers remain shared; no device copy.
    persistent = nnx.Not((Metric, DebugMetric))
    before = nnx.state(nnx.clone(modules), persistent)
    losses = evaluate(*modules)
    after = nnx.state(modules, persistent)
    nnx.update(modules, jax.tree.map(lambda old, new: jnp.where(due, new, old), before, after))
    return losses
  return nnx.cond(due, evaluate, lambda *_: {k: jnp.zeros(()) for k in keys}, *modules)


# ---------------------------------------------------------------------------
# Optimizer configuration: the four Adam knobs, with per-role defaults.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OptimizerConfig:
  lr: float = 3.0e-4
  adam_b1: float = 0.9
  grad_clip: float = 1.0
  grad_clip_mode: str = "element"

  def problems(self) -> list[str]:
    if self.grad_clip_mode in CLIP_MODES:
      return []
    return [f"grad_clip_mode={self.grad_clip_mode!r}: unknown clip mode"]

  def build(self):
    """This component's optax transform: NaN guard, its own clip, then Adam."""
    return adam(self.lr, self.adam_b1, self.grad_clip, self.grad_clip_mode)


@dataclass(frozen=True)
class CriticOptimConfig(OptimizerConfig):
  adam_b1: float = 0.5
  grad_clip_mode: str = "global_norm"


@dataclass(frozen=True)
class ActorOptimConfig(OptimizerConfig):
  pass


@dataclass(frozen=True)
class DualOptimConfig(OptimizerConfig):
  lr: float = 1.0e-3
  grad_clip_mode: str = "global_norm"


class ActorOptim:
  """Mixin for an actor trained by one Adam optimizer under the key "actor"."""

  def create_optimizer(self) -> dict:
    return {"actor": self.config.build()}


CLIP_MODES = ("global_norm", "element", "none")


def grad_clip_transform(grad_clip: float, grad_clip_mode: str):
  """The clip a component puts ahead of its Adam.

  The two modes are different operations, not two strengths of one, which is why
  this is a component's choice and not a run-level number. `global_norm`
  rescales the whole gradient to `grad_clip` and so preserves its direction
  exactly -- and since Adam normalizes per coordinate anyway, that leaves an
  outlier sample free to set the update's direction while the length it removed
  never mattered. `element` bounds each coordinate at +-`grad_clip`, which is
  what actually attenuates one sample's contribution. dime_am clips its control
  net the element-wise way (`cont_time_policy.py`: `chain(zero_nans(), clip(1.0),
  adam(lr))`) and leaves critic and dual unclipped."""
  if grad_clip_mode not in CLIP_MODES:
    raise KeyError(f"grad_clip_mode {grad_clip_mode!r} is not one of {', '.join(CLIP_MODES)}")

  return {"global_norm": lambda: optax.clip_by_global_norm(float(grad_clip)),
          "element":     lambda: optax.clip(float(grad_clip)),
          "none":        optax.identity}[grad_clip_mode]()


def adam(lr: float, adam_b1: float, grad_clip: float, grad_clip_mode: str):
  """This component's optimizer: NaN guard, its own gradient clip, then Adam.

  `zero_nans` goes first so an infinity cannot drag a global norm to inf and
  scale the whole update to nothing. It hides nothing that was being reported:
  the `*/grad_frac_nonfinite` metric is taken from the raw gradient."""
  return optax.chain(optax.zero_nans(),
                     grad_clip_transform(grad_clip, grad_clip_mode),
                     optax.adam(float(lr), b1=float(adam_b1)))


# ---------------------------------------------------------------------------
# The jitted gradient step -- computes gradients, applies optimizers, and
# batches updates. Trainer owns experiment I/O (rollout, eval, checkpointing,
# logging); this module owns the numerics of one step and the N-seed/K-chunk
# dispatch shapes built on top of it.
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnums=(0, 3))
def train_step(graphdef, state, batch, tiers, grad_step=None):
  """One jitted gradient step over one policy state."""
  policy, opt = nnx.merge(graphdef, state)

  def scalar_loss(policy):
    policy.record_batch(batch)
    losses = policy.train_step(batch)
    expected = set(policy.create_optimizer())
    if set(losses) != expected:
      raise ValueError(f"Loss keys {sorted(losses)} differ from optimizer keys {sorted(expected)}")
    return sum(losses.values()), losses

  def loss_fn(policy):
    total, losses = scalar_loss(policy)
    sown, sown_debug = nnx.pop(policy, Metric, DebugMetric)
    return total, (losses, sown, sown_debug)

  with recording(*tiers), gating(grad_step):
    policy.update_observation_stats(batch["obs"])
    policy.before_step()
    (_, (losses, sown, sown_debug)), grads = nnx.value_and_grad(
      loss_fn, argnums=nnx.DiffState(0, TRAINABLE), has_aux=True)(policy)

  primary = collect(sown)
  debug   = collect(sown_debug)

  if tiers:
    labels = dict(nnx.to_flat_state(policy.optimizer_labels()))
    flat_grads = nnx.to_flat_state(grads)
    for key in losses:
      partition = [_raw(v).astype(jnp.float32) for path, v in flat_grads
                   if labels.get(path) == key]
      if PRIMARY in tiers:
        primary[policy.metrics_key(key, "loss")] = losses[key]
        primary[policy.metrics_key(key, "grad_norm")] = optax.tree.norm(partition)
      if DEBUG in tiers and partition:
        total   = sum(g.size for g in partition)
        zeros   = sum((g == 0).sum() for g in partition)
        nonfin  = sum((~jnp.isfinite(g)).sum() for g in partition)
        gmax    = jnp.max(jnp.stack([jnp.abs(g).max() for g in partition]))
        debug[policy.metrics_key(key, "grad_max")] = gmax
        debug[policy.metrics_key(key, "grad_frac_zero")] = zeros / total
        debug[policy.metrics_key(key, "grad_frac_nonfinite")] = nonfin / total

  opt.update(policy, grads)
  policy.after_step()

  return nnx.state((policy, opt)), losses, (primary, _pack(primary)), (debug, _pack(debug))


def _pack(metrics: dict):
  if not metrics:
    return jnp.zeros((0,))
  return jnp.stack([jnp.asarray(metrics[k]) for k in sorted(metrics)])


def _raw(v):
  return v[...] if hasattr(v, "__getitem__") and not isinstance(v, jax.Array) else v

# `grad_step` rides in at in_axes=None: it is the same number for every seed, and
# a per-seed copy of it would batch the `policy_delay` predicate, which turns the
# lax.cond guarding a delayed actor loss into a select over both branches.
seeded_step = jax.jit(jax.vmap(train_step, in_axes=(None, 0, 0, None, None)),
                      static_argnums=(0, 3), donate_argnums=(1,))


@partial(jax.jit, static_argnums=(0, 3), donate_argnums=(1,))
def seeded_chunk(graphdef, state, batches, k, grad_step):
  """`k` gradient steps in one dispatch, over a `(k, num_seeds, batch, ...)` chunk.

  The same update as `k` calls of `seeded_step` with no metric tier armed: the
  same states in the same order, the same per-step `grad_step` feeding the
  `policy_delay` gate, and the same batches (see `sample_chunk`). What it drops
  is the Python between them -- ~1.6 ms of jit dispatch plus a `device_put` per
  step, which for AMDP is 1.38x of the step and for DIME 1.22x.

  The same update, but **not bit-identical**: XLA schedules the fused scan
  differently from a standalone jit, and float addition does not associate. In
  float32 that is 1e-11 on two leaves after one step; in the bfloat16 the nets
  actually use, 2.5e-5, because bfloat16 carries 8 mantissa bits. It is
  reassociation and not a second algorithm -- but a run does not replay a
  pre-chunking run step for step.

  Tier-armed steps do not come through here. They return metrics, and `tiers`
  is static, so an armed step inside the scan would be a second compiled shape
  per position it could occupy; the caller runs those iterations step-by-step
  instead. `k` likewise stays the iteration's whole gradient-step count -- one
  value for the life of a run, hence one compile."""
  def body(carry, batch):
    state, step = carry
    state, _, _, _ = jax.vmap(train_step, in_axes=(None, 0, 0, None, None))(
      graphdef, state, batch, (), step)
    return (state, step + 1), None

  (state, _), _ = jax.lax.scan(body, (state, grad_step), batches, length=k)
  return state

