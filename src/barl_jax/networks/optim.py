import optax

from barl_jax.hparam import HParam


# ---------------------------------------------------------------------------
# Optimizer mixins -- the four optimizer knobs as class attributes, grouped by
# role so every component that optimizes inherits exactly the one that fits it
# (critic uses the critic's b1 and clip, actor the actor's, dual the dual's).
# All four are HParams: learning rate and clip are exactly the values a
# hyperparameter search varies.
# ---------------------------------------------------------------------------


class OptimizerConfig:
  lr            = HParam(3.0e-4, range=(1.0e-4, 1.0e-3))
  adam_b1       = HParam(0.9)
  grad_clip     = HParam(1.0)
  grad_clip_mode = HParam("element", choices=("global_norm", "element", "none"))


class CriticOptim(OptimizerConfig):
  """The critic's optimizer (dime_am's `optim.critic`): b1 0.5, global-norm
  clip -- the b1 the SAC-style critics train with."""

  adam_b1         = HParam(0.5)
  grad_clip_mode  = HParam("global_norm", choices=("global_norm", "element", "none"))


class ActorOptim(OptimizerConfig):
  """The actors' optimizer (`optim.actor`): Adam with a per-element clip, which
  is what actually bounds what one outlier sample can contribute under Adam."""

  grad_clip_mode = HParam("element", choices=("global_norm", "element", "none"))


class DualOptim(OptimizerConfig):
  """The entropy dual's optimizer (`optim.dual`): a higher LR, unclipped."""

  lr            = HParam(1.0e-3)
  grad_clip_mode = HParam("global_norm", choices=("global_norm", "element", "none"))


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
