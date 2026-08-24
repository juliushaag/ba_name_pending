import optax


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
