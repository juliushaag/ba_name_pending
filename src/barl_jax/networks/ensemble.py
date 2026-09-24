from flax import nnx


class Ensemble(nnx.Module):
  """`n` independently-initialized copies of `cls(*args, **kwargs)` run as a
  single vmapped call. Returns the un-reduced `(n, ...)` stack; how it is
  reduced belongs to whoever owns it (see QCritic.reduce_value).

  The torch version hand-rolled this out of stack_module_state +
  functional_call + a meta-device template; here it is what nnx.vmap does
  natively. `nnx.split_rngs(splits=n)` is the load-bearing part: without it
  every member draws the same init key and the ensemble collapses to `n`
  identical heads. Nothing errors -- the spread is simply 0, which silently
  turns a `min` reduction into a plain mean and disables pessimism for the
  whole run."""

  def __init__(self, n: int, cls: type, *args, rngs: nnx.Rngs, **kwargs):
    @nnx.split_rngs(splits=n)
    @nnx.vmap(in_axes=(0,), out_axes=0)
    def make(rngs: nnx.Rngs):
      return cls(*args, rngs=rngs, **kwargs)

    self.members = make(rngs)
    self.n_models = n

  def __call__(self, *args):
    @nnx.vmap(in_axes=(0,) + (None,) * len(args), out_axes=0)
    def call(member, *args):
      return member(*args)

    return call(self.members, *args)
