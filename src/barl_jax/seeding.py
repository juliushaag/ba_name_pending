"""Stable seed derivation -- `hash()` is randomized per Python process
(`PYTHONHASHSEED`), so it cannot be the thing that turns "run seed, eval
identity, slot" into a reproducible episode. `numpy.random.SeedSequence` is
deterministic across processes and numpy versions; this just gives it a
stable integer encoding for the non-int parts (a label, an index)."""

import zlib

import numpy as np


def derive_seed(*parts) -> int:
  """A 32-bit int, stable across processes, for anything -- `jax.random.PRNGKey`,
  `env.reset(seed=...)` -- that wants one deterministic seed derived from a
  tuple of identities. Same `parts` always derives the same seed; different
  `parts` derive independent ones (SeedSequence's own guarantee, not just a
  hash mix), so composing more identities into `parts` never has to be
  checked for accidental collision."""
  entropy = [zlib.crc32(str(p).encode()) if not isinstance(p, int) else p for p in parts]
  return int(np.random.SeedSequence(entropy).generate_state(1)[0])
