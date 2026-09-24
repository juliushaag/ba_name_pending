"""Replay storage and independently seeded sampling."""
import jax
import numpy as np


class ReplayBuffer():
  """A flat ring buffer of single transitions: one preallocated array per field,
  written at a rolling index, evicted oldest-first.

  Replaces the list-of-chunks this used to be, whose `sample()` read from a
  full re-concatenation of every chunk rebuilt whenever the buffer changed.
  That is O(buffer) per rollout and superlinear in the *chunk count*, and the
  chunk length was `min(traj_len, n_steps)` -- so shortening a rollout shortened
  the chunks too and the rebuild went 120 ms (1M transitions, 32-step chunks) to
  3.8 s (1M one-transition chunks). Appending here is O(rows added), which is
  what lets `rollout_steps` drop to `num_envs` without the rebuild dominating.

  `next_obs` is stored per transition rather than reconstructed from the next
  row. It costs a second observation array -- the same memory the chunk list and
  its flattened copy already used between them -- and in exchange there are no
  seams: the observation the env returned *before* its reset is the one stored,
  so a truncation boundary needs no special handling, and oldest-first eviction
  can cut anywhere.

  `term` is termination only, never truncation: a time-limit cut is not a
  terminal state and must keep its bootstrap."""

  def __init__(self, max_transitions : int, seed=None):
    self.rng = np.random.default_rng(seed)
    self.max_transitions = int(max_transitions)
    if self.max_transitions < 1:
      raise ValueError("Replay capacity must be positive")
    self._fields = None
    self._write  = 0
    self._size   = 0
  @property
  def n_transitions(self) -> int:
    return self._size

  def _allocate(self, row: dict) -> None:
    self._fields = {
      name: np.empty((self.max_transitions, *value.shape[1:]), dtype=value.dtype)
      for name, value in row.items()
    }

  def add(self, **row) -> None:
    """Append a batch of transitions, one row per sub-env slot. Every array must
    have the same leading length; each is stored under its keyword name and
    comes back out of `sample_numpy()` under that same name.

    Validates the whole batch -- field set, per-field leading length, dtype,
    and per-row (trailing) shape -- before writing anything. Fields are
    written one at a time below, so validating field N after fields 0..N-1
    already landed would leave those rows in the buffer with only some of
    their fields updated for that logical transition, at an index `_write`/
    `_size` had not yet advanced past -- a corruption invisible until the
    stale field is later sampled. A rejected batch instead leaves the buffer
    exactly as it was."""
    row.pop("slot_ids", None)
    first_write = self._fields is None
    if first_write:
      self._allocate(row)

    if row.keys() != self._fields.keys():
      raise KeyError(
        f"transition fields changed mid-run: buffer holds {sorted(self._fields)}, "
        f"got {sorted(row)} -- a ring buffer cannot backfill the rows already written")

    lengths = {name: len(value) for name, value in row.items()}
    if len(set(lengths.values())) > 1:
      raise ValueError(f"transition fields disagree on batch length: {lengths} -- "
                       f"every field in one add() call is one row per sub-env slot")
    n = next(iter(lengths.values()))

    if n > self.max_transitions:
      raise ValueError(f"{n} rows at once exceeds the buffer's {self.max_transitions}")

    # A field allocated from a shorter-lived dtype/shape than a later call
    # supplies would otherwise cast (dtype) or raise mid-write (shape) --
    # neither of which is "reject the batch, buffer unchanged". Skipped on
    # the very first call: _allocate() just took this row's own dtype/shape
    # as the buffer's, so it trivially matches.
    if not first_write:
      for name, value in row.items():
        expected = self._fields[name]
        if value.dtype != expected.dtype:
          raise TypeError(f"{name}: dtype {value.dtype} does not match the buffer's "
                          f"{expected.dtype} -- cast explicitly if this is intended")
        if value.shape[1:] != expected.shape[1:]:
          raise ValueError(f"{name}: per-row shape {value.shape[1:]} does not match "
                           f"the buffer's {expected.shape[1:]}")

    if n == 0:
      return

    # One wrapped index array rather than two slice assignments: n is num_envs
    # per step, so this is a handful of rows and the fancy index costs less than
    # the branch would.
    idx = (self._write + np.arange(n)) % self.max_transitions
    for name, value in row.items():
      self._fields[name][idx] = value

    self._write = int((self._write + n) % self.max_transitions)
    self._size  = int(min(self._size + n, self.max_transitions))
    self._on_write(idx)

  def _on_write(self, idx: "np.ndarray") -> None:
    """Hook for a subclass that keeps per-slot bookkeeping alongside the
    fields (PrioritizedReplayBuffer's priority and write-generation arrays)
    -- called with the ring indices just written, after they are committed.
    A plain ReplayBuffer has nothing to do here."""

  def sample(self, batch_size : int) -> dict:
    """Draws `batch_size` i.i.d. random transition indices across the whole
    buffer, with replacement (standard SAC-style minibatch). Returns
    the same keys as the trainer expects, already on the device."""
    return jax.device_put(self.sample_numpy(batch_size))

  def sample_numpy(self, batch_size: int) -> dict:
    """`sample()` without the device transfer. A caller drawing several batches
    at once (the multi-seed trainer, one per seed) stacks them on the host and
    crosses once, rather than once per seed per field."""
    if self._size == 0:
      raise ValueError("sample() from an empty buffer")
    idx = self.rng.integers(0, self._size, size=batch_size)
    return {name: value[idx] for name, value in self._fields.items()}

  def snapshot(self) -> dict:
    """Everything needed to exactly restore this buffer: physical ring
    contents (truncated to `_size`, in on-disk field-array form -- a training
    checkpoint's job is to store and publish this, never to read or write the
    ring layout itself), the write cursor, and the sampling RNG's state."""
    fields = self._fields or {}
    return {
      "fields": {name: value[:self._size] for name, value in fields.items()},
      "size": self._size,
      "write": self._write,
      "rng": self.rng.bit_generator.state,
    }

  def restore(self, snapshot: dict) -> None:
    """Inverse of `snapshot()`. `snapshot["fields"]` may be empty (a buffer
    that was never written to when the snapshot was taken) -- restoring
    stays a no-op on the ring contents in that case, matching `_allocate`'s
    own lazy-on-first-write behavior."""
    fields = snapshot["fields"]
    if fields:
      self._allocate(fields)
      for name, values in fields.items():
        self._fields[name][:len(values)] = values
    self._size, self._write = snapshot["size"], snapshot["write"]
    self.rng.bit_generator.state = snapshot["rng"]


class PrioritizedReplayBuffer(ReplayBuffer):
  """Proportional prioritized sampling (Schaul, Quan, Antonoglou & Silver
  2016, "Prioritized Experience Replay") over the same ring-buffer storage
  `ReplayBuffer` uses -- priorities and a per-slot write generation live in
  parallel arrays, not among `_fields`, so they never appear in a sampled
  transition batch.

  A `sample_id` is `generation[index] * max_transitions + index`: stable
  across the row's lifetime (it does not change while the row is not
  overwritten) and distinguishable from any later row a wrapping write puts
  at the same ring index, since that write bumps `generation[index]`.
  `update_priorities` uses this to silently drop a priority update whose row
  has since been overwritten -- a stale update from a training step that ran
  concurrently with (or long after) the sample it was scored from, rather
  than corrupting a different, newer transition's priority."""

  def __init__(self, max_transitions: int, seed=None, *, eps: float = 1e-6):
    super().__init__(max_transitions, seed=seed)
    self._eps = eps
    self._priorities = np.zeros(self.max_transitions, dtype=np.float64)
    self._generation = np.zeros(self.max_transitions, dtype=np.int64)
    # New rows must be sampled at least once to ever be scored -- seeding at
    # the running max priority (1.0 until anything is scored) is the
    # standard PER convention, not an arbitrary default.
    self._max_priority = 1.0

  def _on_write(self, idx: "np.ndarray") -> None:
    self._generation[idx] += 1
    self._priorities[idx] = self._max_priority

  def _sample_ids(self, idx: "np.ndarray") -> "np.ndarray":
    return self._generation[idx].astype(np.int64) * self.max_transitions + idx.astype(np.int64)

  def sample_prioritized(self, batch_size: int, *, alpha: float = 0.6, beta: float = 0.4) -> dict:
    """`alpha` shapes the sampling distribution toward high-priority rows
    (0 = uniform, matching plain `sample()`'s distribution exactly at
    alpha=0); `beta` shapes how strongly importance weights correct for
    that bias (1 = full correction). Returns the sampled fields plus
    `sample_ids` (feed back to `update_priorities` after scoring) and
    `weights` (normalized so the batch's largest weight is 1, the standard
    PER convention -- multiply a per-sample loss by this before reducing,
    never learn at the unweighted scale)."""
    if self._size == 0:
      raise ValueError("sample_prioritized() from an empty buffer")
    live = self._priorities[:self._size] + self._eps
    probs = live ** alpha
    total = probs.sum()
    probs = probs / total if total > 0 else np.full(self._size, 1.0 / self._size)

    idx = self.rng.choice(self._size, size=batch_size, replace=True, p=probs)
    weights = (self._size * probs[idx]) ** (-beta)
    weights = weights / weights.max()

    batch = {name: value[idx] for name, value in self._fields.items()}
    batch["sample_ids"] = self._sample_ids(idx)
    batch["weights"] = weights.astype(np.float32)
    return batch

  def snapshot(self) -> dict:
    """`ReplayBuffer.snapshot()` plus the priority/generation arrays and the
    running max priority -- restoring only the ring contents would silently
    reset every row to the seed priority and break `sample_ids` continuity.
    Truncated to `_size` for the same reason the ring fields are: everything
    past it has never been written."""
    base = super().snapshot()
    base["priorities"] = self._priorities[:self._size].copy()
    base["generation"] = self._generation[:self._size].copy()
    base["max_priority"] = self._max_priority
    return base

  def restore(self, snapshot: dict) -> None:
    super().restore(snapshot)
    priorities, generation = snapshot["priorities"], snapshot["generation"]
    self._priorities[:len(priorities)] = priorities
    self._generation[:len(generation)] = generation
    self._max_priority = snapshot["max_priority"]

  def update_priorities(self, sample_ids, td_errors) -> int:
    """Sets each row's priority to `|td_error| + eps`. Returns how many of
    `sample_ids` were actually applied -- fewer than `len(sample_ids)` is
    normal, not an error, whenever a row was overwritten between sampling
    it and scoring it."""
    sample_ids = np.asarray(sample_ids)
    td_errors = np.asarray(td_errors)
    idx = (sample_ids % self.max_transitions).astype(np.int64)
    current = self._sample_ids(idx)
    valid = current == sample_ids
    priorities = np.abs(td_errors[valid]) + self._eps
    self._priorities[idx[valid]] = priorities
    if priorities.size:
      self._max_priority = max(self._max_priority, float(priorities.max()))
    return int(valid.sum())


class MultiSeedReplay(ReplayBuffer):
  """One ReplayBuffer per seed behind a single rollout loop.

  `num_seeds: 1` is not a separate path; it is the size-1 instance of the same
  routing and sampling behavior the trainer uses at larger seed counts."""

  def __init__(self, num_seeds: int, num_envs: int, max_transitions: int, seed=None,
              *, buffer_cls: type = ReplayBuffer):
    super().__init__(max_transitions, seed=seed)
    self.num_seeds = num_seeds
    self.num_envs  = num_envs
    self.seeds = [buffer_cls(max_transitions, seed=child)
                  for child in np.random.SeedSequence(seed).spawn(num_seeds)]

  def add(self, **row) -> None:
    slot_ids = row.pop("slot_ids", None)
    if slot_ids is None:
      for i, buffer in enumerate(self.seeds):
        block = slice(i * self.num_envs, (i + 1) * self.num_envs)
        buffer.add(**{name: value[block] for name, value in row.items()})
      return

    slot_ids = np.asarray(slot_ids)
    for i, buffer in enumerate(self.seeds):
      lo, hi = i * self.num_envs, (i + 1) * self.num_envs
      mask = (slot_ids >= lo) & (slot_ids < hi)
      if not np.any(mask):
        continue
      buffer.add(**{name: value[mask] for name, value in row.items()})

  @property
  def n_transitions(self) -> int:
    return sum(buffer.n_transitions for buffer in self.seeds)

  def min_transitions(self) -> int:
    return min(buffer.n_transitions for buffer in self.seeds)

  def transitions_per_seed(self) -> list[int]:
    return [buffer.n_transitions for buffer in self.seeds]

  def sample(self, batch_size: int) -> dict:
    return self.sample_chunk(1, batch_size, leading=False)

  def sample_chunk(self, k: int, batch_size: int, *, leading: bool = True) -> dict:
    # Gather straight into one preallocated (k, seeds, batch, ...) array per field;
    # draws stay one per (step, seed) so the RNG stream is unchanged.
    out = {}
    for step in range(k):
      for i, buffer in enumerate(self.seeds):
        for key, value in buffer.sample_numpy(batch_size).items():
          if key not in out:
            out[key] = np.empty((k, self.num_seeds, *value.shape), value.dtype)
          out[key][step, i] = value
    if not leading:
      out = {key: value[0] for key, value in out.items()}
    return jax.device_put(out)
