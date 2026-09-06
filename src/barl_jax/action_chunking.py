import numpy as np


def normalize_action_chunk(action, *, chunk_size: int, act_dim: int, xp=np):
  """Canonical action-plan shape: `(batch, chunk_size, act_dim)`.

  The default case is still a chunk -- a singleton one. Existing actors that
  still return `(batch, act_dim)` therefore stay valid exactly when
  `chunk_size == 1`, and nowhere else."""
  action = xp.asarray(action)
  if action.ndim == 2:
    if chunk_size != 1:
      raise ValueError(
        f"actor returned shape {tuple(action.shape)} but chunk_size={chunk_size}; "
        "a rank-2 action is only valid for singleton chunks")
    action = action[:, None, :]
  if action.ndim != 3:
    raise ValueError(
      f"actor output must have shape (batch, chunk_size, act_dim); got rank {action.ndim} "
      f"shape {tuple(action.shape)}")
  if action.shape[1] != chunk_size or action.shape[2] != act_dim:
    raise ValueError(
      f"actor output shape {tuple(action.shape)} does not match "
      f"(batch, {chunk_size}, {act_dim})")
  return action


class ActionChunkExecutor:
  """Per-slot cached action plans, consumed one primitive action per env step."""

  def __init__(self, infer, *, chunk_size: int, act_dim: int, n_envs: int):
    self._infer = infer
    self.chunk_size = int(chunk_size)
    self.act_dim = int(act_dim)
    self.n_envs = int(n_envs)
    self._chunks = np.zeros((n_envs, chunk_size, act_dim), np.float32)
    self._cursor = np.zeros(n_envs, np.int32)
    self._active = np.zeros(n_envs, bool)
    self._extras = {}
    self._slot_refills = np.zeros(n_envs, np.int32)
    self._slot_wasted = np.zeros(n_envs, np.int32)
    self.refills = 0
    self.wasted = 0
    self._last_refilled = np.zeros(n_envs, bool)

  def restore(self, snapshot: dict | None) -> None:
    if not snapshot:
      return
    self._chunks = np.array(snapshot["chunks"], copy=True)
    self._cursor = np.array(snapshot["cursor"], copy=True)
    self._active = np.array(snapshot["active"], copy=True)
    self._extras = {k: np.array(v, copy=True) for k, v in snapshot.get("extras", {}).items()}
    self._slot_refills = np.array(snapshot.get("slot_refills", self._slot_refills), copy=True)
    self._slot_wasted = np.array(snapshot.get("slot_wasted", self._slot_wasted), copy=True)
    self.refills = int(snapshot.get("refills", 0))
    self.wasted = int(snapshot.get("wasted", 0))
    self._last_refilled = np.zeros(self.n_envs, bool)

  def snapshot(self) -> dict:
    return dict(
      chunks=np.array(self._chunks, copy=True),
      cursor=np.array(self._cursor, copy=True),
      active=np.array(self._active, copy=True),
      extras={k: np.array(v, copy=True) for k, v in self._extras.items()},
      slot_refills=np.array(self._slot_refills, copy=True),
      slot_wasted=np.array(self._slot_wasted, copy=True),
      refills=int(self.refills),
      wasted=int(self.wasted),
    )

  def current(self, obs, key, *, to_device, live=None, full_batch=False) -> tuple[np.ndarray, dict]:
    live = np.ones(self.n_envs, bool) if live is None else np.asarray(live, dtype=bool)
    slots = np.flatnonzero(live & ~self._active)
    self._last_refilled = np.zeros(self.n_envs, bool)
    if len(slots):
      # Vmapped seed policies require every seed slot, even after some
      # episodes finish. Only install plans for slots that need a refill.
      chunk, extras = self._infer(to_device(obs if full_batch else obs[slots]), key)
      if full_batch:
        chunk = np.asarray(chunk)[slots]
        extras = {name: np.asarray(value)[slots] for name, value in extras.items()}
      chunk = normalize_action_chunk(chunk, chunk_size=self.chunk_size,
                                     act_dim=self.act_dim, xp=np).astype(np.float32, copy=False)
      self._chunks[slots] = chunk
      self._cursor[slots] = 0
      self._active[slots] = True
      self._last_refilled[slots] = True
      self.refills += len(slots)
      self._slot_refills[slots] += 1

      extras = {k: np.asarray(v) for k, v in extras.items()}
      for name in set(self._extras) - set(extras):
        self._extras[name][slots] = 0
      for name, value in extras.items():
        if value.shape[0] != len(slots):
          raise ValueError(
            f"step extra {name!r} has leading shape {value.shape[0]} for {len(slots)} refilled slots")
        if name not in self._extras:
          self._extras[name] = np.zeros((self.n_envs, *value.shape[1:]), value.dtype)
        self._extras[name][slots] = value

    action = self._chunks[np.arange(self.n_envs), self._cursor]
    action = np.where(live[:, None], action, 0.0)
    extras = {k: np.where(live.reshape(self.n_envs, *([1] * (v.ndim - 1))), v, 0)
              for k, v in self._extras.items()}
    self._last_plan = self._chunks.copy()
    return action, extras

  def plan(self) -> np.ndarray:
    """The full plan each slot is currently executing, `(n_envs, chunk, dim)`.
    Read after `current()` and before `advance()` it is the plan the returned
    primitive came from -- what the buffer stores so the critic prices whole
    plans rather than single steps."""
    return np.array(self._last_plan if hasattr(self, "_last_plan") else self._chunks,
                    copy=True)

  def advance(self, done) -> None:
    done = np.asarray(done, dtype=bool)
    live = ~done
    finished = done | (live & self._active & (self._cursor == self.chunk_size - 1))
    tail = np.where(done & self._active, self.chunk_size - self._cursor - 1, 0)
    self.wasted += int(tail.sum())
    self._slot_wasted += tail

    self._cursor[live] += 1
    exhausted = live & (self._cursor >= self.chunk_size)
    self._active[done | exhausted] = False
    self._cursor[done | exhausted] = 0
    return finished

  def last_refilled(self) -> np.ndarray:
    return np.array(self._last_refilled, copy=True)

  def slot_refills(self) -> np.ndarray:
    return np.array(self._slot_refills, copy=True)

  def slot_wasted(self) -> np.ndarray:
    return np.array(self._slot_wasted, copy=True)
