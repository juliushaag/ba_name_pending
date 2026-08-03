"""Per-step metric sink with a fixed key schema.

Keys are `<owner>/<quantity>/<reduction>`, owner-first so wandb's sidebar reads
as one section per component (`critic/`, `actor/`, `entropy/`) plus one per
algorithm (`amdp/`, `ppo/`). Policy.build_component hands each slot a
`child(slot)` view, so a component writes `dist("q", pred)` without knowing its
own slot name.

Every family emits exactly one PRIMARY key and puts its expansion on DEBUG --
the headline stays dense enough to watch a run, the wide part is paced by
`debug_every`. Families early-return when their tier is inactive, so call sites
need no guard.

All values are kept as 0-dim tensors until `drain()`, which moves the whole step
in one `.cpu()` -- one GPU sync per step, not one per key."""

import torch

PRIMARY = "primary"
DEBUG   = "debug"


class Metrics:

  def __init__(self):
    self._values: dict[str, dict[str, torch.Tensor]] = {PRIMARY: {}, DEBUG: {}}
    self._active: set[str] = set()

  # ------------------------------------------------------------------
  # Lifecycle
  # ------------------------------------------------------------------

  def arm(self, *tiers: str):
    """Sets which tiers record this step; everything else becomes a no-op."""
    self._values = {PRIMARY: {}, DEBUG: {}}
    self._active = set(tiers)

  def active(self, tier: str = PRIMARY) -> bool:
    return tier in self._active

  def drain(self) -> tuple[dict, dict]:
    out = []
    for tier in (PRIMARY, DEBUG):
      # Grouped by device so each group crosses in a single stack+.cpu() -- one
      # sync per device per step, rather than one .item() per key.
      by_device: dict[torch.device, list[str]] = {}
      for key, value in self._values[tier].items():
        by_device.setdefault(value.device, []).append(key)

      drained = {}
      for device, keys in by_device.items():
        flat = torch.stack([self._values[tier][k] for k in keys]).float().cpu().tolist()
        drained.update(zip(keys, flat))
      out.append(drained)

    self._values = {PRIMARY: {}, DEBUG: {}}
    return out[0], out[1]

  def child(self, namespace: str) -> "Metrics":
    """A view writing under `namespace/`, sharing this recorder's buffers and
    armed tiers -- so a component never spells out its own prefix."""
    return _Namespaced(self, namespace)

  # ------------------------------------------------------------------
  # Recording
  # ------------------------------------------------------------------

  def _put(self, tier: str, key: str, value):
    if tier not in self._active:
      return
    if not torch.is_tensor(value):
      value = torch.as_tensor(value)
    self._values[tier][key] = value.detach().reshape(-1)[0] if value.numel() > 1 else value.detach().reshape(())

  def scalar(self, name: str, value, tier: str = PRIMARY):
    self._put(tier, name, value)

  # ------------------------------------------------------------------
  # Families
  # ------------------------------------------------------------------

  def dist(self, name: str, x: torch.Tensor):
    """Any per-sample scalar quantity: q values, advantages, rewards, entropies."""
    if not self._active:
      return
    x = x.detach().float()
    self._put(PRIMARY, f"{name}/mean", x.mean())
    self._put(PRIMARY, f"{name}/std", x.std() if x.numel() > 1 else x.new_zeros(()))
    self._put(PRIMARY, f"{name}/min", x.min())
    self._put(PRIMARY, f"{name}/max", x.max())

    l = x.min() + 1e-3
    u = x.max() - 1e-3
        
    self._put(PRIMARY, f"{name}/edge_sat", torch.logical_and(x > u, x < l).float().mean())
      
  def vec(self, name: str, x: torch.Tensor, limit: float = None):
    """Per-sample vectors: actions, score fields, flow velocities."""
    if not self._active:
      return
    x = x.detach().float()
    norms = x.norm(dim=-1)
    self._put(PRIMARY, f"{name}/norm_mean", norms.mean())
    self._put(PRIMARY, f"{name}/dim_std_mean", x.std(dim=0).mean() if x.shape[0] > 1 else x.new_zeros(()))

  def ensemble(self, name: str, raw: torch.Tensor, rho: float = 0.0):
    """Epistemic disagreement across an ensemble critic's `(n_models, B, 1)`
    heads -- distinct from a distributional critic's return spread, which goes
    through dist()."""
    if not self._active:
      return
    raw = raw.detach().float()
    mean, std = raw.mean(dim=0), raw.std(dim=0)
    self._put(PRIMARY, f"{name}/disagreement", std.mean())
    self._put(PRIMARY, f"{name}/pessimism_gap", (rho * std).mean())
    
  def ema_gap(self, name: str, ema):
    """Target-net drift for an EMA/Shadow pair: ||src - tgt|| / ||src||.
    Accumulated per-tensor rather than over a flat cat() -- this runs on the
    primary tier, and an ensemble critic's parameters are large enough that the
    copy would dominate the reduction."""
    if not self._active:
      return
    with torch.no_grad():
      num = den = None
      for p, q in zip(ema.source.parameters(), ema.target.parameters()):
        d = (p.detach().float() - q.detach().float()).pow(2).sum()
        s = p.detach().float().pow(2).sum()
        num, den = (d if num is None else num + d), (s if den is None else den + s)
      if num is None:
        return
      self._put(PRIMARY, f"{name}/gap", (num / den.clamp_min(1e-24)).sqrt())

class _Namespaced(Metrics):
  """A `child()` view: prefixes keys, delegates buffers and armed tiers to the root."""

  def __init__(self, root: Metrics, namespace: str):
    self._root = root
    self._ns   = namespace

  @property
  def _active(self):
    return self._root._active

  @_active.setter
  def _active(self, value):
    self._root._active = value

  @property
  def _values(self):
    return self._root._values

  @_values.setter
  def _values(self, value):
    self._root._values = value

  def _put(self, tier: str, key: str, value):
    self._root._put(tier, f"{self._ns}/{key}", value)

  def child(self, namespace: str) -> "Metrics":
    return _Namespaced(self._root, f"{self._ns}/{namespace}")

  def arm(self, *tiers: str):
    raise RuntimeError("arm() the root Metrics, not a child view")

  def drain(self):
    raise RuntimeError("drain() the root Metrics, not a child view")
