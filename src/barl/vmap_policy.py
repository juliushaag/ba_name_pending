import copy
import inspect
import types

import torch
from torch.func import stack_module_state, functional_call, vmap

from barl.policy import Policy


def _bind_forward(template: Policy, method_name: str):
  """functional_call always invokes module.forward(); this rebinds forward
  to whichever Policy method we actually want to vmap this call (train_step
  or inference_step) for the duration of that call."""
  method = getattr(type(template), method_name)
  template.forward = types.MethodType(method, template)


class VmapPolicyGroup:
  """
  N independently-initialized replicas of the same Policy class/hparams,
  trained (and used for rollout action selection) in parallel via
  torch.func.vmap instead of N sequential Python-level train_step/
  inference_step calls -- the same stack_module_state + functional_call +
  vmap pattern networks/ensemble.py already uses for a critic's N MLP
  heads, just applied to a whole Policy.

  A single optimizer over the stacked (N, ...) parameter tensors is exactly
  N independent per-replica optimizers: torch.optim updates (Adam, AdamW,
  SGD, ...) are elementwise per parameter, with no cross-element coupling,
  so grouping N replicas' same-named parameter into one (N, ...) leaf and
  running one optimizer step over it produces bit-identical results to N
  separate optimizers each stepping their own (unbatched) replica.

  Not every Policy is safe to vmap as written -- anything that calls
  a.requires_grad_(True); loss.backward(); return a.grad (populating .grad
  as a side effect) instead of a proper functional transform like
  torch.func.grad, or that reads a batched tensor's .item()/.data, breaks
  under vmap. QCritic/IQLCritic's score() and EMA.step() have already been
  made vmap-safe; anything else that breaks needs the same treatment.
  """

  def __init__(self, model_name: str, env, num_seeds: int, base_seed: int = 0, device: str = "cuda", **hparams):
    assert num_seeds >= 1

    self.model_name = model_name
    self.num_seeds  = num_seeds
    self.device     = device

    self.policies: list[Policy] = []
    for i in range(num_seeds):
      torch.manual_seed(base_seed + i)
      self.policies.append(Policy.make(model_name, env, **hparams).to(device))

    self.params, self.buffers = stack_module_state(self.policies)
    for p in self.params.values():
      p.requires_grad_(True)

    # meta-device template used by functional_call to describe the group's
    # shape -- forward() gets rebound per call (see _bind_forward) to
    # whichever underlying Policy method that call actually wants to run.
    # Must be a genuine deepcopy: .to("meta") mutates parameters in place,
    # so operating on policies[0] directly would corrupt the real replica.
    self.template = copy.deepcopy(self.policies[0]).to("meta")

  def train_step(self, batch: dict) -> tuple[dict, dict]:
    """`batch` values must have a leading (num_seeds, ...) dim -- replica i
    trains on batch[i]. Returns (losses, stats), each leaf shaped (num_seeds, ...)."""
    _bind_forward(self.template, "train_step")

    def call(p, b, batch):
      return functional_call(self.template, (p, b), (batch,))

    return vmap(call, in_dims=(0, 0, 0), randomness="different")(self.params, self.buffers, batch)

  def _call_lifecycle_hook(self, method_name: str):
    """Runs a no-args, no-return Policy hook (on_epoch_start/on_epoch_end)
    once per replica, vmapped -- e.g. DAM's on_epoch_end() copies the live
    diffusion net into its trust-region snapshot in place; functional_call
    substitutes self.params/self.buffers for the call's duration, and (like
    EMA.step() above) in-place copy_() on those substituted tensors persists
    back into the real stacked buffers, no explicit return value needed."""
    _bind_forward(self.template, method_name)

    def call(p, b):
      functional_call(self.template, (p, b), ())
      return torch.zeros(())

    # no_grad: every stacked param has requires_grad=True (needed for
    # train_step's backward pass), even ones a given Policy treats as a
    # frozen target (e.g. DAM's diffusion_act_old) -- in-place copy_() onto
    # a grad-requiring leaf outside no_grad raises, same reason EMA.step()
    # carries its own @torch.no_grad().
    with torch.no_grad():
      vmap(call, in_dims=(0, 0), randomness="different")(self.params, self.buffers)

  def on_epoch_start(self):
    self._call_lifecycle_hook("on_epoch_start")

  def on_epoch_end(self):
    self._call_lifecycle_hook("on_epoch_end")

  def inference_step(self, observation: torch.Tensor, **kwargs) -> tuple[torch.Tensor, dict]:
    """`observation` must have a leading (num_seeds, ...) dim -- replica i
    acts on observation[i]. Returns (action, extras), each leaf shaped
    (num_seeds, ...). `kwargs` (step_info / z / eps_greedy -- the second
    param name differs per Policy) is passed through unbatched, identical
    for every replica."""
    _bind_forward(self.template, "inference_step")

    def call(p, b, obs):
      return functional_call(self.template, (p, b), (obs,), kwargs)

    return vmap(call, in_dims=(0, 0, 0), randomness="different")(self.params, self.buffers, observation)

  def create_optimizer(self) -> dict[str, torch.optim.Optimizer]:
    """Reconstructs the same {key: Optimizer} shape a single Policy's
    create_optimizer() returns, but built over the stacked (N, ...)
    parameter tensors instead of one replica's own -- by matching parameter
    object identity against a reference replica's create_optimizer() to
    recover which optimizer key each named parameter belongs to, then
    replaying that optimizer's exact type + hyperparameters (lr, betas,
    weight_decay, maximize, ...) over the corresponding stacked tensors."""
    ref_policy = self.policies[0]
    ref_opts   = ref_policy.create_optimizer()

    id_to_key = {}
    for key, opt in ref_opts.items():
      for group in opt.param_groups:
        for p in group["params"]:
          id_to_key[id(p)] = key

    grouped: dict[str, list[torch.nn.Parameter]] = {}
    for name, p in ref_policy.named_parameters():
      key = id_to_key.get(id(p))
      if key is not None:
        grouped.setdefault(key, []).append(self.params[name])

    opts = {}
    for key, params in grouped.items():
      opt_cls = ref_opts[key].__class__
      # .defaults can carry internal derived flags (e.g. fused=True implies
      # decoupled_weight_decay) that aren't accepted back as __init__
      # kwargs -- keep only what the constructor actually declares.
      valid = set(inspect.signature(opt_cls.__init__).parameters) - {"self", "params"}
      kwargs = {k: v for k, v in ref_opts[key].defaults.items() if k in valid}
      opts[key] = opt_cls(params, **kwargs)
    return opts

  def sync_replicas(self):
    """Copies the current stacked params/buffers back into self.policies[i]
    -- needed before per-replica checkpointing (Policy.save()) or any other
    single-replica operation, since training only ever mutates the stacked
    tensors, not the original policies' own parameters.

    strict=False, not True: EMA.source is deliberately the SAME nn.Module
    object as whatever it wraps (e.g. critic.q_function), not a copy, so
    e.g. critic.q_ema.source.* legitimately has no separate entry in
    self.params/self.buffers (named_parameters()'s default dedup already
    dropped it upstream, in stack_module_state) even though state_dict()
    still lists that path -- copying into the aliased critic.q_function.*
    path already updates the same underlying storage, nothing is lost.
    """
    with torch.no_grad():
      for i, policy in enumerate(self.policies):
        state = {**{n: t[i] for n, t in self.params.items()},
                  **{n: t[i] for n, t in self.buffers.items()}}
        policy.load_state_dict(state, strict=False)
