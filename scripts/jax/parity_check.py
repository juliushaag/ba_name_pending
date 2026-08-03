"""Numeric parity between the torch `barl` implementation and the JAX
`barl_jax` port.

Loads torch weights into the corresponding nnx module and compares outputs on a
fixed batch. This is the gate every phase of the port is checked against: an
architecture that resolves from the same YAML and produces the same numbers is
the actual definition of "ported", and a learning-curve comparison is only
meaningful once these pass.

Both packages are imported in one process; torch and JAX coexist fine as long as
JAX has already grabbed its GPU memory fraction (or both stay on CPU, which is
what this script forces -- it compares math, not kernels).

  uv run python scripts/jax/parity_check.py
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # parity is about math, not kernels
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import warnings

warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx

TOL = 1e-5

torch.manual_seed(0)  # the torch modules below are randomly initialized -- keep it reproducible

_results: list[tuple[str, float, bool]] = []


def compare(name: str, torch_out, jax_out, tol: float = TOL, scale: float = None):
  """Scaled tolerance: `tol` bounds max|diff| / scale, where `scale` defaults to
  max(1, max|expected|).
  A flat 1e-5 absolute bound is meaningless for a quantity of magnitude ~1000 --
  float32 eps alone puts it at ~1.2e-4 -- so comparing at that scale would
  either fail on healthy code or have to be loosened to the point of catching
  nothing on small quantities.

  Shapes are compared after broadcasting: torch's Normal broadcasts a
  state-independent log_std up to the batch, the JAX version leaves it (act_dim,)
  and relies on the same broadcast at use, which is the same function."""
  a = torch_out.detach().cpu().numpy() if torch.is_tensor(torch_out) else np.asarray(torch_out)
  b = np.asarray(jax_out)
  try:
    a, b = np.broadcast_arrays(a, b)
  except ValueError:
    _results.append((f"{name}  [shape {a.shape} vs {b.shape}]", float("inf"), False))
    return
  # An explicit `scale` is for quantities computed by cancellation, where the
  # result is small but the summands are not -- judging those against their own
  # output magnitude would demand precision float32 never had.
  scale = max(1.0, float(np.max(np.abs(a)))) if scale is None else scale
  diff = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) / scale
  _results.append((name, diff, diff <= tol))


# ----------------------------------------------------------------------
# Weight mapping
# ----------------------------------------------------------------------


def copy_linear(t_lin, j_lin):
  """torch Linear stores (out, in); flax stores (in, out)."""
  j_lin.kernel[...] = jnp.asarray(t_lin.weight.detach().cpu().numpy().T)
  j_lin.bias[...]   = jnp.asarray(t_lin.bias.detach().cpu().numpy())


def copy_layernorm(t_ln, j_ln):
  j_ln.scale[...] = jnp.asarray(t_ln.weight.detach().cpu().numpy())
  j_ln.bias[...]  = jnp.asarray(t_ln.bias.detach().cpu().numpy())


def copy_mlp(t_mlp, j_mlp):
  """The torch MLP is one flat nn.Sequential; the JAX one names its pieces
  (stem / blocks / norms / head). Map positionally: first Linear is the stem,
  last is the head, the rest are hidden blocks in order."""
  from torch import nn
  linears = [m for m in t_mlp.layers if isinstance(m, nn.Linear)]
  norms   = [m for m in t_mlp.layers if isinstance(m, nn.LayerNorm)]

  copy_linear(linears[0], j_mlp.stem)
  copy_linear(linears[-1], j_mlp.head)
  for t_lin, j_block in zip(linears[1:-1], j_mlp.blocks):
    copy_linear(t_lin, j_block)

  j_norms = [n for n in j_mlp.norms if n is not None]
  for t_ln, j_ln in zip(norms, j_norms):
    copy_layernorm(t_ln, j_ln)


# ----------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------


def check_mlp():
  from barl.networks.mlp import MLP as TMLP
  from barl_jax.networks.mlp import MLP as JMLP

  rng = np.random.default_rng(0)
  for layernorm in (True, False):
    for residual in (False,):
      t = TMLP(11, 5, 32, 3, layernorm=layernorm, residual=residual).eval()
      j = JMLP(11, 5, 32, 3, layernorm=layernorm, residual=residual, rngs=nnx.Rngs(0))
      copy_mlp(t, j)

      x = rng.normal(size=(7, 11)).astype(np.float32)
      with torch.no_grad():
        to = t(torch.from_numpy(x))
      jo = j(jnp.asarray(x))
      compare(f"MLP(layernorm={layernorm})", to, jo)

  # multi-arg concat path
  t = TMLP(11 + 4, 5, 32, 2, layernorm=False).eval()
  j = JMLP(11 + 4, 5, 32, 2, layernorm=False, rngs=nnx.Rngs(0))
  copy_mlp(t, j)
  a = rng.normal(size=(7, 11)).astype(np.float32)
  b = rng.normal(size=(7, 4)).astype(np.float32)
  with torch.no_grad():
    to = t(torch.from_numpy(a), torch.from_numpy(b))
  compare("MLP(*args concat)", to, j(jnp.asarray(a), jnp.asarray(b)))


def check_squash():
  from barl.networks.squash import ErfSquash as TErf, TanhSquash as TTanh
  from barl_jax.networks.squash import ErfSquash as JErf, TanhSquash as JTanh

  rng = np.random.default_rng(1)
  x = rng.normal(size=(64, 6)).astype(np.float32) * 2.0
  y = np.clip(rng.uniform(-0.98, 0.98, size=(64, 6)).astype(np.float32), -0.98, 0.98)

  tt, jt = TTanh(), JTanh()
  with torch.no_grad():
    compare("TanhSquash.forward", tt(torch.from_numpy(x)), jt(jnp.asarray(x)))
    compare("TanhSquash.inverse", tt.inverse(torch.from_numpy(y)), jt.inverse(jnp.asarray(y)))
    compare("TanhSquash.log_abs_det_jacobian",
            tt.log_abs_det_jacobian(torch.from_numpy(x)),
            jt.log_abs_det_jacobian(jnp.asarray(x)))

  k = 0.3149204930294064
  te, je = TErf(erf_k=k), JErf(erf_k=k)
  with torch.no_grad():
    compare("ErfSquash.forward", te(torch.from_numpy(x)), je(jnp.asarray(x)))
    compare("ErfSquash.inverse", te.inverse(torch.from_numpy(y)), je.inverse(jnp.asarray(y)))
    compare("ErfSquash.log_abs_det_jacobian",
            te.log_abs_det_jacobian(torch.from_numpy(x)),
            je.log_abs_det_jacobian(jnp.asarray(x)))


def check_sinusoidal():
  from barl.networks.flow import sinusoidal_emb as t_emb
  from barl_jax.networks.flow import sinusoidal_emb as j_emb

  rng = np.random.default_rng(2)
  t = rng.uniform(0, 1, size=(9, 1)).astype(np.float32)
  with torch.no_grad():
    to = t_emb(torch.from_numpy(t), 8)
  compare("sinusoidal_emb", to, j_emb(jnp.asarray(t), 8))


def check_hlgauss():
  from barl.env_stats import EnvStats as TEnvStats
  from barl.networks.critic import HLGaussCritic as TCritic
  from barl_jax.env_stats import EnvStats as JEnvStats
  from barl_jax.networks.critic import HLGaussCritic as JCritic

  kw = dict(hidden_dim=32, layers=2, num_bins=51, sigma_to_bin_ratio=0.75,
            polyak_tau=0.005, gamma=0.99, v_min=-200.0, v_max=200.0)
  stats = dict(obs_dim=11, act_dim=3, return_min=-200.0, return_max=200.0)

  t = TCritic(TEnvStats(stats), **kw).eval()
  j = JCritic(JEnvStats(stats), rngs=nnx.Rngs(0), **kw)
  copy_mlp(t.q_function, j.q_function.net)

  rng = np.random.default_rng(3)
  v = rng.uniform(-250, 250, size=(32,)).astype(np.float32)
  with torch.no_grad():
    th = t._to_histogram(torch.from_numpy(v))
  compare("HLGauss._to_histogram", th, j._to_histogram(jnp.asarray(v)))

  s = rng.normal(size=(32, 11)).astype(np.float32)
  a = rng.uniform(-1, 1, size=(32, 3)).astype(np.float32)
  with torch.no_grad():
    te = t.expectation(torch.from_numpy(s), torch.from_numpy(a))
  # sum over 51 bins of probs * centres: the terms are O(v_max) even though the
  # result is near 0 for untrained logits, so v_max is the honest scale here.
  compare("HLGauss.expectation", te, j.expectation(jnp.asarray(s), jnp.asarray(a)),
          scale=200.0)

  # cross-entropy loss against the same target
  target = np.asarray(j._to_histogram(jnp.asarray(v)))
  t_losses, j_losses = {}, {}
  with torch.no_grad():
    t.loss(torch.from_numpy(s), torch.from_numpy(a), torch.from_numpy(target), t_losses)
  j.loss(jnp.asarray(s), jnp.asarray(a), jnp.asarray(target), j_losses)
  compare("HLGauss.loss", t_losses["q"], j_losses["q"])


def check_ensemble():
  from barl.env_stats import EnvStats as TEnvStats
  from barl.networks.critic import MeanStdEnsembleCritic as TCritic
  from barl_jax.env_stats import EnvStats as JEnvStats
  from barl_jax.networks.critic import MeanStdEnsembleCritic as JCritic

  kw = dict(hidden_dim=32, layers=2, n_models=5, rho=2.0, polyak_tau=0.005, gamma=0.99)
  stats = dict(obs_dim=11, act_dim=3)

  j = JCritic(JEnvStats(stats), rngs=nnx.Rngs(0), **kw)

  # The load-bearing ensemble property: members must differ at init, or
  # mean - rho*std silently degenerates to a plain mean.
  flat = nnx.to_flat_state(nnx.state(j.q_function.ensemble.members, nnx.Param))
  kernels = [np.asarray(v[...]) for path, v in flat if path[-1] == "kernel"]
  spread = float(min(k.std(0).mean() for k in kernels))
  _results.append(("Ensemble members differ at init", spread, spread > 1e-4))

  # value() reduction against a hand-computed mean - rho*std over the same heads
  rng = np.random.default_rng(4)
  s = rng.normal(size=(16, 11)).astype(np.float32)
  a = rng.uniform(-1, 1, size=(16, 3)).astype(np.float32)
  raw = np.asarray(j.raw(jnp.asarray(s), jnp.asarray(a)))
  expect = raw.mean(0) - 2.0 * np.sqrt(np.maximum(raw.var(0), 1e-8))
  compare("MeanStdEnsemble.value reduction", expect, j.value(jnp.asarray(s), jnp.asarray(a)))


def check_score_actor():
  from barl.env_stats import EnvStats as TEnvStats
  from barl.networks.actor import ScoreContinuousActor as TActor
  from barl_jax.env_stats import EnvStats as JActorStats
  from barl_jax.networks.actor import ScoreContinuousActor as JActor

  kw = dict(latent_dim=32, layers=2, max_steps=16, sigma_min=2.0, sigma_max=2.5,
            tau_min=0.0, tau_max=1.0)
  stats = dict(obs_dim=11, act_dim=3)
  t = TActor(TEnvStats(stats), **kw).eval()
  j = JActor(JActorStats(stats), rngs=nnx.Rngs(0), **kw)

  tau = np.linspace(0.0, 1.0, 17).astype(np.float32)
  with torch.no_grad():
    compare("ScoreActor.sigma_fn", t.sigma_fn(torch.from_numpy(tau)), j.sigma_fn(jnp.asarray(tau)))
    compare("ScoreActor.sigma_sq_int_fn",
            t.sigma_sq_int_fn(torch.from_numpy(tau)), j.sigma_sq_int_fn(jnp.asarray(tau)))
  compare("ScoreActor.step_var", t.step_var, j.step_var[...])
  _results.append(("ScoreActor.terminal_var",
                   abs(t.terminal_var - j.terminal_var), abs(t.terminal_var - j.terminal_var) <= TOL))

  # zero-init head: u ~ 0 at construction
  head_max = float(np.abs(np.asarray(j.diffusion_act.head.kernel[...])).max())
  _results.append(("ScoreActor head zero-init", head_max, head_max == 0.0))


def check_stochastic_actor():
  from barl.env_stats import EnvStats as TEnvStats
  from barl.networks.actor import StochasticActor as TActor
  from barl_jax.env_stats import EnvStats as JEnvStats
  from barl_jax.networks.actor import StochasticActor as JActor

  kw = dict(hidden_dim=32, layers=2, squash_cls="TanhSquash")
  stats = dict(obs_dim=11, act_dim=3)
  t = TActor(TEnvStats(stats), **kw).eval()
  j = JActor(JEnvStats(stats), rngs=nnx.Rngs(0), **kw)
  copy_mlp(t.net, j.net)
  j.log_std[...] = jnp.asarray(t.log_std.detach().cpu().numpy())

  rng = np.random.default_rng(5)
  s = rng.normal(size=(16, 11)).astype(np.float32)
  a = np.clip(rng.uniform(-0.9, 0.9, size=(16, 3)), -0.9, 0.9).astype(np.float32)

  with torch.no_grad():
    td = t.dist(torch.from_numpy(s))
    tm, ts = td.mean, td.stddev
    tlp = t.log_prob(td, torch.from_numpy(a))
  jm, js = j.dist(jnp.asarray(s))
  compare("StochasticActor.dist.mean", tm, jm)
  compare("StochasticActor.dist.std", ts, js)
  compare("StochasticActor.log_prob", tlp, j.log_prob((jm, js), jnp.asarray(a)))

  with torch.no_grad():
    tmode = t.mode(torch.from_numpy(s))
  compare("StochasticActor.mode", tmode, j.mode(jnp.asarray(s)))


def check_metric_packing():
  """The packed-metrics round trip, at one and several seeds.

  Not a torch comparison, but it belongs with the other invariants: the packing
  that makes draining one transfer instead of ~47 depends on both sides ordering
  keys the same way, and when they disagreed the result was not an error but
  silently wrong numbers in every log line."""
  import barl_jax  # noqa: F401
  from barl_jax.metrics import DEBUG, PRIMARY, to_host
  from barl_jax.multiseed import StackedPolicy
  from barl_jax.trainer import seeded_step, setup

  cfg = dict(model="SAC", env="HalfCheetah", num_envs=4, buffer_size=20000,
             batch_size=32, total_env_steps=10000, utd_ratio=1.0, lr_warmup_steps=10,
             seed=0, random_steps=0, eval_runs=1, eval_videos=0)
  for num_seeds in (1, 3):
    run = setup(cfg, num_seeds)
    run.buffer.rollout(StackedPolicy(run.graphdef, run.state, num_seeds,
                                     run.num_envs, run.reference),
                       env=run.rollout_env, n_steps=40, seed=0)
    batch = run.buffer.sample(cfg["batch_size"])
    _, _, primary, _ = seeded_step(run.graphdef, run.state, batch, (PRIMARY, DEBUG))

    packed = to_host(primary)
    per_key = {k: np.asarray(jax.device_get(v)) for k, v in primary[0].items()}
    worst = max((float(np.max(np.abs(packed[k] - per_key[k]))) for k in per_key), default=0.0)
    _results.append((f"metric packing == per-key transfer (seeds={num_seeds})", worst, worst == 0.0))
    run.env.close(); run.rollout_env.close()


def main():
  for check in (check_mlp, check_squash, check_sinusoidal, check_hlgauss,
                check_ensemble, check_score_actor, check_stochastic_actor,
                check_metric_packing):
    try:
      check()
    except Exception as e:
      _results.append((f"{check.__name__} RAISED: {type(e).__name__}: {e}", float("inf"), False))

  width = max(len(n) for n, _, _ in _results)
  print(f"\n{'check':<{width}}  {'rel|diff|':>12}   result")
  print("-" * (width + 26))
  for name, diff, ok in _results:
    print(f"{name:<{width}}  {diff:>12.3e}   {'ok' if ok else 'FAIL'}")

  failed = [n for n, _, ok in _results if not ok]
  print()
  if failed:
    print(f"{len(failed)}/{len(_results)} checks FAILED")
    raise SystemExit(1)
  print(f"all {len(_results)} checks passed (tol {TOL:g})")


if __name__ == "__main__":
  main()
