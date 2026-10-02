"""Shared scaffolding for the component suite.

Every registered component is built from its typed `config_type` defaults --
narrowed by `TINY` -- so a required field that gets renamed or dropped fails
here, while the whole suite still runs in seconds. `TINY` only replaces fields
the schema already has, so it never injects a kwarg a class does not declare.
"""

import dataclasses
import functools
import os
import pathlib

# All of this has to happen before the first `import jax`, which is why it sits
# at the top of conftest rather than in pyproject's addopts.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MUJOCO_GL", "egl")

# The suite is ~200 tiny distinct computations, so it is dominated by XLA
# compilation rather than by anything it actually computes. Two settings make
# that affordable: don't optimize code whose runtime nobody measures, and keep
# the compiled executables on disk between runs. The first run after a change to
# a network fills the cache and costs ~15s; every run after it is under 10.
# `.pytest_cache/` is already gitignored.
_CACHE = pathlib.Path(__file__).resolve().parent.parent / ".pytest_cache" / "jax"
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(_CACHE))
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")
os.environ.setdefault("XLA_FLAGS", "--xla_backend_optimization_level=0")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

import barl_jax  # noqa: F401  -- import side effects populate the registry
from barl_jax.components import Actor, Critic
from barl_jax.critics import crossq_mlp, target_mlp
from barl_jax.env import EnvStats
from barl_jax.networks.schedule import Schedule
from barl_jax.networks.squash import Squash
from barl_jax.registration import available_components, get_component

OBS_DIM, ACT_DIM, BATCH = 5, 3, 4
RETURN_MIN, RETURN_MAX = -10.0, 10.0

# Width is the only thing shrunk: everything that changes what a component
# *means* -- the number of atoms or bins, the length of a reverse loop, the
# compute dtype -- stays at whatever config/components/ ships, so these are
# the real models tested narrow rather than different models tested small.
TINY = {"latent_dim": 8}

# A schedule's ladder is always spelled out by the slot that names it, so a
# standalone build has to supply one.
SCHEDULE_KWARGS = {"start": 1.0, "end": 0.1, "steps": 4}

def _members(base) -> list[str]:
    return sorted(n for n in available_components() if issubclass(get_component(n), base))


CRITICS = _members(Critic)
ACTORS = _members(Actor)
SQUASHES = _members(Squash)
SCHEDULES = _members(Schedule)
OTHERS = sorted(set(available_components()) - set(CRITICS + ACTORS + SQUASHES + SCHEDULES))
ALL = sorted(available_components())


def make_env_stats(**overrides) -> EnvStats:
    base = dict(
        obs_shape=(OBS_DIM,), obs_dim=OBS_DIM,
        act_discrete=False, act_dim=ACT_DIM,
        act_low=np.full(ACT_DIM, -1.0, np.float32),
        act_high=np.full(ACT_DIM, 1.0, np.float32),
        max_episode_steps=16,
        return_min=RETURN_MIN, return_max=RETURN_MAX)
    base.update(overrides)
    return EnvStats(**base)


# Kwargs a Policy derives and passes explicitly at its build_component() call
# site rather than reading from config (see e.g. AMDP.__init__'s erf_k). A
# standalone build has to stand in for the Policy and supply them.
POLICY_SUPPLIED = {
    "AutotuneRegulator": {"target": -1.0},
    "ErfSquash": {"erf_k": 1.0},
}


def tiny_kwargs(name: str, **extra) -> dict:
    # Legacy helper retained for introspection; new code builds typed configs
    # via `typed_config`. Returns flat overrides only.
    out = dict(TINY)
    out.update(extra)
    return out


def typed_config(name: str, **extra):
  """Materialize a component's `config_type` defaults with overrides.

  `TINY` narrows width where the schema has it; `extra` supplies test-specific
  fields (e.g. `flow_steps=2`). Schedule standalone builds supply their ladder
  via `SCHEDULE_KWARGS` defaults.
  """
  from barl_jax.config.serialization import merge
  cls = get_component(name)
  config_type = getattr(cls, "config_type", None)
  if config_type is None:
    return None
  values: dict = {}
  if issubclass(cls, Schedule):
    values.update(SCHEDULE_KWARGS)
  declared = {f.name for f in dataclasses.fields(config_type)}
  values.update({k: v for k, v in TINY.items() if k in declared})
  values.update(extra)
  unknown = set(values) - declared
  if unknown:
    raise KeyError(f"{name}: unknown config fields {sorted(unknown)}; "
                   f"available: {sorted(declared)}")
  return merge(config_type(), values)


def _merge_nested(config_type, values: dict):
  """Apply overrides that may include nested dataclass fields."""
  from barl_jax.config.serialization import merge
  return merge(config_type(), values)


def build(name: str, *, seed: int = 0, env_stats: EnvStats = None, **extra):
    """`extra` overrides the typed config, except for the keys
    `POLICY_SUPPLIED` names for this class -- those are real plumbing kwargs
    (a Policy passes them explicitly, never through config), so an override
    of one (e.g. `build("AutotuneRegulator", target=-1.0)`) routes there
    instead of into the typed config."""
    stats = make_env_stats() if env_stats is None else env_stats
    plumbing_keys = set(POLICY_SUPPLIED.get(name, {}))
    plumbing = {**POLICY_SUPPLIED.get(name, {}),
                **{k: v for k, v in extra.items() if k in plumbing_keys}}
    config_extra = {k: v for k, v in extra.items() if k not in plumbing_keys}
    config = typed_config(name, **config_extra)
    return get_component(name)(stats, config, rngs=nnx.Rngs(seed), **plumbing)


@functools.lru_cache(maxsize=None)
def built(name: str):
    """One shared instance per class, for tests that only read from it.
    Anything that steps or mutates a component must call `build` instead."""
    return build(name)


# The critic composition grid: every value head crossed with every
# bootstrap/normalization/reduction/objective choice `QCritic` accepts.
HEADS = ("DirectCritic", "EnsembleCritic", "HLGaussCritic", "C51Critic")
BOOTSTRAPS = {
    "crossq": {"target": {"kind": "none", "joint_forward": True}},
    "target": {"target": {"kind": "polyak", "joint_forward": False}},
    "xqc": {"target": {"kind": "polyak", "joint_forward": True}, "weight_norm": True},
}
NORMS = {
    "none": {"mlp": {"norm": "none", "norm_kwargs": {}}},
    "layer": {"mlp": {"norm": "layer", "norm_kwargs": target_mlp().norm_kwargs}},
    "batchrenorm": {"mlp": {"norm": "batchrenorm", "norm_kwargs": crossq_mlp().norm_kwargs}},
}
REDUCES = {r: {"ensemble": {"value_reduce": r, "target_reduce": r}} for r in ("mean", "min")}
OBJECTIVES = {"td": {}, "noise": {"noise": {"enabled": True}}}


def _deep_merge(*dicts) -> dict:
    out: dict = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _variant_grid() -> dict[str, tuple[str, dict]]:
    grid = {}
    for head in HEADS:
        for b, bv in BOOTSTRAPS.items():
            for n, nv in NORMS.items():
                if n == "batchrenorm" and not bv["target"]["joint_forward"]:
                    continue      # rejected by QCritic.config_problems
                for r, rv in REDUCES.items():
                    for o, ov in OBJECTIVES.items():
                        grid[f"{head}-{b}-{n}-{r}-{o}"] = (head, _deep_merge(bv, nv, rv, ov))
    return grid


VARIANTS = _variant_grid()


def build_variant(variant_id: str, **extra):
    head, overrides = VARIANTS[variant_id]
    return build(head, **_deep_merge(overrides, extra))


def target_expectation(critic, target):
    """E[target] whatever representation `bellman_target` returned it in --
    a scalar already, or a distribution over the critic's support."""
    if target.shape[-1] == 1:
        return target.squeeze(-1)
    return (target * critic.support_values()).sum(axis=-1)


def params_of(module):
    return jax.tree.leaves(nnx.state(module, nnx.Param))


def finite(x) -> bool:
    return bool(jnp.all(jnp.isfinite(x)))


@pytest.fixture(scope="session")
def key():
    return jax.random.PRNGKey(0)


@pytest.fixture(scope="session")
def obs():
    return jax.random.normal(jax.random.PRNGKey(1), (BATCH, OBS_DIM))


@pytest.fixture(scope="session")
def act():
    return jnp.tanh(jax.random.normal(jax.random.PRNGKey(2), (BATCH, ACT_DIM)))
