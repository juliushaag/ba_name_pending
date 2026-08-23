"""Shared scaffolding for the component suite.

Every registered component is built from its real `config/components/` entry
-- presets expanded and at the smallest size, exactly as a run's first layers
resolve it -- narrowed by `TINY` -- so a required kwarg that gets renamed or dropped
from that file fails here, while the whole suite still runs in seconds. `TINY`
only replaces keys the entry already has, so it never injects a kwarg a class
does not declare (every component has a `**kwargs` catch-all that would swallow
one silently).
"""

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
import barl_jax.model_config as model_config
from barl_jax.env_stats import EnvStats
from barl_jax.networks.registry import available_components, get_component
from barl_jax.networks.schedule import Schedule
from barl_jax.networks.squash import Squash
from barl_jax.policy import Actor, Critic

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
        return_min=RETURN_MIN, return_max=RETURN_MAX,
        live={})
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
    base = model_config.tree().component_kwargs(name)
    if issubclass(get_component(name), Schedule):
        base.update(SCHEDULE_KWARGS)
    base.update(POLICY_SUPPLIED.get(name, {}))
    base.update({k: v for k, v in TINY.items() if k in base})
    base.update(extra)
    return base


def build(name: str, *, seed: int = 0, env_stats: EnvStats = None, **extra):
    stats = make_env_stats() if env_stats is None else env_stats
    return get_component(name)(stats, **tiny_kwargs(name, **extra), rngs=nnx.Rngs(seed))


@functools.lru_cache(maxsize=None)
def built(name: str):
    """One shared instance per class, for tests that only read from it.
    Anything that steps or mutates a component must call `build` instead."""
    return build(name)


def critic_action(critic, act):
    """`(a, level)` widened into the action an NC critic's inner net was built
    over; the identity for every other critic."""
    return critic.augment(act, 0) if hasattr(critic, "augment") else act


def target_expectation(critic, target):
    """E[target] whatever representation `bellman_target` returned it in --
    a scalar already, or a distribution over the critic's support."""
    if target.shape[-1] == 1:
        return target.squeeze(-1)
    return (target * critic.support[...]).sum(axis=-1)


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
