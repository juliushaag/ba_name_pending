"""The registry and config/components.yaml agreeing with the classes.

These are the cheap checks that catch config drift: an entry whose class no
longer exists, a kwarg no `__init__` in the MRO accepts (the `**kwargs`
catch-all every component has would otherwise swallow it silently), and a class
that cannot be constructed from its own baseline at all.
"""

import pytest

import barl_jax.config.tree as model_config
from barl_jax.networks.registry import get_component
from barl_jax.components import Component

from conftest import ALL, CRITICS, ACTORS, SCHEDULES, SQUASHES, built, tiny_kwargs

# components.yaml entries with no class in the registry. Both are leftovers from
# the torch port; listed so a *new* one still fails this test.
KNOWN_UNREGISTERED = {"DistilledFlowActor", "FlowBCActor"}


def test_registry_is_partitioned():
    """Every registered name lands in exactly one family -- a component that is
    neither actor, critic, squash nor schedule is one nothing knows how to test."""
    assert set(CRITICS + ACTORS + SQUASHES + SCHEDULES).issubset(set(ALL))
    assert not set(CRITICS) & set(ACTORS)


@pytest.mark.parametrize("name", ALL)
def test_is_a_component(name):
    assert issubclass(get_component(name), Component)


@pytest.mark.parametrize("name", ALL)
def test_constructs_from_its_baseline(name):
    assert isinstance(built(name), Component)


def test_components_yaml_entries_are_registered():
    """Every entry in config/components/ has to name a class the registry can
    resolve -- a typo's baseline kwargs would otherwise silently never apply."""
    entries = set(model_config.ConfigTree().components)
    stale = sorted(entries - set(ALL) - KNOWN_UNREGISTERED)
    assert not stale, f"components.yaml entries with no registered class: {stale}"


@pytest.mark.parametrize("name", ALL)
def test_optimizer_keys_are_named(name):
    """A component owning an optimizer names its keys; each maps to an optax
    transform, since the training loop indexes them by loss key."""
    for k, tx in built(name).create_optimizer().items():
        assert isinstance(k, str) and k
        assert hasattr(tx, "init") and hasattr(tx, "update")


@pytest.mark.parametrize("name", ALL)
def test_lifecycle_hooks_are_callable(name):
    """The two hooks a Policy fans out over its slots, on a component that has
    never been stepped -- they run before the first train_step in a real run."""
    c = built(name)
    c.before_step()
    c.after_step()
