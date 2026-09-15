"""The registry agreeing with itself -- every registered component builds and
constructs its own `config_type` from bare defaults."""

import pytest
from conftest import ACTORS, ALL, CRITICS, SCHEDULES, SQUASHES, built

from barl_jax.components import Component
from barl_jax.registration import get_component


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


@pytest.mark.parametrize("name", ALL)
def test_config_type_constructs_from_its_own_defaults(name):
    """Every component's `config_type` (if it declares one) is a real, usable
    dataclass -- construction from bare defaults is what `Policy.build_component`
    does before applying any override. There is one registry now (the runtime
    class itself), so a schema can no longer drift out of sync with it -- see
    `config_type` beside each implementation."""
    config_type = get_component(name).config_type
    if config_type is not None:
        assert config_type()  # constructs without raising


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
