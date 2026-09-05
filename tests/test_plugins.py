"""Plugin discovery (barl_jax.registration): installed entry points under
`barl_jax.<group>` as a third way to name an implementation, after built-ins
and before/alongside `module:Class`. Discovery is scanned once and cached;
loading (importing the target) is separate and only happens when a name is
actually resolved."""

from importlib.metadata import EntryPoint

import pytest

import barl_jax.registration as registration
from barl_jax.components import Component


class FakeDist:
  def __init__(self, name):
    self.name = name


def ep(name, value, dist="fake-dist"):
  return EntryPoint(name=name, value=value, group="barl_jax.components")._for(FakeDist(dist))


@pytest.fixture(autouse=True)
def clear_plugin_cache(monkeypatch):
  # _discover_plugins caches per group for the process; each test gets a
  # clean cache so one test's fake entry points can't leak into another's.
  monkeypatch.setattr(registration, "_PLUGINS", {})
  yield
  monkeypatch.setattr(registration, "_PLUGINS", {})


class FakePlugin(Component):
  """A stand-in "plugin" class -- importable from this test module, which is
  exactly what a real plugin's entry point value would point at."""


class NotAComponent:
  pass


def test_discovery_is_cached_after_the_first_scan(monkeypatch):
  calls = []
  def fake_entry_points(*, group):
    calls.append(group)
    return [ep("Foo", "test_plugins:FakePlugin")] if group == "barl_jax.components" else []
  monkeypatch.setattr(registration._metadata, "entry_points", fake_entry_points)

  registration._discover_plugins("components")
  registration._discover_plugins("components")
  assert calls == ["barl_jax.components"]  # scanned once, not twice


def test_resolve_named_finds_a_plugin_by_name(monkeypatch):
  monkeypatch.setattr(registration._metadata, "entry_points",
                      lambda *, group: [ep("Foo", "test_plugins:FakePlugin")])
  cls = registration.resolve_named("components", "Foo", {}, Component)
  assert cls is FakePlugin


def test_resolve_named_returns_none_for_an_unknown_name(monkeypatch):
  monkeypatch.setattr(registration._metadata, "entry_points", lambda *, group: [])
  assert registration.resolve_named("components", "NoSuchThing", {}, Component) is None


def test_a_name_shared_by_two_distributions_is_a_deterministic_error(monkeypatch):
  monkeypatch.setattr(registration._metadata, "entry_points", lambda *, group: [
    ep("Foo", "test_plugins:FakePlugin", dist="pkg-a"),
    ep("Foo", "test_plugins:FakePlugin", dist="pkg-b"),
  ])
  with pytest.raises(ValueError, match="pkg-a.*pkg-b|pkg-b.*pkg-a"):
    registration.resolve_named("components", "Foo", {}, Component)


def test_a_name_shared_with_a_built_in_is_a_deterministic_error(monkeypatch):
  monkeypatch.setattr(registration._metadata, "entry_points",
                      lambda *, group: [ep("Built", "test_plugins:FakePlugin", dist="pkg-a")])
  builtins = {"Built": FakePlugin}
  with pytest.raises(ValueError, match="built-in"):
    registration.resolve_named("components", "Built", builtins, Component)


def test_a_plugin_not_subclassing_the_base_is_rejected(monkeypatch):
  monkeypatch.setattr(registration._metadata, "entry_points",
                      lambda *, group: [ep("Bad", "test_plugins:NotAComponent")])
  with pytest.raises(TypeError, match="does not implement"):
    registration.resolve_named("components", "Bad", {}, Component)


def test_a_plugin_whose_import_fails_reports_the_distribution(monkeypatch):
  monkeypatch.setattr(registration._metadata, "entry_points",
                      lambda *, group: [ep("Missing", "no.such.module:Thing", dist="pkg-a")])
  with pytest.raises(ImportError, match="pkg-a"):
    registration.resolve_named("components", "Missing", {}, Component)


def test_available_names_lists_plugins_without_loading_them(monkeypatch):
  loaded = []
  class TrackedEntryPoint(EntryPoint):
    def load(self):
      loaded.append(self.name)
      return super().load()

  fake = TrackedEntryPoint(name="Foo", value="test_plugins:FakePlugin",
                           group="barl_jax.components")._for(FakeDist("pkg-a"))
  monkeypatch.setattr(registration._metadata, "entry_points", lambda *, group: [fake])
  names = registration.available_names("components", {"Existing": FakePlugin})
  assert names == ["Existing", "Foo"]
  assert loaded == []  # listing never imports the plugin's target


def test_get_component_resolves_a_plugin_by_name(monkeypatch):
  from barl_jax.registration import get_component
  monkeypatch.setattr(registration._metadata, "entry_points",
                      lambda *, group: [ep("PluginComponent", "test_plugins:FakePlugin")]
                      if group == "barl_jax.components" else [])
  assert get_component("PluginComponent") is FakePlugin


def test_module_class_path_bypasses_plugin_discovery_entirely(monkeypatch):
  """module:Class stays a direct, explicit path -- unaffected by whatever
  plugins are or are not installed."""
  from barl_jax.registration import get_component
  def fail(*, group):
    raise AssertionError("module:Class must not trigger plugin discovery")
  monkeypatch.setattr(registration._metadata, "entry_points", fail)
  assert get_component("test_plugins:FakePlugin") is FakePlugin
