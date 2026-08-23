"""The config tree's guarantees, as tests rather than as convention.

Three things this suite exists to hold: a size means one thing everywhere, no
tunable can hide in Python, and a saved run or checkpoint replays exactly no
matter what `config/` says afterwards.
"""

import json
import shutil

import pytest
from flax import nnx

import barl_jax.model_config as model_config
from barl_jax.config import INJECTED_KEYS, load_config, run_keys
from barl_jax.config_tree import CONFIG_DIR, TRAINING_KEYS, ConfigTree
from barl_jax.networks.registry import available_components, get_component

MODELS = sorted(ConfigTree().available_models())


@pytest.fixture
def tree():
    return ConfigTree()


# --- no defaults in the code ----------------------------------------------

@pytest.mark.parametrize("name", sorted(available_components()))
def test_no_component_hides_a_tunable_in_python(name):
    """The invariant the whole tree exists for: a component's kwargs must all be
    keyword-only and defaultless, so a value can never differ from what config/
    says and never be missing from the run's wandb config."""
    problems = model_config.no_default_problems(get_component(name))
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("name", sorted(available_components()))
def test_every_component_resolves_from_the_tree_alone(name, tree):
    """Whatever a component requires, config/ supplies -- checked by actually
    constructing it, which is the only way a missing kwarg shows up."""
    if name not in tree.components:
        pytest.skip(f"{name} has no entry; every slot naming it spells its kwargs out")
    from conftest import build
    build(name)


def test_the_tree_is_structurally_sound():
    assert not model_config.structure_problems()


@pytest.mark.parametrize("name", MODELS)
def test_every_model_validates(name):
    assert not model_config.validate(name)


# --- sizes ----------------------------------------------------------------

def test_a_size_means_the_same_keys_across_a_kind(tree):
    """A ladder whose rungs set different keys would make a size comparison
    silently vary *which* knobs move, not just their values."""
    for kind, table in tree.presets["sizes"].items():
        shapes = {name: frozenset(entry) for name, entry in table.items()}
        assert len(set(shapes.values())) == 1, f"{kind}: {shapes}"


def test_size_keys_are_owned_by_the_ladder_alone(tree):
    """Nothing but the ladder may set a size key -- otherwise a component's own
    baseline would shadow the size a slot asked for."""
    for cls_name, entry in tree.components.items():
        owned = tree.size_keys(entry["kind"])
        assert not (owned & set(entry)), f"{cls_name} sets ladder-owned {sorted(owned & set(entry))}"


@pytest.mark.parametrize("kind", ["critic", "actor"])
def test_the_ladder_is_monotone_in_width(tree, kind):
    """small < medium < large < xlarge, at one depth -- the property that makes
    the ladder a capacity axis rather than an assortment of nets."""
    table = tree.presets["sizes"][kind]
    widths = [table[n]["latent_dim"] for n in ("small", "medium", "large", "xlarge")]
    assert widths == sorted(widths) and len(set(widths)) == len(widths)
    assert len({table[n]["layers"] for n in table}) == 1, "a size ladder varies width, not depth"


def test_a_size_override_moves_exactly_the_ladder_keys(tree):
    """What `--set model.components.critic.size=large` has to do, and no more."""
    base  = tree.resolved("SAC")
    large = tree.with_overrides({"components.critic.size": "large"}).resolved("SAC")

    moved = {k for k in base if base[k] != large.get(k)}
    expected = {f"components.critic.{k}" for k in tree.size_keys("critic")
                if base[f"components.critic.{k}"] != tree.preset("sizes.critic.large")[k]}
    assert moved == expected | {"components.critic.size"}
    assert large["components.critic.latent_dim"] == tree.preset("sizes.critic.large")["latent_dim"]


def test_large_is_the_same_net_whichever_critic_class(tree):
    """The reason sizes are shared tables rather than per-class blocks."""
    sized = {}
    for cls_name in ("C51Critic", "HLGaussCritic", "EnsembleCritic"):
        kwargs = tree.component_kwargs(cls_name, size="large")
        sized[cls_name] = {k: kwargs[k] for k in tree.size_keys("critic")}
    assert len({tuple(sorted(v.items())) for v in sized.values()}) == 1, sized


# --- run keys -------------------------------------------------------------

def test_every_run_key_is_declared():
    """A run entry may not carry a key config/run/defaults.yaml doesn't declare;
    such a key could never be set and would never be logged."""
    for run in load_config(str(CONFIG_DIR / "experiments" / "online_learning.yaml")):
        assert not set(run) - run_keys() - INJECTED_KEYS


def test_training_keys_are_a_subset_of_the_run_keys():
    """A model's `training:` block writes into the run config, so every key it
    may set has to exist there."""
    assert TRAINING_KEYS <= run_keys()


# --- replay ---------------------------------------------------------------

def test_a_snapshot_replays_through_an_edit_to_the_tree(tmp_path):
    """The guarantee that lets a finished sweep run be re-run months later: the
    snapshot serves resolved values, so editing config/ underneath it cannot
    move them."""
    snapshot = ConfigTree().snapshot_for("SAC")
    original = ConfigTree(snapshot=snapshot).resolved("SAC")

    sizes = CONFIG_DIR / "presets" / "sizes.yaml"
    backup = tmp_path / "sizes.yaml"
    shutil.copy(sizes, backup)
    try:
        # Edit whichever rung SAC's critic actually names, so changing a slot's
        # `size:` cannot quietly turn this into a test of nothing.
        rung = ConfigTree().policy_cfg("SAC")["components"]["critic"]["size"]
        line = next(ln for ln in backup.read_text().splitlines()
                    if ln.strip().startswith(f"{rung}:") and "latent_dim" in ln)
        sizes.write_text(backup.read_text().replace(
            line, f"  {rung}: {{latent_dim: 99, layers: 7, n_models: 5}}"))
        assert ConfigTree().resolved("SAC")["components.critic.latent_dim"] == 99, \
            "the edit changed nothing, so this proves nothing"
        assert ConfigTree(snapshot=snapshot).resolved("SAC") == original
    finally:
        shutil.copy(backup, sizes)


def test_a_snapshot_is_json_round_trippable():
    """It is written into a run's YAML and a checkpoint's meta.json, so it has to
    survive serialization with no Python-only types in it."""
    snapshot = ConfigTree().snapshot_for("AMDP")
    assert json.loads(json.dumps(snapshot)) == snapshot


@pytest.mark.parametrize("name", MODELS)
def test_a_snapshot_resolves_what_the_live_tree_resolves(name):
    """A snapshot taken now and the tree it came from must agree -- otherwise the
    thing a run logs is not the thing it ran."""
    tree = ConfigTree()
    assert ConfigTree(snapshot=tree.snapshot_for(name)).resolved(name) == tree.resolved(name)


# --- profiles and aliases -------------------------------------------------

def test_a_profile_reached_through_an_alias_overrides_every_block(tmp_path):
    """The named-variant layer: an alias resolves to a (policy, profile) pair and
    the profile may move a slot's `size:`, its kwargs, the policy's `params:` and
    its `training:` -- while a run override still beats all of it. Exercised here
    because the tree ships no profiles of its own."""
    root = tmp_path / "config"
    shutil.copytree(CONFIG_DIR, root)
    (root / "models" / "sac.yaml").write_text(
        (root / "models" / "sac.yaml").read_text() + """
  profiles:
    big:
      training: {batch_size: 1024}
      params:   {entropy_scale: -2.0}
      critic:   {size: xlarge}
      actor:    {size: large, lr: 1.0e-4}
""")
    (root / "models" / "aliases.yaml").write_text(
        "aliases:\n  sac_big: {policy: SAC, profile: big}\n")

    tree = ConfigTree(root)
    assert "sac_big" in tree.available_models()
    big = tree.resolved("sac_big")
    assert big["components.critic.latent_dim"] == tree.preset("sizes.critic.xlarge")["latent_dim"]
    assert big["components.actor.size"] == "large"
    assert big["components.actor.lr"] == 1.0e-4      # a profile kwarg beats the size preset
    assert big["params.entropy_scale"] == -2.0
    assert big["training.batch_size"] == 1024

    overridden = ConfigTree(root, overrides={"components.critic.size": "small"}).resolved("sac_big")
    assert overridden["components.critic.latent_dim"] == tree.preset("sizes.critic.small")["latent_dim"]

    snapshot = tree.snapshot_for("sac_big")
    assert ConfigTree(root, snapshot=snapshot).resolved("sac_big") == big


# --- provenance -----------------------------------------------------------

@pytest.mark.parametrize("name", MODELS)
def test_every_resolved_value_is_attributable(name, tree):
    """explain() has to account for every value resolved(), or print_config.py
    would be showing a different config than the run uses."""
    explained = tree.explain(name)
    for slot, block in explained["components"].items():
        resolved = tree.resolve_slot(*tree.resolve_model_name(name)[:1], slot,
                                     tree.resolve_model_name(name)[1])[1]
        assert set(block["kwargs"]) == set(resolved)
        for key, info in block["kwargs"].items():
            assert info["value"] == resolved[key]
            assert info["source"]
