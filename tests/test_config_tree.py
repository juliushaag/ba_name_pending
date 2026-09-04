"""The config tree's guarantees, as tests rather than as convention.

Three things this suite exists to hold: a tier means one thing everywhere, no
tunable can hide in Python, and a saved run or checkpoint replays exactly no
matter what `config/` says afterwards.
"""

import json
import shutil

import pytest
from flax import nnx

import barl_jax.config.tree as model_config
from barl_jax.config import INJECTED_KEYS, load_config, run_keys
from barl_jax.config.toolkit import CONFIG_DIR
from barl_jax.config.tree import TRAINING_KEYS, ConfigTree
from barl_jax.networks.registry import available_components, get_component

MODELS = sorted(ConfigTree().available_models())


@pytest.fixture
def tree():
    return ConfigTree()


@pytest.mark.parametrize("name", sorted(available_components()))
def test_every_component_resolves_from_the_tree_alone(name, tree):
    """Whatever a component requires, config/ supplies -- checked by actually
    constructing it, which is the only way a missing kwarg shows up. The class
    HParam declarations ARE the baseline now, so every registered component
    resolves from the tree even with no config/components/ entry."""
    from conftest import build
    build(name)


def test_the_tree_is_structurally_sound():
    assert not model_config.structure_problems()


@pytest.mark.parametrize("name", MODELS)
def test_every_model_validates(name):
    assert not model_config.validate(name)


# --- tiers ----------------------------------------------------------------

def _ladder_kinds(tree) -> list[str]:
    from barl_jax.hparam import BUNDLES
    return sorted({name.split(".")[1] for name in BUNDLES if name.startswith("tiers.")})


def test_a_tier_means_the_same_keys_across_a_kind(tree):
    """A ladder whose rungs set different keys would make a tier comparison
    silently vary *which* knobs move, not just their values."""
    for kind in _ladder_kinds(tree):
        table = tree.tier_table(kind)
        shapes = {name: frozenset(entry) for name, entry in table.items()}
        assert len(set(shapes.values())) == 1, f"{kind}: {shapes}"


def test_tier_keys_are_owned_by_the_ladder_alone(tree):
    """Nothing but the ladder may set a tier key -- otherwise a component's own
    default would shadow the tier a slot asked for."""
    for name in sorted(available_components()):
        cls = get_component(name)
        owned = set(tree.tier_keys(tree.component_kind(name)))
        declared = set(getattr(cls, "_hparams", {}))
        assert not (owned & declared), f"{name} declares ladder-owned {sorted(owned & declared)}"


@pytest.mark.parametrize("kind", ["critic", "actor"])
def test_the_ladder_is_monotone_in_width(tree, kind):
    """low < medium < large < xlarge, at one depth -- the property that makes
    the ladder a capacity axis rather than an assortment of nets."""
    table = tree.tier_table(kind)
    widths = [table[n]["latent_dim"] for n in ("low", "medium", "large", "xlarge")]
    assert widths == sorted(widths) and len(set(widths)) == len(widths)
    assert len({table[n]["layers"] for n in table}) == 1, "a tier ladder varies width, not depth"


def test_a_tier_override_moves_exactly_the_ladder_keys(tree):
    """What `--set model.components.critic.tier=large` has to do, and no more."""
    from barl_jax import hparam
    base  = tree.resolved("SAC")
    large = ConfigTree(overrides={"components.critic.tier": "large"}).resolved("SAC")

    moved = {k for k in base if base[k] != large.get(k)}
    expected = {f"components.critic.{k}" for k in tree.tier_keys("critic")
                if base[f"components.critic.{k}"] != hparam.tier("critic", "large")[k]}
    assert moved == expected | {"components.critic.tier"}
    assert large["components.critic.latent_dim"] == hparam.tier("critic", "large")["latent_dim"]


def test_large_is_the_same_net_whichever_critic_class(tree):
    """The reason tiers are shared tables rather than per-class blocks."""
    sized = {}
    for cls_name in ("C51Critic", "HLGaussCritic", "EnsembleCritic"):
        kwargs = tree.component_kwargs(cls_name, tier="large")
        sized[cls_name] = {k: kwargs[k] for k in tree.tier_keys("critic")}
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


# --- completeness: no value is a silent default -------------------------------

def test_resolved_covers_every_declared_hparam(tree):
    """The requirement defaults moving into code was bought at: the resolved
    config a run logs contains every declared HParam or Config -- parameter or
    slot -- with its effective value. A class default a model never mentions
    still shows up. Nothing can hide behind a Python default."""
    from barl_jax.policy import Policy
    for model in tree.available_models():
        policy_name = tree.policy_class_name(model)
        cls = Policy._classes[policy_name]
        res = tree.resolved(model)
        params = {k[len("params."):] for k in res if k.startswith("params.")}
        assert set(cls._hparams) | set(cls._configs) <= params, \
            f"{model}: declared params missing from resolved()"
        for slot in cls.SLOTS:
            comp = get_component(res[f"components.{slot}.class"])
            keys = {k[len(f"components.{slot}."):] for k in res
                    if k.startswith(f"components.{slot}.")
                    and k not in (f"components.{slot}.class", f"components.{slot}.tier")}
            assert set(comp._hparams) | set(comp._configs) <= keys, \
                f"{model}.{slot}: declared hparams missing from resolved()"


def test_a_resolved_run_has_every_run_key():
    """The run side of the same guarantee: a resolved run dict carries a value
    for every declared run key, so trainer's `cfg[...]` indexing and the
    wandb `run/*` block are complete by construction."""
    from barl_jax.config import load_config
    from barl_jax.config import RUN_CONFIGS
    runs = load_config(str(CONFIG_DIR / "experiments" / "online_learning.yaml"))
    assert runs
    for run in runs:
        assert set(RUN_CONFIGS) <= set(run), \
            f"resolved run for {run['model']} on {run['env']} missing run keys"


# --- replay ---------------------------------------------------------------

def test_a_snapshot_replays_through_an_edit_to_the_tree():
    """The guarantee that lets a finished sweep run be re-run months later: the
    snapshot serves resolved values, so editing the tree underneath it cannot
    move them. Tiers now live in the code ladder (barl_jax.hparam), so "editing
    the tree" means monkeypatching the registered ladder."""
    import barl_jax.hparam as hparam

    snapshot = ConfigTree().snapshot_for("SAC")
    original = ConfigTree(snapshot=snapshot).resolved("SAC")

    rung = ConfigTree().policy_cfg("SAC")["components"]["critic"]["tier"]
    key = f"tiers.critic.{rung}"
    saved = hparam.BUNDLES[key]
    try:
        hparam.BUNDLES[key] = {**saved, "latent_dim": 99, "layers": 7, "n_models": 5}
        assert ConfigTree().resolved("SAC")["components.critic.latent_dim"] == 99, \
            "the edit changed nothing, so this proves nothing"
        assert ConfigTree(snapshot=snapshot).resolved("SAC") == original
    finally:
        hparam.BUNDLES[key] = saved


def test_a_stale_snapshot_overlays_missing_keys_instead_of_raising(tmp_path):
    """A snapshot taken before a config/ change added a new required key must
    not fail to replay: the recorded keys stay authoritative, but a key the
    snapshot never saw falls back to today's live value instead of being
    missing from the resolved dict."""
    snapshot = ConfigTree().snapshot_for("SAC")
    resolved = dict(snapshot["resolved"])

    # Simulate "before a key existed": drop one params.* and one
    # components.<slot>.* key the snapshot actually recorded.
    dropped_param = next(k for k in resolved if k.startswith("params."))
    dropped_slot_kwarg = next(k for k in resolved
                              if k.startswith("components.actor.")
                              and k not in ("components.actor.class", "components.actor.tier"))
    stale = dict(snapshot, resolved={k: v for k, v in resolved.items()
                                     if k not in (dropped_param, dropped_slot_kwarg)})

    tree = ConfigTree(snapshot=stale)
    live_params = ConfigTree().resolve_params("SAC", None)
    live_slot = ConfigTree().resolve_slot("SAC", "actor", None)[1]

    replayed_params = tree.resolve_params("SAC", None)
    replayed_slot = tree.resolve_slot("SAC", "actor", None)[1]

    dropped_param_name = dropped_param[len("params."):]
    dropped_slot_name = dropped_slot_kwarg.split(".", 2)[2]
    assert replayed_params[dropped_param_name] == live_params[dropped_param_name]
    assert replayed_slot[dropped_slot_name] == live_slot[dropped_slot_name]

    # A key the stale snapshot *did* keep still wins over the live value, even
    # if the live tree would resolve it differently -- that is the guarantee
    # the overlay must not weaken. Mutate the recorded value so it disagrees
    # with the live tree, and confirm the replayed value is the recorded one.
    kept_slot_kwarg = next(k for k in resolved
                           if k.startswith("components.actor.")
                           and k not in ("components.actor.class", "components.actor.tier",
                                         dropped_slot_kwarg))
    kept_name = kept_slot_kwarg.split(".", 2)[2]
    mutated_value = "unmistakably-not-a-live-value"
    stale["resolved"][kept_slot_kwarg] = mutated_value
    tree = ConfigTree(snapshot=stale)
    assert tree.resolve_slot("SAC", "actor", None)[1][kept_name] == mutated_value


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
    the profile may move a slot's `tier:`, its kwargs, the policy's `params:` and
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
      critic:   {tier: xlarge}
      actor:    {tier: large, lr: 1.0e-4}
""")
    (root / "models" / "aliases.yaml").write_text(
        "aliases:\n  sac_big: {policy: SAC, profile: big}\n")

    tree = ConfigTree(root)
    from barl_jax import hparam
    assert "sac_big" in tree.available_models()
    big = tree.resolved("sac_big")
    assert big["components.critic.latent_dim"] == hparam.tier("critic", "xlarge")["latent_dim"]
    assert big["components.actor.tier"] == "large"
    assert big["components.actor.lr"] == 1.0e-4      # a profile kwarg beats the tier preset
    assert big["params.entropy_scale"] == -2.0
    assert big["training.batch_size"] == 1024

    overridden = ConfigTree(root, overrides={"components.critic.tier": "low"}).resolved("sac_big")
    assert overridden["components.critic.latent_dim"] == hparam.tier("critic", "low")["latent_dim"]

    snapshot = tree.snapshot_for("sac_big")
    assert ConfigTree(root, snapshot=snapshot).resolved("sac_big") == big

    # Policy.make resolves the alias before constructing the policy, then asks
    # snapshot_for(canonical_model, explicit_profile). That path must preserve
    # profile training values just as the original alias path does.
    canonical = tree.resolved("SAC", "big")
    assert canonical["training.batch_size"] == 1024
    assert tree.snapshot_for("SAC", "big")["resolved"]["training.batch_size"] == 1024


def test_template_policy_defaults_are_inherited_without_yaml_repetition(tmp_path):
    root = tmp_path / "config"
    shutil.copytree(CONFIG_DIR, root)
    path = root / "models" / "sac.yaml"
    path.write_text(path.read_text().replace("    normalize_obs: true\n", ""))

    resolved = ConfigTree(root).resolve_params("SAC", None)
    assert resolved["normalize_obs"] is True


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
