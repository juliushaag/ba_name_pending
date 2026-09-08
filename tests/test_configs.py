"""The general YAML-with-includes + typed-access toolkit config_tree.py is
built on (barl_jax.configs) -- adapted from a sibling project's own test
suite for the module this was ported from. Exercises the toolkit in
isolation, against synthetic tmp_path trees, since config_tree.py's own
tests (test_config_tree.py) cover the precedence chain built on top of it.
"""


import pytest
import yaml

from barl_jax.config.toolkit import Config, ConfigMixin, _check_type, _format_type, deep_merge

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_config(tmp_path):
  data = {
    "model_type": "test",
    "device": "cpu",
    "seed": 42,
    "model": {
      "latent_dim": 256,
      "dropout": 0.1,
      "layers": [64, 128, 256],
      "use_batch_norm": True,
    },
    "training": {"lr": 0.001, "epochs": 100, "batch_size": 32},
  }
  path = tmp_path / "test.yaml"
  path.write_text(yaml.safe_dump(data))
  return path


@pytest.fixture
def config_with_includes(tmp_path):
  shared_dir = tmp_path / "shared"
  shared_dir.mkdir()
  (shared_dir / "training.yaml").write_text(yaml.safe_dump(
    {"lr": 0.0001, "weight_decay": 0.01, "warmup_steps": 100}))

  model_dir = tmp_path / "model"
  model_dir.mkdir()
  (model_dir / "base.yaml").write_text(yaml.safe_dump({
    "model_type": "test",
    "device": "cuda",
    "training": "$(/shared/training.yaml)",
    "model": {"latent_dim": 512},
  }))
  return model_dir / "base.yaml", tmp_path


# ---------------------------------------------------------------------------
# loading + includes
# ---------------------------------------------------------------------------

class TestConfigLoading:

  def test_load_simple_config(self, simple_config):
    config = Config.load(simple_config)
    assert config.get("model_type") == "test"
    assert config.get("device") == "cpu"
    assert config.get("seed") == 42

  def test_load_nested_values(self, simple_config):
    config = Config.load(simple_config)
    assert config.get("model.latent_dim") == 256
    assert config.get("training.lr") == 0.001

  def test_load_with_includes(self, config_with_includes):
    path, root = config_with_includes
    config = Config.load(path, config_root=root)
    assert config.get("device") == "cuda"
    assert config.get("training.lr") == 0.0001
    assert config.get("training.weight_decay") == 0.01
    assert config.get("training.warmup_steps") == 100

  def test_missing_file_error(self, tmp_path):
    with pytest.raises(FileNotFoundError):
      Config.load(tmp_path / "nonexistent.yaml")

  def test_missing_include_error(self, tmp_path):
    path = tmp_path / "test.yaml"
    path.write_text(yaml.safe_dump({"training": "$(missing.yaml)"}))
    with pytest.raises(FileNotFoundError, match="Included config not found"):
      Config.load(path)


# ---------------------------------------------------------------------------
# typed access
# ---------------------------------------------------------------------------

class TestTypeChecking:

  def test_int_and_float(self, simple_config):
    config = Config.load(simple_config)
    assert config.get("seed", int) == 42
    assert config.get("model.dropout", float) == 0.1

  def test_int_accepted_as_float(self, simple_config):
    value = Config.load(simple_config).get("seed", float)
    assert value == 42
    assert isinstance(value, float), "an int YAML value asked for as float should come back as one"

  def test_str_and_bool(self, simple_config):
    config = Config.load(simple_config)
    assert config.get("device", str) == "cpu"
    assert config.get("model.use_batch_norm", bool) is True

  def test_list_and_list_of_int(self, simple_config):
    config = Config.load(simple_config)
    assert config.get("model.layers", list) == [64, 128, 256]
    assert config.get("model.layers", list[int]) == [64, 128, 256]

  def test_wrong_type_raises(self, simple_config):
    with pytest.raises(TypeError, match="wrong type"):
      Config.load(simple_config).get("seed", str)

  def test_bool_int_distinction(self, simple_config):
    config = Config.load(simple_config)
    config.get("seed", int)  # int as int: fine
    with pytest.raises(TypeError):
      config.get("seed", bool)
    with pytest.raises(TypeError):
      config.get("model.use_batch_norm", int)

  def test_union_and_optional(self):
    _check_type(1.5, float | None, "x", "")
    _check_type(None, float | None, "x", "")
    with pytest.raises(TypeError):
      _check_type("nope", float | None, "x", "")


class TestMissingKeyErrors:

  def test_missing_top_level_key(self, simple_config):
    with pytest.raises(KeyError, match="not found"):
      Config.load(simple_config).get("nonexistent_key")

  def test_missing_nested_key(self, simple_config):
    with pytest.raises(KeyError, match="not found"):
      Config.load(simple_config).get("model.nonexistent_key")

  def test_error_shows_available_keys(self, simple_config):
    with pytest.raises(KeyError, match="Available keys"):
      Config.load(simple_config).get("model.missing")


# ---------------------------------------------------------------------------
# ConfigMixin
# ---------------------------------------------------------------------------

class TestConfigMixin:

  def test_section_access(self, simple_config):
    config = Config.load(simple_config)

    class Model(ConfigMixin):
      def __init__(self, data):
        ConfigMixin.__init__(self, "model", data)

    obj = Model(config.root)
    assert obj.get("latent_dim", int) == 256
    assert obj.get("dropout", float) == 0.1

  def test_get_optional(self, simple_config):
    config = Config.load(simple_config)
    assert config.get_optional("seed", 0) == 42
    assert config.get_optional("missing_key", 100) == 100
    assert config.get_optional("model.missing", "default") == "default"

  def test_get_section(self, simple_config):
    section = Config.load(simple_config).get_section("model")
    assert isinstance(section, dict) and section["latent_dim"] == 256


# ---------------------------------------------------------------------------
# deep merge / format_type
# ---------------------------------------------------------------------------

class TestDeepMerge:

  def test_simple_merge(self):
    assert deep_merge({"a": 1, "b": 2}, {"b": 3, "c": 4}) == {"a": 1, "b": 3, "c": 4}

  def test_nested_merge_keeps_untouched_keys(self):
    base = {"model": {"latent_dim": 256, "dropout": 0.1}, "training": {"lr": 0.001}}
    override = {"model": {"dropout": 0.2, "layers": [64, 128]}}
    result = deep_merge(base, override)
    assert result["model"] == {"latent_dim": 256, "dropout": 0.2, "layers": [64, 128]}
    assert result["training"] == {"lr": 0.001}

  def test_does_not_mutate_inputs(self):
    base = {"a": {"b": 1}}
    deep_merge(base, {"a": {"c": 2}})
    assert base == {"a": {"b": 1}}


def test_format_type():
  assert _format_type(int) == "int"
  assert _format_type(list[int]) == "list[int]"
  assert _format_type(dict[str, float]) == "dict[str, float]"
  assert _format_type(float | None) == "float | NoneType"


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------

def test_empty_config(tmp_path):
  path = tmp_path / "empty.yaml"
  path.write_text("")
  assert Config.load(path).root == {}


def test_repr(simple_config):
  assert "Config" in repr(Config.load(simple_config))


# ---------------------------------------------------------------------------
# real consumers: barl_jax.config.load_config and barl_jax.launch.grid both take
# $() includes for free through the same toolkit, exercised here through
# their actual production code paths rather than through configs.py directly.
# ---------------------------------------------------------------------------

def test_experiment_file_runs_list_can_be_included(tmp_path):
  from barl_jax.config.run import load_config

  (tmp_path / "run").mkdir()
  (tmp_path / "run" / "defaults.yaml").write_text(yaml.safe_dump({}))
  (tmp_path / "run" / "envs.yaml").write_text(yaml.safe_dump({}))
  (tmp_path / "experiments").mkdir()
  (tmp_path / "experiments" / "shared_runs.yaml").write_text(yaml.safe_dump(
    [{"model": "SAC", "env": "HalfCheetah"}, {"model": "AMDP", "env": "Walker2d"}]))
  main = tmp_path / "experiments" / "main.yaml"
  main.write_text(yaml.safe_dump({"defaults": {}, "runs": "$(/experiments/shared_runs.yaml)"}))

  runs = load_config(str(main), root=tmp_path)
  assert [(r["model"], r["env"]) for r in runs] == [("SAC", "HalfCheetah"), ("AMDP", "Walker2d")]


def test_grid_spec_exclude_list_can_be_included(tmp_path):
  from barl_jax.launch import grid

  (tmp_path / "shared_exclude.yaml").write_text(yaml.safe_dump([{"algo": "SAC", "seed": 2}]))
  spec = tmp_path / "spec.yaml"
  spec.write_text(yaml.safe_dump({
    "axes": {"algo": ["SAC"], "env": ["HalfCheetah"], "seed": [0, 1, 2]},
    "exclude": "$(shared_exclude.yaml)",
  }))

  axes, exclude = grid.build_axes(grid=str(spec))
  assert exclude == [{"algo": "SAC", "seed": 2}]
  combos = grid.build_combos(axes, exclude)
  assert combos == [{"algo": "SAC", "env": "HalfCheetah", "seed": 0},
                    {"algo": "SAC", "env": "HalfCheetah", "seed": 1}]
