"""The guarantee that a run reaches its GPU.

Nothing downstream can tell a CPU-bound run from a slow algorithm -- it trains,
logs and checkpoints exactly as it should, ~160x slower per gradient step -- so
the guarantee has to hold at the launcher, on all three paths a run is started
from, and one more time inside the run itself.
"""

import importlib.util
import os
import pathlib

import pytest

from barl_jax.launch import executors, grid

LAUNCH = pathlib.Path(grid.__file__).parent


def test_check_config_leaves_no_device_var_behind(monkeypatch):
    """The launcher's own environment is what sbatch --export=ALL, tmux and
    LocalExecutor hand to every run of the sweep, so the config check must not
    be able to write to it -- whatever it needs set for itself."""
    for var in executors.DEVICE_VARS:
        monkeypatch.delenv(var, raising=False)

    assert grid.check_config(["SAC"], ["HalfCheetah"]) == []
    assert not [var for var in executors.DEVICE_VARS if var in os.environ]


@pytest.mark.parametrize("var", executors.DEVICE_VARS)
def test_child_env_drops_device_vars(var, monkeypatch):
    monkeypatch.setenv(var, "cpu")
    assert var not in executors.child_env()
    assert executors.child_env(CUDA_VISIBLE_DEVICES="0")["CUDA_VISIBLE_DEVICES"] == "0"


@pytest.mark.parametrize("script", ["run_rl.slurm", "tmux_pool.py"])
def test_started_runs_unset_device_vars(script):
    """Neither sbatch's `--export=ALL` nor a tmux server's environment is the
    launcher's to filter, so both paths unset the vars at their own end."""
    text = (LAUNCH / script).read_text()
    assert f"unset {' '.join(executors.DEVICE_VARS)}" in text


def test_run_refuses_to_start_without_an_accelerator(monkeypatch):
    """The backstop, which holds whatever put the run on the CPU: the suite
    itself runs under JAX_PLATFORMS=cpu, so this is the real refusal."""
    monkeypatch.delenv("BARL_ALLOW_CPU", raising=False)
    from barl_jax.launch import worker as online_learning

    with pytest.raises(SystemExit, match="no GPU/TPU backend"):
        online_learning.require_accelerator()

    monkeypatch.setenv("BARL_ALLOW_CPU", "1")
    online_learning.require_accelerator()


def test_local_run_command_prefers_uv(monkeypatch):
    monkeypatch.setattr(grid.shutil, "which", lambda name: "/tmp/uv" if name == "uv" else None)
    run = grid.Run(index=0, tag="t", combo={}, dir="t", config="t/config.yaml")
    assert run.command("/tmp/sweep")[:5] == ["/tmp/uv", "run", "python", "-m", "barl_jax.launch.worker"]


@pytest.mark.parametrize("exit_code", [0, 2, None])
def test_submit_sweep_records_results_and_propagates_failure(tmp_path, exit_code):
    spec = importlib.util.spec_from_file_location(
        "submit", pathlib.Path(__file__).resolve().parent.parent / "scripts/submit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    submit_sweep = module.submit_sweep
    class FakeExecutor:
        blocking = True

        def submit(self, runs, *args, **kwargs):
            runs[0].exit_code = exit_code
            return {"Result": "finished"}

    run = grid.Run(index=0, tag="t", combo={}, dir="t", config="t/config.yaml")
    args = (FakeExecutor(), [run], "sweep", tmp_path, {"executor": "fake"})
    if exit_code == 0:
        submit_sweep(*args, max_parallel=1)
    else:
        with pytest.raises(SystemExit, match="1"):
            submit_sweep(*args, max_parallel=1)
    assert (tmp_path / "sweep.yaml").is_file()


def test_grid_rejects_ignored_base_config(tmp_path):
    spec = tmp_path / "grid.yaml"
    spec.write_text("base: ignored.yaml\naxes: {algo: [SAC], env: [HalfCheetah]}\n")
    with pytest.raises(SystemExit, match="use run"):
        grid.build_axes(grid=str(spec))


def test_grid_tag_uses_seed_axis_override():
    axes = {"algo": ["SAC"], "env": ["HalfCheetah"], "seed": [0], "run.num_seeds": [3]}
    assert grid.name_combos(grid.build_combos(axes, []), axes, 5) == ["SAC_HalfCheetah_3seeds"]


def test_grid_materialization_freezes_overrides(tmp_path):
    import yaml

    combo = {"algo": "SAC", "env": "HalfCheetah", "seed": 7,
             "run.batch_size": 32, "model.params.entropy_scale": -0.5}
    runs = grid.materialize([combo], ["run"], defaults={}, sweep_dir=str(tmp_path), num_seeds=1)
    data = yaml.safe_load((tmp_path / runs[0].config).read_text())
    assert data["runs"][0]["seed"] == 7
    assert data["runs"][0]["batch_size"] == 32
    assert data["runs"][0]["num_seeds"] == 1
    assert data["snapshot"]["resolved"]["params.entropy_scale"] == -0.5
