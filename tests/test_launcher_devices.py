"""The guarantee that a run reaches its GPU.

Nothing downstream can tell a CPU-bound run from a slow algorithm -- it trains,
logs and checkpoints exactly as it should, ~160x slower per gradient step -- so
the guarantee has to hold at the launcher, on all three paths a run is started
from, and one more time inside the run itself.
"""

import importlib.util
import os
import pathlib
import sys

import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import executors  # noqa: E402
import grid  # noqa: E402


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    text = (SCRIPTS / script).read_text()
    assert f"unset {' '.join(executors.DEVICE_VARS)}" in text


def test_run_refuses_to_start_without_an_accelerator(monkeypatch):
    """The backstop, which holds whatever put the run on the CPU: the suite
    itself runs under JAX_PLATFORMS=cpu, so this is the real refusal."""
    monkeypatch.delenv("BARL_ALLOW_CPU", raising=False)
    online_learning = load_script("online_learning")

    with pytest.raises(SystemExit, match="no GPU/TPU backend"):
        online_learning.require_accelerator()

    monkeypatch.setenv("BARL_ALLOW_CPU", "1")
    online_learning.require_accelerator()


def test_local_run_command_prefers_uv(monkeypatch):
    monkeypatch.setattr(grid.shutil, "which", lambda name: "/tmp/uv" if name == "uv" else None)
    run = grid.Run(index=0, tag="t", combo={}, dir="t", config="t/config.yaml")
    assert run.command("/tmp/sweep")[:4] == ["/tmp/uv", "run", "python", str(SCRIPTS / "online_learning.py")]
