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


def _load_submit():
    """scripts/submit.py isn't a package module -- same load-by-path trick
    tests/test_evaluate.py uses for scripts/evaluate.py."""
    spec = importlib.util.spec_from_file_location(
        "submit", pathlib.Path(__file__).resolve().parent.parent / "scripts/submit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_train():
    """scripts/train.py isn't a package module either -- same trick."""
    spec = importlib.util.spec_from_file_location(
        "train", pathlib.Path(__file__).resolve().parent.parent / "scripts/train.py")
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


@pytest.mark.parametrize("script", ["run_rl.slurm", "run_eval.slurm", "tmux_pool.py"])
def test_started_runs_unset_device_vars(script):
    """Neither sbatch's `--export=ALL` nor a tmux server's environment is the
    launcher's to filter, so both paths unset the vars at their own end."""
    text = (LAUNCH / script).read_text()
    assert f"unset {' '.join(executors.DEVICE_VARS)}" in text


def test_run_refuses_to_start_without_an_accelerator(monkeypatch):
    """The backstop, which holds whatever put the run on the CPU: the suite
    itself runs under JAX_PLATFORMS=cpu, so this is the real refusal."""
    monkeypatch.delenv("BARL_ALLOW_CPU", raising=False)
    train = _load_train()

    with pytest.raises(SystemExit, match="no GPU/TPU backend"):
        train.require_accelerator()

    monkeypatch.setenv("BARL_ALLOW_CPU", "1")
    train.require_accelerator()


def test_local_run_command_prefers_uv(monkeypatch):
    monkeypatch.setattr(grid.shutil, "which", lambda name: "/tmp/uv" if name == "uv" else None)
    run = grid.Run(index=0, tag="t", combo={}, dir="t", config="t/config.yaml")
    assert run.command("/tmp/sweep") == \
        ["/tmp/uv", "run", "python", "scripts/train.py", "--config", "/tmp/sweep/t/config.yaml"]


def test_eval_run_command_invokes_evaluate_py(monkeypatch):
    monkeypatch.setattr(grid.shutil, "which", lambda name: "/tmp/uv" if name == "uv" else None)
    run = grid.Run(index=0, tag="batch000", combo={"tags": ["x"]}, dir="batch000",
                   config="batch000/eval.yaml", kind="eval")
    assert run.command("/tmp/sweep") == \
        ["/tmp/uv", "run", "python", "scripts/evaluate.py", "--config", "/tmp/sweep/batch000/eval.yaml"]


def test_command_for_eval_passes_through_an_explicit_output(monkeypatch):
    monkeypatch.setattr(grid.shutil, "which", lambda name: None)
    cmd = grid.command_for("eval", "/x/eval.yaml", output_path="/scratch/results.json")
    assert cmd[-2:] == ["--output", "/scratch/results.json"]


@pytest.mark.parametrize("kind, script", [("train", "run_rl.slurm"), ("eval", "run_eval.slurm")])
def test_slurm_submit_picks_the_script_matching_run_kind(tmp_path, monkeypatch, kind, script):
    monkeypatch.setattr(executors.shutil, "which", lambda name: "/usr/bin/sbatch" if name == "sbatch" else None)
    calls = []

    class FakeCompleted:
        returncode = 0
        stdout = "12345;cluster"
        stderr = ""

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return FakeCompleted()

    monkeypatch.setattr(executors.subprocess, "run", fake_run)
    run = grid.Run(index=0, tag="t", combo={}, dir="t", config="t/config.yaml", kind=kind)
    executors.SlurmExecutor().submit([run], "sweep", tmp_path, max_parallel=1, debug=False)

    sbatch_cmd = calls[0]
    assert sbatch_cmd[-1].endswith(script)


def test_batch_tags_defaults_to_one_batch_and_chunks_when_given_a_size():
    tags = ["a", "b", "c", "d", "e"]
    assert grid.batch_tags(tags, None) == [tags]
    assert grid.batch_tags(tags, 2) == [["a", "b"], ["c", "d"], ["e"]]


def test_materialize_eval_writes_one_self_contained_config_per_batch(tmp_path):
    import yaml

    entries = [{"path": "/trained/SAC_HalfCheetah_1seeds"},
              {"path": "/trained/NCLQL_HalfCheetah_1seeds", "env": "HalfCheetah"},
              {"path": "/trained/DIME_HalfCheetah_1seeds"}]
    runs = grid.materialize_eval(entries, batch_size=2, eval_defaults={"episodes": 5},
                                 sweep_dir=str(tmp_path / "evals"))

    assert [r.kind for r in runs] == ["eval", "eval"]
    assert [len(r.combo["paths"]) for r in runs] == [2, 1]

    first = yaml.safe_load((tmp_path / "evals" / runs[0].config).read_text())
    assert first["defaults"] == {"episodes": 5}
    assert first["output"] == str(tmp_path / "evals" / "batch000" / "results.json")
    assert first["runs"] == entries[:2]

    second = yaml.safe_load((tmp_path / "evals" / runs[1].config).read_text())
    assert second["runs"] == entries[2:]


@pytest.mark.parametrize("exit_code", [0, 2, None])
def test_submit_sweep_records_results_and_propagates_failure(tmp_path, exit_code):
    module = _load_submit()
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


def test_eval_config_and_from_sweep_are_mutually_exclusive():
    module = _load_submit()
    with pytest.raises(SystemExit):
        module.parse_args(["--eval-config", "x.yaml", "--from-sweep", "y"])


def test_eval_config_rejects_axes_flags():
    module = _load_submit()
    with pytest.raises(SystemExit):
        module.parse_args(["--eval-config", "x.yaml", "--algos", "SAC"])


def test_eval_config_rejects_from_sweep_only_flags():
    module = _load_submit()
    with pytest.raises(SystemExit):
        module.parse_args(["--eval-config", "x.yaml", "--checkpoint", "final"])


def test_grid_rejects_ignored_base_config(tmp_path):
    spec = tmp_path / "grid.yaml"
    spec.write_text("base: ignored.yaml\naxes: {algo: [SAC], env: [HalfCheetah]}\n")
    with pytest.raises(SystemExit, match="not supported"):
        grid.build_axes(grid=str(spec))


def test_grid_tag_uses_seed_axis_override():
    axes = {"algo": ["SAC"], "env": ["HalfCheetah"], "seed": [0], "run.num_seeds": [3]}
    assert grid.name_combos(grid.build_combos(axes, []), axes, 5) == ["SAC_HalfCheetah_3seeds"]


def test_grid_materialization_freezes_overrides(tmp_path):
    import yaml

    from barl_jax.config import from_yaml
    combo = {"algo": "SAC", "env": "HalfCheetah", "seed": 7,
             "run.batch_size": 32, "model.params.entropy_scale": -0.5}
    runs = grid.materialize([combo], ["run"], defaults={}, sweep_dir=str(tmp_path), num_seeds=1)
    data = yaml.safe_load((tmp_path / runs[0].config).read_text())
    # Single-run mapping, loadable by the training CLI.
    assert data["model"] == "SAC" and data["env"] == "HalfCheetah"
    assert data["seed"] == 7
    assert data["batch_size"] == 32
    assert data["num_seeds"] == 1
    assert data["model_overrides"]["params"]["entropy_scale"] == -0.5
    # The exact emitted file round-trips through the public loader.
    assert from_yaml(tmp_path / runs[0].config).model == "SAC"


def test_submit_eval_end_to_end_through_local_executor(tmp_path):
    """--from-sweep, all the way through: a real checkpoint on disk, a real
    LocalExecutor subprocess (scripts/evaluate.py, which a monkeypatched
    Environment.make can't reach -- it runs in its own process), a real
    results.json. Two checkpoints, default batching (one batch total): both
    must land in the one task's results, matching the ask this whole feature
    is for -- every combo evaluated on the same node."""
    import json

    from barl_jax.checkpoints import save_policy
    from barl_jax.env import Environment
    from barl_jax.policy import Policy

    trained = tmp_path / "trained"
    for algo in ("SAC", "NCLQL"):
        tag = f"{algo}_HalfCheetah_1seeds"
        env_obj = Environment.class_for("HalfCheetah")()
        env = env_obj.get_env()
        try:
            policy = Policy.make(algo, env, env_stats=env_obj.get_env_stats(env), seed=0)
            save_policy(policy, trained / tag / "checkpoints" / "seed0" / "best")
        finally:
            env.close()
        (trained / tag / "config.yaml").write_text(
            f"runs: [{{model: {algo}, env: HalfCheetah, seed: 0}}]\n")

    module = _load_submit()

    sweep_dir = tmp_path / "evals"
    args = module.parse_args([
      "--algos", "SAC", "NCLQL", "--envs", "HalfCheetah", "--seeds", "1",
      "--from-sweep", str(trained), "--executor", "local",
      "--episodes", "2", "--num-candidates", "1", "--sweep-dir", str(sweep_dir),
    ])
    module.submit_eval(args)

    import math

    results = json.loads((sweep_dir / "batch000" / "results.json").read_text())
    assert len(results) == 2
    for row in results.values():
        assert "error" not in row
        assert math.isfinite(row["default"]["mean_return"])


def test_submit_eval_config_end_to_end_through_local_executor(tmp_path):
    """--eval-config, all the way through: two checkpoints saved with no
    training-sweep tag structure at all (no config.yaml to infer env from --
    the point of --eval-config over --from-sweep) and an explicit `env:` per
    entry in a hand-authored eval.yaml, submitted via a real LocalExecutor
    subprocess, real results.json."""
    import json
    import math

    from barl_jax.checkpoints import save_policy
    from barl_jax.env import Environment
    from barl_jax.policy import Policy

    checkpoints = {}
    for name, algo in [("agent-a", "SAC"), ("agent-b", "NCLQL")]:
        env_obj = Environment.class_for("HalfCheetah")()
        env = env_obj.get_env()
        try:
            policy = Policy.make(algo, env, env_stats=env_obj.get_env_stats(env), seed=0)
            path = tmp_path / "somewhere" / name / "best"
            save_policy(policy, path)
            checkpoints[name] = path
        finally:
            env.close()

    import yaml
    eval_config = tmp_path / "eval.yaml"
    eval_config.write_text(yaml.safe_dump({
      "defaults": {"episodes": 2, "num_candidates": 1},
      "runs": [{"path": str(p), "env": "HalfCheetah"} for p in checkpoints.values()],
    }))

    module = _load_submit()

    sweep_dir = tmp_path / "evals"
    args = module.parse_args(["--eval-config", str(eval_config), "--executor", "local",
                              "--sweep-dir", str(sweep_dir)])
    module.submit_eval_config(args)

    results = json.loads((sweep_dir / "batch000" / "results.json").read_text())
    assert set(results) == {str(p) for p in checkpoints.values()}
    for row in results.values():
        assert "error" not in row
        assert math.isfinite(row["default"]["mean_return"])
