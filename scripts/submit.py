#!/usr/bin/env python3

"""Expand a grid and launch it through Slurm, tmux, or local workers.

    uv run python scripts/submit.py --algos SAC DIME --envs HalfCheetah --seeds 5
    uv run python scripts/submit.py --grid src/barl_jax/defaults/experiments/sweep_grid.yaml --dry-run
    uv run python scripts/submit.py --algos SAC --envs HalfCheetah --executor local --validate
    uv run python scripts/submit.py --eval-config experiments/eval.yaml --executor local
    uv run python scripts/submit.py --algos SAC DIME --envs HalfCheetah --from-sweep sweeps/<earlier>

A sweep can contain one run. --set pins model.<path> or run.<key> axes; --grid
adds cross products and exclusions. Configs are frozen before submission.

--eval-config and --from-sweep evaluate instead of training, through the same
executors, batched --batch-size checkpoints per task rather than one task per
checkpoint -- evaluation is cheap enough that one-task-per-checkpoint would
waste a node's worth of scheduling overhead. --eval-config submits exactly the
file scripts/evaluate.py --config already reads (see experiments/eval.yaml):
every `path` is whatever the file says, batched as-is. --from-sweep is the
narrower, axes-driven case: the same {algo, env, seed} axes training used
select which of an earlier training sweep's checkpoints to evaluate (its tags
have to match, since that is how a checkpoint is found)."""

import argparse
import datetime
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

from barl_jax.launch import grid
from barl_jax.launch.executors import EXECUTORS, SWEEP_NAME_FORMAT, SWEEPS_DIR, resolve

EVAL_MODES = ("default", "mean", "median", "bon", "topk", "axis_shuffle", "perturb")


def submit_sweep(executor, runs, sweep_name, sweep_dir, head, *,
                 max_parallel, debug=False, stream=False):
    """Submit prepared runs, record blocking results, and propagate failures."""
    report = executor.submit(runs, sweep_name, sweep_dir,
                             max_parallel=max_parallel, debug=debug, stream=stream)
    if executor.blocking:
        grid.write_sweep_yaml(str(sweep_dir), head, runs)
    print()
    width = max(map(len, report), default=0)
    for label, value in report.items():
        print(f"  {label + ':':<{width + 1}} {value}")
    if executor.blocking and any(r.exit_code != 0 for r in runs):
        raise SystemExit(1)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Axes: algo/env/seed, plus model.<dotted path> (a model override, e.g. "
               "model.components.critic.tier) and run.<key> (a run-config key, e.g. "
               "run.utd_ratio). Seeds are not an axis -- --seeds N vmaps N of them inside "
               "each run, so `--algos SAC DIME --seeds 5` is 2 runs of 5 seeds, not 10 runs.")

    what = parser.add_argument_group("what to run")
    what.add_argument("--config", help="Replay an experiment config in the foreground")
    what.add_argument("--algos", nargs="+",
                      help="`algo` axis, e.g. SAC DIME AMDP. Required unless --grid declares one.")
    what.add_argument("--envs", nargs="+",
                      help="`env` axis, e.g. HalfCheetah CheetahRun. Required unless --grid "
                           "declares one.")
    what.add_argument("--set", action="append", metavar="AXIS=VALUE", dest="sets",
                      help="Pin one axis to a single value across the sweep, e.g. "
                           "--set model.components.critic.tier=large --set run.batch_size=512. "
                           "Repeatable.")
    what.add_argument("--grid", default=None,
                      help="YAML sweep spec: {axes: {name: [values]}, "
                           "exclude: [{axis: value, ...}]}. Combined with any --algos/--envs/--set "
                           "given, which win per axis.")
    what.add_argument("--seeds", type=int, default=None,
                      help="Seeds per run, vmapped inside the one process. Overrides `num_seeds` "
                           "in config/run/defaults.yaml, which is otherwise the only place the "
                           "seed count is declared.")

    where = parser.add_argument_group("where it runs")
    where.add_argument("--executor", choices=("auto", *EXECUTORS), default="auto",
                       help="`auto` (default) picks slurm when `sbatch` is on PATH, tmux when "
                            "only tmux is, and local otherwise -- so the same command works on "
                            "the cluster and on a workstation. `local` runs the sweep in the "
                            "foreground on this machine and is never chosen automatically.")
    where.add_argument("--max-parallel", type=int, default=None,
                       help="Runs in flight at once (Slurm's --array %%N). Default: 8 under slurm, "
                            "one per detected GPU under tmux/local. Setting it above the GPU count "
                            "is how you deliberately put more than one run on a GPU.")
    where.add_argument("--sweep-dir", default=None,
                       help="Where this sweep's tree goes (default: sweeps/<timestamp>).")

    how = parser.add_argument_group("how")
    how.add_argument("--validate", action="store_true",
                     help="Smoke-test every {algo x env} instead of training: one rollout chunk + "
                          "one train_step + one optimizer step each, no eval/checkpointing/wandb. "
                          "Collapses every other axis to its first value, and streams each run's "
                          "output.")
    how.add_argument("--debug", action="store_true",
                     help="Smoke-test submission: one run at a time and a 30-minute wall clock cap "
                          "(--max-parallel is ignored; under slurm it also means the dev queue). "
                          "The runs are real, just cut short -- whatever finished is still "
                          "collected when the cap kills them.")
    how.add_argument("--skip-check", action="store_true",
                     help="Don't resolve and validate every algorithm's model config first. Only "
                          "worth it if the check itself is in the way -- a bad config fails every "
                          "task identically.")
    how.add_argument("--dry-run", action="store_true",
                     help="Print the runs the grid expands to and exit, without writing or "
                          "submitting anything.")

    ev = parser.add_argument_group("evaluation (--eval-config or --from-sweep)")
    ev.add_argument("--eval-config",
                    help="Submit exactly the batch of evaluations this file describes -- the same "
                         "{output, defaults, runs: [{path, env?, ...}]} shape "
                         "scripts/evaluate.py --config reads (see experiments/eval.yaml). Its own "
                         "`defaults:`/`runs:` are authoritative; --checkpoint/--mode/--episodes/etc. "
                         "below don't apply here -- only --batch-size does.")
    ev.add_argument("--from-sweep",
                    help="Evaluate an earlier training sweep's checkpoints instead of training -- "
                         "the same {algo, env, seed} axes select which of its tags to evaluate.")
    ev.add_argument("--batch-size", type=int, default=None,
                    help="Checkpoints per evaluation task/node (default: all of them, one task "
                         "total -- pass a smaller number for more parallelism).")
    ev.add_argument("--checkpoint", choices=("best", "final"), default="best",
                    help="--from-sweep only.")
    ev.add_argument("--mode", nargs="+", choices=EVAL_MODES, default=["default"],
                    help="--from-sweep only. Action-selection mode(s), see barl_jax.execution.")
    ev.add_argument("--episodes", type=int, default=10, help="--from-sweep only.")
    ev.add_argument("--num-envs", type=int, default=8, help="--from-sweep only.")
    ev.add_argument("--num-candidates", type=int, default=8, help="--from-sweep only.")
    ev.add_argument("--top-k", type=int, default=2, help="--from-sweep only. Only used by --mode topk")
    ev.add_argument("--sigma", type=float, default=0.1, help="--from-sweep only. Only used by --mode perturb")
    ev.add_argument("--seed", type=int, default=0,
                    help="--from-sweep only. Shared evaluation seed for paired comparisons")

    args = parser.parse_args(argv)
    if args.seeds is not None and args.seeds < 1:
        parser.error("--seeds must be >= 1")
    if args.max_parallel is not None and args.max_parallel < 1:
        parser.error("--max-parallel must be >= 1")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.eval_config and args.from_sweep:
        parser.error("--eval-config and --from-sweep are two different ways to select what to "
                     "evaluate; pass only one")
    if args.batch_size is not None and not (args.eval_config or args.from_sweep):
        parser.error("--batch-size only applies with --eval-config or --from-sweep")
    if not args.from_sweep and (args.checkpoint != "best" or args.mode != ["default"]
                                or args.episodes != 10 or args.num_envs != 8
                                or args.num_candidates != 8 or args.top_k != 2
                                or args.sigma != 0.1 or args.seed != 0):
        parser.error("--checkpoint/--mode/--episodes/--num-envs/--num-candidates/--top-k/--sigma/"
                     "--seed only apply with --from-sweep")
    if args.eval_config and (args.algos or args.envs or args.grid or args.sets or args.seeds is not None):
        parser.error("--eval-config's file already says what to evaluate; "
                     "--algos/--envs/--grid/--set/--seeds don't apply")
    return args


def _submit_eval(args, entries, eval_defaults, *, source_desc, extra_head, preview_lines=()):
    """Shared by --eval-config and --from-sweep, once each has built its own
    `entries` (each a full evaluate.py `runs:` item, `{path, env?, ...}`) and
    `eval_defaults` (each batch's shared `defaults:`): preview, the --dry-run
    gate, materialize_eval(), and the same submit_sweep() call training uses
    -- same executors, same reporting, same blocking-vs-detached semantics."""
    executor = resolve(args.executor)
    max_parallel = 1 if args.debug else (args.max_parallel or executor.default_parallelism)

    sweep_name = datetime.datetime.now().strftime(SWEEP_NAME_FORMAT)  # noqa: DTZ005
    sweep_dir = Path(args.sweep_dir or (SWEEPS_DIR / f"eval_{sweep_name}")).absolute()
    batches = grid.batch_tags(entries, args.batch_size)

    # ---- what we're about to do -------------------------------------------
    print()
    print(f"Eval sweep: {len(entries)} checkpoint(s) from {source_desc}, "
          f"{len(batches)} batch(es) of up to {args.batch_size or len(entries)}")
    for line in preview_lines:
        print(line)
    print(f"Executor:   {executor.name}" + (" (auto)" if args.executor == "auto" else ""))
    print(f"Parallel:   {max_parallel}")
    if args.debug:
        print(f"Debug:      {executor.describe_debug()}")
    print(f"Dir:        {sweep_dir}")
    print()
    for index, batch in enumerate(batches):
        paths = ", ".join(entry["path"] for entry in batch)
        print(f"  [{index:03d}] batch{index:03d} -- {len(batch)} checkpoint(s): {paths}")

    if args.dry_run:
        print()
        print("Dry run: nothing written, nothing submitted.")
        return

    # ---- expand the batches onto disk --------------------------------------
    runs = grid.materialize_eval(entries, batch_size=args.batch_size,
                                 eval_defaults=eval_defaults, sweep_dir=str(sweep_dir))
    head = {"sweep_tag": sweep_name, "kind": "eval", "command": shlex.join(sys.argv),
           "executor": executor.name, "batch_size": args.batch_size, **extra_head}

    submit_sweep(executor, runs, sweep_name, sweep_dir, head,
                max_parallel=max_parallel, debug=args.debug, stream=False)


def submit_eval_config(args):
    """--eval-config: the file is authoritative. Its own `defaults:`/`runs:`
    -- the exact {output, defaults, runs} shape scripts/evaluate.py --config
    already reads, see experiments/eval.yaml -- are what gets batched and
    submitted; no axes are involved at all."""
    data = yaml.safe_load(Path(args.eval_config).read_text()) or {}
    entries = data.get("runs")
    if not entries:
        raise SystemExit(f"{args.eval_config}: no `runs` entries")
    for i, entry in enumerate(entries):
        if "path" not in entry:
            raise SystemExit(f"{args.eval_config}: runs[{i}] has no `path`")

    _submit_eval(args, entries, data.get("defaults") or {}, source_desc=args.eval_config,
                extra_head={"eval_config": str(Path(args.eval_config).resolve())})


def submit_eval(args):
    """--from-sweep: same axes as training (build_axes/build_combos/
    name_combos, unchanged) -- what guarantees a tag this produces is the same
    string the earlier training submission wrote under --from-sweep.
    Synthesizes one `{"path": "<from_sweep>/<tag>"}` entry per tag; `env` is
    left unset, evaluate.py's own discovery reads it from that tag's own
    config.yaml."""
    axes, exclude = grid.build_axes(grid=args.grid, algos=args.algos,
                                    envs=args.envs, sets=args.sets)
    if args.seeds is not None and "run.num_seeds" in axes:
        axes["run.num_seeds"] = [args.seeds]

    combos = grid.build_combos(axes, exclude)
    if not combos:
        raise SystemExit("grid is empty -- every combination was excluded")

    # Only to reconstruct the same tags training wrote -- no config.yaml of its
    # own is built from these combos the way materialize() does.
    num_seeds = args.seeds if args.seeds is not None else grid.default_num_seeds()
    tags = grid.name_combos(combos, axes, num_seeds)
    from_sweep = os.path.abspath(args.from_sweep)
    entries = [{"path": os.path.join(from_sweep, tag)} for tag in tags]

    eval_defaults = {"checkpoint": args.checkpoint, "mode": args.mode, "episodes": args.episodes,
                     "num_envs": args.num_envs, "num_candidates": args.num_candidates,
                     "top_k": args.top_k, "sigma": args.sigma, "seed": args.seed}
    preview = ["Axes:       " + ", ".join(f"{k}={v}" for k, v in axes.items())]
    if exclude:
        preview.append(f"Exclude:    {exclude}")

    _submit_eval(args, entries, eval_defaults, source_desc=args.from_sweep, preview_lines=preview,
                extra_head={"from_sweep": from_sweep, "axes": {k: list(v) for k, v in axes.items()},
                           "exclude": exclude or None})


def main():
    args = parse_args()
    if args.config:
        if (args.algos or args.envs or args.grid or args.sets or args.validate or args.debug
                or args.executor != "auto" or args.max_parallel or args.sweep_dir or args.dry_run
                or args.from_sweep or args.eval_config):
            raise SystemExit("--config replays directly; only --seeds may override the saved config")
        command = [sys.executable, "-m", "barl_jax.launch.worker", "--config", args.config]
        if args.seeds is not None:
            command += ["--seeds", str(args.seeds)]
        raise SystemExit(subprocess.call(command))

    if args.eval_config:
        if args.validate or args.skip_check:
            raise SystemExit("--eval-config evaluates, not trains; --validate/--skip-check don't apply")
        return submit_eval_config(args)

    if args.from_sweep:
        if args.validate or args.skip_check:
            raise SystemExit("--from-sweep evaluates, not trains; --validate/--skip-check don't apply")
        return submit_eval(args)

    axes, exclude = grid.build_axes(grid=args.grid, algos=args.algos,
                                          envs=args.envs, sets=args.sets)
    if args.seeds is not None and "run.num_seeds" in axes:
        axes["run.num_seeds"] = [args.seeds]
    if args.validate:
        # One run per {algo, env}: every other axis collapses to its first value.
        axes = {k: (v if k in ("algo", "env") else v[:1]) for k, v in axes.items()}

    combos = grid.build_combos(axes, exclude)
    if not combos:
        raise SystemExit("grid is empty -- every combination was excluded")

    num_seeds = args.seeds if args.seeds is not None else grid.default_num_seeds()
    tags = grid.name_combos(combos, axes, num_seeds)

    executor = resolve(args.executor)
    max_parallel = 1 if args.debug else (args.max_parallel or executor.default_parallelism)

    sweep_name = datetime.datetime.now().strftime(SWEEP_NAME_FORMAT)  # noqa: DTZ005
    sweep_dir = Path(args.sweep_dir or (SWEEPS_DIR / sweep_name)).absolute()

    # ---- what we're about to do -------------------------------------------
    print()
    print(f"Sweep:    {grid.describe(combos, axes, num_seeds)}")
    print("Axes:     " + ", ".join(f"{k}={v}" for k, v in axes.items()))
    if exclude:
        print(f"Exclude:  {exclude}")
    print(f"Executor: {executor.name}" + (" (auto)" if args.executor == "auto" else ""))
    print(f"Parallel: {max_parallel}")
    if args.debug:
        print(f"Debug:    {executor.describe_debug()}")
    if args.validate:
        print("Validate: one rollout chunk + one train step per run, nothing logged")
    print(f"Dir:      {sweep_dir}")
    print()
    for index, tag in enumerate(tags):
        print(f"  [{index:03d}] {tag}")

    if args.dry_run:
        print()
        print("Dry run: nothing written, nothing submitted.")
        return

    # ---- check before it costs queue time ---------------------------------
    if args.skip_check:
        print("\nSkipping the config check (--skip-check).")
    else:
        print("\nChecking model configs ...", flush=True)
        problems = grid.check_config(sorted({c["algo"] for c in combos}),
                                sorted({c["env"] for c in combos}))
        if problems:
            print("\nRefusing to submit -- unusable config:")
            for problem in problems:
                print(f"  {problem}")
            raise SystemExit(1)
        print("  OK")

    # ---- expand the grid onto disk ----------------------------------------
    runs, head = grid.prepare_sweep(
        combos, tags, sweep_dir=str(sweep_dir),
        num_seeds=args.seeds, validate=args.validate, sweep_name=sweep_name,
        executor_name=executor.name, seeds=num_seeds,
        head_extra={"validate": bool(args.validate), "debug": bool(args.debug),
                   "axes": {k: list(v) for k, v in axes.items()}, "exclude": exclude or None})

    submit_sweep(executor, runs, sweep_name, sweep_dir, head,
                 max_parallel=max_parallel, debug=args.debug, stream=args.validate)


if __name__ == "__main__":
    main()
