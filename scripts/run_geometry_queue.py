"""Bounded one-GPU geometry search, with explicit validation/test selection labels."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("must be finite")
    return number


def positive_float(value):
    number = finite_float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def simple_name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise argparse.ArgumentTypeError("must be a simple directory name")
    return value


V2_SPEC = re.compile(r"^(slv2|euclidv2)(?:n(\d+))?$")
DATASET_EPOCH_DEFAULTS = {
    "wn9_img": (10, 200),
    "wikiart_mkg_v1": (5, 50),
    "wikiart_mkg_v2": (2, 20),
}
OPTIONAL_RUNNER_INTS = ("batch_size", "negatives", "score_chunk", "candidate_chunk", "query_batch")


def resolve_model(spec):
    """Map a queue model spec to (runner model name, matrix_dim or None).

    ``slv2``/``euclidv2`` use the queue's ``--matrix-dim``; ``slv2n28`` pins n=28.
    v1 names (euclidean, mure, murp, sl8) pass through unchanged.
    """
    match = V2_SPEC.match(spec)
    if match is None:
        return spec, None
    return match.group(1), (int(match.group(2)) if match.group(2) else None)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--queue-id", type=simple_name, required=True)
    parser.add_argument("--backup-root", type=Path)
    parser.add_argument("--dataset", choices=tuple(DATASET_EPOCH_DEFAULTS), default="wn9_img")
    parser.add_argument("--selection-split", choices=["validation", "test"], default="validation",
                        help="Test selection is explicitly labeled test_tuned_not_held_out")
    parser.add_argument("--pilot-epochs", type=positive_int,
                        help="Default by dataset: WN9 10, WikiArt v1 5, WikiArt v2 2")
    parser.add_argument("--formal-epochs", type=positive_int,
                        help="Default by dataset: WN9 200, WikiArt v1 50, WikiArt v2 20")
    parser.add_argument("--max-hours", type=positive_float, default=6.0)
    parser.add_argument("--learning-rates", type=positive_float, nargs="+", default=[0.01, 0.03])
    parser.add_argument("--models", type=simple_name, nargs="+", default=["euclidean", "mure", "murp", "sl8"])
    parser.add_argument("--coordinate-scale", type=positive_float, default=0.1)
    parser.add_argument("--chart-radius", type=positive_float, default=0.5)
    parser.add_argument("--initial-logit-scale", type=positive_float, default=100.0,
                        help="Base Euclidean/SL logit scale; MuRP uses this value / 4 for local metric calibration")
    parser.add_argument("--initial-offset", type=finite_float, default=0.0)
    parser.add_argument("--embedding-dim", type=positive_int, default=768,
                        help="Author entity/fusion width before the v2 coordinate mapping")
    parser.add_argument("--matrix-dim", type=positive_int, default=8, help="v2 default n; a spec like slv2n28 overrides it")
    parser.add_argument("--entity-mapping", choices=["linear", "fixed_pad", "direct"], default="linear",
                        help="v2 mapping: projection, zero padding, or direct same-width coordinates")
    parser.add_argument("--relation-radius", type=positive_float, default=1.5)
    parser.add_argument("--relation-init-norm", type=finite_float, default=0.5)
    parser.add_argument("--relation-mode", choices=["left", "sandwich"], default="left")
    parser.add_argument("--score-form", choices=["linear", "squared"], default="linear")
    parser.add_argument("--schatten-p", type=positive_float, default=2.0)
    parser.add_argument("--log-sqrt-steps", type=int, default=1)
    parser.add_argument("--log-terms", type=positive_int, default=12)
    parser.add_argument("--patience", type=positive_int, default=50)
    parser.add_argument("--validate-every", type=positive_int, default=1)
    for name in OPTIONAL_RUNNER_INTS:
        parser.add_argument("--" + name.replace("_", "-"), type=positive_int,
                            help="Override the runner default only when explicitly supplied")
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--pilots-only", action="store_true",
                           help="Select each model's learning rate from pilots without formal training")
    execution.add_argument("--formal-only-lr", type=positive_float,
                           help="Skip pilots and train all selected models at this explicit learning rate")
    parser.add_argument("--continue-failed-pilots", action="store_true",
                        help="Skip nonzero pilot runner exits after backup; malformed results remain fatal")
    args = parser.parse_args(argv)
    pilot_default, formal_default = DATASET_EPOCH_DEFAULTS[args.dataset]
    if args.pilot_epochs is None:
        args.pilot_epochs = pilot_default
    if args.formal_epochs is None:
        args.formal_epochs = formal_default
    if args.log_sqrt_steps < 0:
        parser.error("--log-sqrt-steps must be non-negative")
    if args.entity_mapping != "direct" and args.embedding_dim != 768:
        parser.error("linear/fixed_pad adapters require embedding_dim=768; use direct for a different width")
    for spec in args.models:
        runner_model, matrix_dim = resolve_model(spec)
        if runner_model not in ("euclidean", "mure", "murp", "sl8", "slv2", "euclidv2"):
            parser.error(f"unknown model spec {spec}")
        if args.entity_mapping in ("fixed_pad", "direct"):
            if runner_model not in ("slv2", "euclidv2"):
                parser.error(f"--entity-mapping {args.entity_mapping} requires only v2 models")
            resolved_dim = args.matrix_dim if matrix_dim is None else matrix_dim
            coordinate_dim = resolved_dim * resolved_dim - 1
            if args.entity_mapping == "fixed_pad" and coordinate_dim < args.embedding_dim:
                parser.error(f"--entity-mapping fixed_pad requires matrix_dim**2 - 1 >= embedding_dim; invalid model {spec}")
            if args.entity_mapping == "direct" and coordinate_dim != args.embedding_dim:
                parser.error(f"--entity-mapping direct requires embedding_dim == matrix_dim**2 - 1; invalid model {spec}")
    if len(set(args.models)) != len(args.models):
        parser.error("--models must not contain duplicates")
    if len(set(args.learning_rates)) != len(args.learning_rates):
        parser.error("--learning-rates must not contain duplicates")
    return args


def actual_initial_logit_scale(args, model):
    # With exp0 coordinates, squared hyperbolic distance is locally 4 * Euclidean squared distance.
    return args.initial_logit_scale / 4 if model == "murp" else args.initial_logit_scale


def runner_command(args, root, run_dir, model, learning_rate, kind):
    runner_model, matrix_dim = resolve_model(model)
    command = [
        sys.executable, "-u", str(root / "scripts" / "run_geometry.py"),
        "--root", str(root), "--model", runner_model, "--run-dir", str(run_dir),
        "--epochs", str(args.pilot_epochs if kind == "pilot" else args.formal_epochs),
        "--patience", str(args.patience), "--seed", "42", "--lr", str(learning_rate),
        "--coordinate-scale", str(args.coordinate_scale),
        "--chart-radius", str(args.chart_radius),
        "--initial-logit-scale", str(actual_initial_logit_scale(args, runner_model)),
        "--initial-offset", str(args.initial_offset), "--kind", kind,
        "--validate-every", str(args.validate_every),
    ]
    # Keep the WN9 invocation compatible with older runners and existing jobs.
    if args.dataset != "wn9_img":
        command += ["--dataset", args.dataset]
    if args.selection_split == "test":
        command += ["--selection-split", "test"]
    for name in OPTIONAL_RUNNER_INTS:
        value = getattr(args, name)
        if value is not None:
            command += ["--" + name.replace("_", "-"), str(value)]
    if runner_model in ("slv2", "euclidv2"):
        command += ["--matrix-dim", str(matrix_dim if matrix_dim else args.matrix_dim),
                    "--embedding-dim", str(args.embedding_dim),
                    "--entity-mapping", args.entity_mapping,
                    "--relation-radius", str(args.relation_radius),
                    "--relation-init-norm", str(args.relation_init_norm),
                    "--relation-mode", args.relation_mode, "--score-form", args.score_form,
                    "--schatten-p", str(args.schatten_p),
                    "--log-sqrt-steps", str(args.log_sqrt_steps), "--log-terms", str(args.log_terms)]
    return command


def read_result(run_dir, kind, selection_split="validation"):
    if selection_split not in ("validation", "test"):
        raise ValueError("Invalid selection_split")
    result = json.loads((run_dir / "result.json").read_text())
    if result.get("status") != "completed":
        raise ValueError("Runner did not report a completed result")
    if selection_split == "test":
        if (result.get("selection_split") != "test"
                or result.get("test_used_for_selection") is not True
                or result.get("evaluation_label") != "test_tuned_not_held_out"):
            raise ValueError("Test selection requires explicit test_tuned_not_held_out metadata")
    elif (result.get("selection_split", "validation") != "validation"
          or result.get("test_used_for_selection") is True
          or result.get("test_selection") is not None):
        raise ValueError("Validation selection unexpectedly used the test set")
    metric_key = "best_test_mrr" if selection_split == "test" else "best_validation_mrr"
    value = result.get(metric_key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Missing or invalid {metric_key}")
    selection_value = result.get("best_selection_mrr", value)
    if (isinstance(selection_value, bool) or not isinstance(selection_value, (int, float))
            or not math.isfinite(selection_value) or selection_value != value):
        raise ValueError(f"best_selection_mrr does not match {metric_key}")
    epoch = result.get("best_epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
        raise ValueError("Missing or invalid best_epoch")
    if kind == "pilot" and result.get("final_test") is not None:
        raise ValueError("Pilot unexpectedly reported final_test from the test set")
    if kind == "formal" and not isinstance(result.get("final_test"), dict):
        raise ValueError("Formal run is missing final_test")
    if kind == "formal" and selection_split == "test":
        final = result["final_test"]
        if (final.get("evaluation_label") != "test_tuned_not_held_out"
                or final.get("test_used_for_selection") is not True):
            raise ValueError("Formal test-selected result must be labeled test_tuned_not_held_out")
    return result


def stop_child(process):
    """Signal only the child we own, never other GPU users or tmux jobs."""
    if process is None or process.poll() is not None:
        return
    try:
        process.send_signal(signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main(argv=None):
    args = parse_args(argv)
    root = args.root.resolve()
    if not (root / "scripts" / "run_geometry.py").is_file():
        raise FileNotFoundError("Missing scripts/run_geometry.py under --root")
    queue = root / "runs" / args.queue_id
    backup_queue = (args.backup_root.resolve() / args.queue_id).resolve() if args.backup_root else None
    source_queue = queue.resolve()
    if backup_queue is not None and (backup_queue == source_queue or source_queue in backup_queue.parents):
        raise ValueError("Backup must be outside the source queue directory")
    (root / "runs").mkdir(parents=True, exist_ok=True)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Queue received signal {signum}")

    original_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in original_handlers:
        signal.signal(sig, interrupted)
    try:
        with (root / "runs" / "gpu0.lock").open("a+") as lock:
            # Acquire before creating the queue; even an empty prior queue is not reusable.
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            queue.mkdir(exist_ok=False)
            started = time.monotonic()
            deadline = started + args.max_hours * 3600
            state_path = queue / "queue.json"
            state = {
                "pid": os.getpid(), "status": "running", "started_at": timestamp(),
                "dataset": args.dataset,
                "selection_split": args.selection_split,
                "test_used_for_selection": args.selection_split == "test",
                "evaluation_label": ("test_tuned_not_held_out" if args.selection_split == "test"
                                     else "validation_selected"),
                "seed": 42, "max_hours": args.max_hours, "pilot_epochs": args.pilot_epochs,
                "formal_epochs": args.formal_epochs, "patience": args.patience,
                "validate_every": args.validate_every,
                "models": args.models, "learning_rates": args.learning_rates,
                "pilots_only": args.pilots_only, "formal_only_lr": args.formal_only_lr,
                "continue_failed_pilots": args.continue_failed_pilots,
                "runner_overrides": {name: getattr(args, name) for name in OPTIONAL_RUNNER_INTS
                                     if getattr(args, name) is not None},
                "geometry": {
                    "coordinate_scale": args.coordinate_scale, "chart_radius": args.chart_radius,
                    "initial_logit_scale": args.initial_logit_scale, "initial_offset": args.initial_offset,
                },
                "geometry_v2": {
                    "matrix_dim": args.matrix_dim, "entity_mapping": args.entity_mapping,
                    "embedding_dim": args.embedding_dim,
                    "relation_radius": args.relation_radius,
                    "relation_init_norm": args.relation_init_norm, "relation_mode": args.relation_mode,
                    "score_form": args.score_form, "schatten_p": args.schatten_p,
                    "log_sqrt_steps": args.log_sqrt_steps, "log_terms": args.log_terms,
                    "resolved_models": {spec: dict(zip(("runner_model", "matrix_dim"), resolve_model(spec))) for spec in args.models},
                },
                "local_metric_calibration": {
                    "base_initial_logit_scale": args.initial_logit_scale,
                    "model_scale_factors": {model: 0.25 if model == "murp" else 1.0 for model in args.models},
                    "description": "Preserve standard MuRP squared distance. Since exp0 coordinates give locally 4 times Euclidean squared distance, initialize MuRP logit scale to base / 4; other models use base.",
                },
                "selection_policy": (
                    f"Explicit formal-only learning rate; checkpoint selection uses {args.selection_split} MRR. Formal runs start from scratch."
                    if args.formal_only_lr is not None else
                    f"Maximum completed pilot {args.selection_split} MRR per model; ties use first configured learning rate. Formal runs start from scratch."
                ),
                "jobs": [], "selection": {},
            }
            selection_key = "test_selection" if args.selection_split == "test" else "validation_selection"
            score_key = "best_test_mrr" if args.selection_split == "test" else "best_validation_mrr"
            state[selection_key] = {}

            def save():
                state["elapsed_seconds"] = time.monotonic() - started
                temporary = state_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")
                os.replace(temporary, state_path)

            def finish(status, code, error=None):
                state.update(status=status, ended_at=timestamp())
                if error is not None:
                    state["error"] = repr(error)
                save()
                return code

            def run_job(model, learning_rate, kind):
                if time.monotonic() >= deadline:
                    return finish("budget_exhausted", 124)
                lr_label = str(learning_rate).replace(".", "p").replace("+", "")
                name = f"{kind}-{model}-lr-{lr_label}-seed42"
                run_dir = queue / name
                log = queue / f"{name}.log"
                command = runner_command(args, root, run_dir, model, learning_rate, kind)
                job = {
                    "name": name, "model": model, "kind": kind, "lr": learning_rate,
                    "dataset": args.dataset,
                    "selection_split": args.selection_split,
                    "test_used_for_selection": state["test_used_for_selection"],
                    "evaluation_label": state["evaluation_label"],
                    "seed": 42, "status": "running", "started_at": timestamp(),
                    "actual_initial_logit_scale": actual_initial_logit_scale(args, model),
                    "command": command, "log": str(log), "run_dir": str(run_dir),
                }
                state["jobs"].append(job)
                save()
                process = None
                phase = "run"

                def backup_job():
                    nonlocal phase
                    if backup_queue is None:
                        return
                    phase = "backup"
                    backup = backup_queue / name
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    if run_dir.exists():
                        shutil.copytree(run_dir, backup)
                    else:
                        # A runner may exit before creating its run directory.
                        backup.mkdir(exist_ok=False)
                    shutil.copy2(log, backup / "train.log")
                    job["backup"] = str(backup)
                    job["backup_status"] = "completed"

                try:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        job.update(status="not_started", ended_at=timestamp())
                        return finish("budget_exhausted", 124)
                    env = os.environ.copy()
                    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
                    env.update(PYTHONHASHSEED="42", PYTHONUNBUFFERED="1")
                    with log.open("x") as output:
                        # Keep the flock in the training process even if this queue is SIGKILLed.
                        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                                   cwd=root, env=env, pass_fds=(lock.fileno(),))
                        job["pid"] = process.pid
                        save()
                        code = process.wait(timeout=max(0, deadline - time.monotonic()))
                    job["exit_code"] = code
                    if code != 0:
                        if kind == "pilot" and args.continue_failed_pilots:
                            job.update(status="failed", ended_at=timestamp(),
                                       error=f"Runner exited with status {code}; see {log}",
                                       skipped_failed_pilot=True)
                            save()
                            backup_job()
                            save()
                            return 0
                        raise RuntimeError(f"Runner exited with status {code}; see {log}")
                    result = read_result(run_dir, kind, args.selection_split)
                    job.update(status="completed", ended_at=timestamp(),
                               best_selection_mrr=result.get("best_selection_mrr", result[score_key]),
                               best_epoch=result["best_epoch"])
                    job[score_key] = result[score_key]
                    if args.selection_split == "test" and result.get("test_selection") is not None:
                        job["test_selection"] = result["test_selection"]
                    if kind == "formal":
                        job["final_test"] = result["final_test"]
                    save()
                    backup_job()
                    save()
                    return 0
                except subprocess.TimeoutExpired:
                    # Ignore repeated termination signals while draining our own child.
                    for sig in original_handlers:
                        signal.signal(sig, signal.SIG_IGN)
                    stop_child(process)
                    job.update(status="budget_exhausted", ended_at=timestamp(),
                               exit_code=process.returncode if process is not None else None)
                    return finish("budget_exhausted", 124)
                except KeyboardInterrupt as error:
                    for sig in original_handlers:
                        signal.signal(sig, signal.SIG_IGN)
                    stop_child(process)
                    job.update(status="interrupted", ended_at=timestamp(), error=repr(error))
                    return finish("interrupted", 130, error)
                except Exception as error:
                    stop_child(process)
                    status = "backup_failed" if phase == "backup" else "failed"
                    job.update(status=status, ended_at=timestamp(), error=repr(error))
                    if phase == "backup":
                        job["backup_status"] = "failed"
                    return finish(status, 1, error)

            save()
            try:
                if args.formal_only_lr is None:
                    for model in args.models:
                        for learning_rate in args.learning_rates:
                            code = run_job(model, learning_rate, "pilot")
                            if code:
                                return code
                    for model in args.models:
                        pilots = [job for job in state["jobs"]
                                  if job["kind"] == "pilot" and job["model"] == model
                                  and job["status"] == "completed"]
                        if not pilots:
                            raise RuntimeError(f"No completed pilots for model {model}; cannot select a learning rate")
                        best = max(pilots, key=lambda job: job["best_selection_mrr"])
                        selected = {
                            "lr": best["lr"], "best_selection_mrr": best["best_selection_mrr"],
                            score_key: best[score_key], "selection_split": args.selection_split,
                            "test_used_for_selection": state["test_used_for_selection"],
                            "evaluation_label": state["evaluation_label"],
                            "best_epoch": best["best_epoch"], "pilot_run_dir": best["run_dir"],
                        }
                        state["selection"][model] = selected
                        state[selection_key][model] = selected
                    save()
                if args.pilots_only:
                    return finish("completed", 0)
                for model in args.models:
                    learning_rate = (args.formal_only_lr if args.formal_only_lr is not None
                                     else state["selection"][model]["lr"])
                    code = run_job(model, learning_rate, "formal")
                    if code:
                        return code
                return finish("completed", 0)
            except KeyboardInterrupt as error:
                return finish("interrupted", 130, error)
            except Exception as error:
                return finish("failed", 1, error)
    finally:
        for sig, handler in original_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
