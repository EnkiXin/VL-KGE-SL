"""Bounded, one-GPU geometry search; select pilots on validation only."""

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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--queue-id", type=simple_name, required=True)
    parser.add_argument("--backup-root", type=Path)
    parser.add_argument("--pilot-epochs", type=positive_int, default=10)
    parser.add_argument("--formal-epochs", type=positive_int, default=200)
    parser.add_argument("--max-hours", type=positive_float, default=6.0)
    parser.add_argument("--learning-rates", type=positive_float, nargs="+", default=[0.01, 0.03])
    parser.add_argument("--models", type=simple_name, nargs="+", default=["euclidean", "mure", "murp", "sl8"])
    parser.add_argument("--coordinate-scale", type=positive_float, default=0.1)
    parser.add_argument("--chart-radius", type=positive_float, default=0.5)
    parser.add_argument("--initial-logit-scale", type=positive_float, default=100.0,
                        help="Base Euclidean/SL logit scale; MuRP uses this value / 4 for local metric calibration")
    parser.add_argument("--initial-offset", type=finite_float, default=0.0)
    args = parser.parse_args(argv)
    if len(set(args.models)) != len(args.models):
        parser.error("--models must not contain duplicates")
    if len(set(args.learning_rates)) != len(args.learning_rates):
        parser.error("--learning-rates must not contain duplicates")
    return args


def actual_initial_logit_scale(args, model):
    # With exp0 coordinates, squared hyperbolic distance is locally 4 * Euclidean squared distance.
    return args.initial_logit_scale / 4 if model == "murp" else args.initial_logit_scale


def runner_command(args, root, run_dir, model, learning_rate, kind):
    return [
        sys.executable, "-u", str(root / "scripts" / "run_geometry.py"),
        "--root", str(root), "--model", model, "--run-dir", str(run_dir),
        "--epochs", str(args.pilot_epochs if kind == "pilot" else args.formal_epochs),
        "--patience", "50", "--seed", "42", "--lr", str(learning_rate),
        "--coordinate-scale", str(args.coordinate_scale),
        "--chart-radius", str(args.chart_radius),
        "--initial-logit-scale", str(actual_initial_logit_scale(args, model)),
        "--initial-offset", str(args.initial_offset), "--kind", kind,
    ]


def read_result(run_dir, kind):
    result = json.loads((run_dir / "result.json").read_text())
    if result.get("status") != "completed":
        raise ValueError("Runner did not report a completed result")
    value = result.get("best_validation_mrr")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Missing or invalid best_validation_mrr")
    epoch = result.get("best_epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
        raise ValueError("Missing or invalid best_epoch")
    if kind == "pilot" and result.get("final_test") is not None:
        raise ValueError("Pilot unexpectedly evaluated the test set")
    if kind == "formal" and not isinstance(result.get("final_test"), dict):
        raise ValueError("Formal run is missing final_test")
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
                "seed": 42, "max_hours": args.max_hours, "pilot_epochs": args.pilot_epochs,
                "formal_epochs": args.formal_epochs, "patience": 50,
                "models": args.models, "learning_rates": args.learning_rates,
                "geometry": {
                    "coordinate_scale": args.coordinate_scale, "chart_radius": args.chart_radius,
                    "initial_logit_scale": args.initial_logit_scale, "initial_offset": args.initial_offset,
                },
                "local_metric_calibration": {
                    "base_initial_logit_scale": args.initial_logit_scale,
                    "model_scale_factors": {model: 0.25 if model == "murp" else 1.0 for model in args.models},
                    "description": "Preserve standard MuRP squared distance. Since exp0 coordinates give locally 4 times Euclidean squared distance, initialize MuRP logit scale to base / 4; other models use base.",
                },
                "selection_policy": "Maximum pilot validation MRR per model; ties use first configured learning rate. Formal runs start from scratch.",
                "jobs": [], "validation_selection": {},
            }

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
                    "seed": 42, "status": "running", "started_at": timestamp(),
                    "actual_initial_logit_scale": actual_initial_logit_scale(args, model),
                    "command": command, "log": str(log), "run_dir": str(run_dir),
                }
                state["jobs"].append(job)
                save()
                process = None
                phase = "run"
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
                        raise RuntimeError(f"Runner exited with status {code}; see {log}")
                    result = read_result(run_dir, kind)
                    job.update(status="completed", ended_at=timestamp(),
                               best_validation_mrr=result["best_validation_mrr"],
                               best_epoch=result["best_epoch"])
                    if kind == "formal":
                        job["final_test"] = result["final_test"]
                    save()
                    if backup_queue is not None:
                        phase = "backup"
                        backup = backup_queue / name
                        backup.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copytree(run_dir, backup)
                        shutil.copy2(log, backup / "train.log")
                        job["backup"] = str(backup)
                        job["backup_status"] = "completed"
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
                for model in args.models:
                    for learning_rate in args.learning_rates:
                        code = run_job(model, learning_rate, "pilot")
                        if code:
                            return code
                for model in args.models:
                    pilots = [job for job in state["jobs"] if job["kind"] == "pilot" and job["model"] == model]
                    best = max(pilots, key=lambda job: job["best_validation_mrr"])
                    state["validation_selection"][model] = {
                        "lr": best["lr"], "best_validation_mrr": best["best_validation_mrr"],
                        "best_epoch": best["best_epoch"], "pilot_run_dir": best["run_dir"],
                    }
                save()
                for model in args.models:
                    code = run_job(model, state["validation_selection"][model]["lr"], "formal")
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
