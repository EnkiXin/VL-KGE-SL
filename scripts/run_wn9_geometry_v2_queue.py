"""Execute explicit WN9 V2 trials, without test scoring, before one UTC deadline.

No automatic promotions, resume, or extra wall budget. A control-file sentinel
requests a stop after the current trial and its persistent backup are finished.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import subprocess
import sys
import time


DEFAULTS = {
    "kind": "pilot", "epochs": 30, "patience": 50, "seed": 42,
    "lr": .003, "coordinate_scale": 1.0, "chart_radius": 1.5,
    "initial_logit_scale": 1.0, "initial_offset": 0.0,
    "target_init_norm": .5, "batch_size": 512, "negatives": 100,
    "score_chunk": 1024, "query_batch": 32, "candidate_chunk": 256,
    "grad_clip": 5.0, "log_order": 16, "validation_seed": 260906,
    "max_train_batches": None, "eval_limit": None,
}
MODELS = ("euclidean", "hyperbolic", "sl8")
PROTOCOL = "strict_directional_v2"
TRAIN_RESERVE_SECONDS = 45.0
SHUTDOWN_GRACE_SECONDS = 10.0


def utc():
    return datetime.now(timezone.utc).isoformat()


def parse_utc(value):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
            raise ValueError("explicit UTC timezone is required")
        return parsed.astimezone(timezone.utc)
    except (ValueError, TypeError, AttributeError) as error:
        raise argparse.ArgumentTypeError("use an explicit UTC timestamp, e.g. 2026-09-06T12:17:36Z") from error


def name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("Queue and job IDs must be simple directory names")
    return value


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def load_plan(path, deadline):
    raw = json.loads(path.read_text())
    if isinstance(raw, list):
        raw = {"schema_version": 1, "jobs": raw}
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("Expected schema_version=1 plan with a jobs list")
    if "deadline_utc" in raw and parse_utc(raw["deadline_utc"]) != deadline:
        raise ValueError("CLI deadline differs from the authorized plan deadline")
    common, entries = raw.get("common", {}), raw.get("jobs")
    if not isinstance(common, dict) or set(common) - set(DEFAULTS):
        raise ValueError("Unknown common trainer settings")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Plan has no jobs")
    jobs = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) - (set(DEFAULTS) | {"job_id", "stage", "model"}):
            raise ValueError("Unknown job fields; arbitrary commands are forbidden")
        config = {**DEFAULTS, **common, **{k: v for k, v in entry.items() if k in DEFAULTS}}
        model, stage = entry.get("model"), entry.get("stage")
        if model not in MODELS or stage not in ("smoke", "profile", "screen"):
            raise ValueError("Invalid model or stage (no automatic confirmation/test stage)")
        config["model"] = model
        if config["kind"] not in ("smoke", "pilot"):
            raise ValueError("Only validation-only smoke/pilot trials are allowed")
        if stage == "screen" and config["kind"] != "pilot":
            raise ValueError("Screen trials must train/evaluate complete epochs and full validation")
        for key in ("epochs", "patience", "batch_size", "negatives", "score_chunk", "query_batch", "candidate_chunk", "log_order"):
            if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if config["log_order"] not in (16, 32):
            raise ValueError("Use explicitly audited GL16 or GL32")
        for key in ("seed", "validation_seed"):
            if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        for key in ("lr", "coordinate_scale", "chart_radius", "initial_logit_scale", "target_init_norm", "grad_clip"):
            if isinstance(config[key], bool) or not isinstance(config[key], (float, int)) or not math.isfinite(config[key]) or config[key] <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if isinstance(config["initial_offset"], bool) or not isinstance(config["initial_offset"], (float, int)) or not math.isfinite(config["initial_offset"]):
            raise ValueError("initial_offset must be finite")
        for key in ("max_train_batches", "eval_limit"):
            value = config[key]
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0 or config["kind"] != "smoke"):
                raise ValueError(f"{key} is a positive smoke-only option")
        if config["target_init_norm"] >= config["chart_radius"]:
            raise ValueError("target_init_norm must be strictly below chart_radius")
        jobs.append({"job_id": name(entry.get("job_id")), "stage": stage, "config": config, "status": "not_started"})
    if len({job["job_id"] for job in jobs}) != len(jobs):
        raise ValueError("Duplicate job IDs")
    order = {"smoke": 0, "profile": 1, "screen": 2}
    if [order[j["stage"]] for j in jobs] != sorted(order[j["stage"]] for j in jobs):
        raise ValueError("Run smoke/profile before screening")
    return raw, jobs


def source_fingerprints(root, plan):
    paths = [root / "scripts/run_wn9_geometry_v2.py", root / "scripts/run_wn9_geometry_v2_queue.py",
             root / "scripts/run_geometry.py", plan]
    for folder in (root / "geometry", root / "structural_kge", root / "upstream/vl-kge/vlkge"):
        if folder.is_dir():
            paths += sorted(folder.rglob("*.py"))
    for folder in (root / "vendor/sl-manifold-core/src/sl_manifold", root.parent / "sl-manifold-core/src/sl_manifold"):
        if folder.is_dir():
            paths += sorted(folder.glob("*.py"))
            break
    # Dataset byte verification is independently performed by the trainer.
    return {str(path): digest(path) for path in paths if path.is_file()}


def runner_command(root, run_dir, config):
    command = [sys.executable, "-u", str(root / "scripts/run_wn9_geometry_v2.py"),
               "--root", str(root), "--run-dir", str(run_dir), "--validation-only"]
    for key, value in config.items():
        if value is not None:
            command.extend(("--" + key.replace("_", "-"), str(value)))
    return command


def checked_result(run_dir, job):
    result = json.loads((run_dir / "result.json").read_text())
    if (result.get("status") != "completed_validation_only" or result.get("test_evaluated") is not False
            or "final_test" in result or result.get("model") != job["config"]["model"]
            or result.get("evaluation_protocol") != PROTOCOL
            or result.get("training_negative_filter") != "train_only_directional"
            or result.get("validation_filter") != "all_splits_directional"
            or result.get("tie_policy") != "realistic_average"
            or result.get("negative_sampling") != "uniform_without_replacement_per_triple"
            or result.get("score_distance_power") != 1 or result.get("optimizer") != "Adagrad"
            or result.get("dataset") != "WN9-IMG"):
        raise ValueError("Result violates the requested test-free WN9 V2 protocol")
    actual = result.get("config", result.get("args"))
    if not isinstance(actual, dict) or any(actual.get(k) != v for k, v in job["config"].items()):
        raise ValueError("Effective trainer configuration differs from the saved queue configuration")
    score, epoch, completed = (result.get(k) for k in ("best_validation_mrr", "best_epoch", "completed_epochs"))
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Invalid validation MRR")
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (epoch, completed)) or not 1 <= epoch <= completed <= job["config"]["epochs"]:
        raise ValueError("Invalid completed/best epoch provenance")
    count, total = result.get("validation_count"), result.get("validation_total_count")
    expected = min(job["config"]["eval_limit"] or 1337, 1337)
    if count != expected or total != 1337 or result.get("validation_scope") != ("full" if expected == 1337 else "subset"):
        raise ValueError("Validation scope/count mismatch")
    indices = result.get("validation_indices")
    if (not isinstance(indices, list) or len(indices) != count
            or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < total for i in indices)
            or len(set(indices)) != count):
        raise ValueError("Missing or invalid ordered validation indices")
    expected_hash = hashlib.sha256(struct.pack("<" + "q" * len(indices), *indices)).hexdigest()
    if result.get("subset_indices_hash") != expected_hash:
        raise ValueError("Validation index hash mismatch")
    if job["config"]["kind"] == "pilot" and result.get("training_complete") is not True:
        raise ValueError("Pilot result contains truncated training")
    if not (run_dir / "best.pt").is_file():
        raise ValueError("Completed result is missing best.pt")
    return result


def backup_worker(source, log, destination, queue_state):
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name("." + destination.name + ".partial")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(temporary)
    shutil.copytree(source, temporary, symlinks=True, ignore=shutil.ignore_patterns("*.tmp"))
    shutil.copy2(log, temporary / "train.log")
    shutil.copy2(queue_state, temporary / "queue_snapshot.json")
    os.rename(temporary, destination)


def stop_child(process, remaining):
    """Signal only the actual Popen child; never inspect or kill other GPU users."""
    if process is None or process.poll() is not None:
        return
    try:
        process.send_signal(signal.SIGINT)
        process.wait(timeout=max(0, min(SHUTDOWN_GRACE_SECONDS, remaining() - 2)))
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=max(0, min(2, remaining())))
        except subprocess.TimeoutExpired:
            pass  # Child was killed; its inherited lock prevents overlap until exit.


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--queue-id", type=name, required=True)
    parser.add_argument("--deadline-utc", type=parse_utc, required=True)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--stop-after-current", type=Path,
                        help="Control file: existence stops before the next trial, after backup. Default: queue/STOP_AFTER_CURRENT")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root, plan = args.root.resolve(), args.plan.resolve()
    if not (root / "scripts/run_wn9_geometry_v2.py").is_file():
        raise FileNotFoundError("Missing WN9 V2 trainer")
    raw_plan, jobs = load_plan(plan, args.deadline_utc)
    started = time.monotonic()
    budget = (args.deadline_utc - datetime.now(timezone.utc)).total_seconds()
    if budget <= TRAIN_RESERVE_SECONDS:
        raise ValueError("UTC deadline expired or leaves no bounded shutdown/backup reserve")
    monotonic_deadline = started + budget

    def remaining():
        # A backwards wall-clock adjustment must not grant extra running time.
        return min(monotonic_deadline - time.monotonic(),
                   (args.deadline_utc - datetime.now(timezone.utc)).total_seconds())

    directory = root / "runs" / args.queue_id
    backup = args.backup_root.resolve() / args.queue_id
    if backup == directory or backup in directory.parents or directory in backup.parents:
        raise ValueError("Persistent backup must be separate from the local queue tree")
    control = args.stop_after_current.resolve() if args.stop_after_current else directory / "STOP_AFTER_CURRENT"
    if control == plan:
        raise ValueError("Control file must not overwrite the plan")
    fingerprints = source_fingerprints(root, plan)
    (root / "runs").mkdir(exist_ok=True)
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    process, current = None, None

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Queue received signal {signum}")

    try:
        for sig in handlers:
            signal.signal(sig, interrupt)
        with (root / "runs/gpu0.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if directory.exists() or directory.is_symlink() or backup.exists() or backup.is_symlink():
                raise FileExistsError("Existing queue/backup is never resumed or overwritten; use a new ID")
            directory.mkdir()
            save_json(directory / "plan.json", raw_plan)
            for job in jobs:
                job["run_dir"] = str(directory / job["job_id"])
            state = {"schema_version": 1, "status": "running", "started_at": utc(), "pid": os.getpid(),
                     "deadline_utc": args.deadline_utc.isoformat(), "initial_remaining_seconds": budget,
                     "source_sha256": fingerprints, "plan_sha256": digest(plan), "backup_root": str(backup),
                     "stop_after_current_file": str(control), "test_scoring_allowed": False,
                     "profile_gate": "success only; no automatic promotions or assurance all jobs fit",
                     "deadline_policy": "reserve 45 seconds for owned-child shutdown and bounded checkpoint backup",
                     "jobs": jobs, "validation_records": []}

            def save():
                state["elapsed_seconds"] = time.monotonic() - started
                state["remaining_seconds"] = max(0, remaining())
                state["counts"] = {label: sum(j["status"] == label for j in jobs)
                                   for label in ("not_started", "running", "completed", "failed", "interrupted", "deadline_reached")}
                state["validation_records"] = [{"job_id": j["job_id"], "stage": j["stage"],
                                                "config": j["config"], "result": j["validated_result"],
                                                "backup_status": j.get("backup_status")}
                                               for j in jobs if j["status"] == "completed"]
                save_json(directory / "queue.json", state)

            def finish(status, code, error=None):
                state.update(status=status, ended_at=utc())
                if error is not None:
                    state["error"] = repr(error)
                save()
                return code

            def run_backup(job, suffix=""):
                nonlocal process
                source, log = Path(job["run_dir"]), Path(job["log"])
                if not source.is_dir() or remaining() <= 2:
                    job["backup_status"] = "unavailable_or_deadline"
                    return False
                destination = backup / (job["job_id"] + suffix)
                command = [sys.executable, str(Path(__file__).resolve()), "--_backup-worker",
                           str(source), str(log), str(destination), str(directory / "queue.json")]
                job.update(backup_status="running", backup_path=str(destination))
                save()
                with (directory / (job["job_id"] + suffix + ".backup.log")).open("x") as stream:
                    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                               cwd=root, pass_fds=(lock.fileno(),))
                    code = process.wait(timeout=max(0, remaining() - 2))
                process = None
                if code:
                    raise RuntimeError(f"Backup exited {code}; local files retained")
                job["backup_status"] = "completed"
                save()
                return True

            save()
            try:
                for job in jobs:
                    current = job
                    if control.exists():
                        return finish("stopped_after_current", 0)
                    if remaining() <= TRAIN_RESERVE_SECONDS:
                        return finish("deadline_reached", 124)
                    if source_fingerprints(root, plan) != fingerprints:
                        raise RuntimeError("Source or plan changed; refusing mixed-source trials")
                    run_dir = Path(job["run_dir"])
                    if run_dir.exists() or run_dir.is_symlink():
                        raise FileExistsError(run_dir)
                    command = runner_command(root, run_dir, job["config"])
                    log = directory / (job["job_id"] + ".log")
                    job.update(status="running", started_at=utc(), command=command, log=str(log))
                    save()
                    environment = os.environ.copy()
                    environment.setdefault("CUDA_VISIBLE_DEVICES", "0")
                    environment.update(PYTHONHASHSEED=str(job["config"]["seed"]), PYTHONUNBUFFERED="1")
                    with log.open("x") as stream:
                        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                                   cwd=root, env=environment, pass_fds=(lock.fileno(),))
                        job["pid"] = process.pid
                        save()
                        code = process.wait(timeout=max(0, remaining() - TRAIN_RESERVE_SECONDS))
                    process = None
                    job["exit_code"] = code
                    if code:
                        raise RuntimeError(f"Trainer exited {code}; startup failures may have no status.json")
                    if source_fingerprints(root, plan) != fingerprints:
                        raise RuntimeError("Source or plan changed during this trial")
                    result = checked_result(run_dir, job)
                    for previous in jobs:
                        previous_result = previous.get("validated_result")
                        if (previous_result and previous["config"]["validation_seed"] == job["config"]["validation_seed"]
                                and previous_result["validation_count"] == result["validation_count"]
                                and previous_result["subset_indices_hash"] != result["subset_indices_hash"]):
                            raise ValueError("Matched trials used different ordered validation subsets")
                    job.update(status="completed", ended_at=utc(), validated_result=result, backup_status="pending")
                    save()
                    if not run_backup(job):
                        return finish("backup_failed", 1)
                    print(f"COMPLETED {job['job_id']}: {result['validation_scope']} validation MRR={result['best_validation_mrr']:.8f}; test not evaluated; backed up", flush=True)
                return finish("completed", 0)
            except BaseException as error:
                for sig in handlers:
                    signal.signal(sig, signal.SIG_IGN)
                was_backup = bool(current and current.get("backup_status") == "running")
                stop_child(process, remaining)
                process = None
                timed_out = isinstance(error, subprocess.TimeoutExpired)
                interrupted = isinstance(error, KeyboardInterrupt)
                label = "deadline_reached" if timed_out else "interrupted" if interrupted else "backup_failed" if was_backup else "failed"
                if current:
                    current["error"] = repr(error)
                    if was_backup:
                        current["backup_status"] = label
                    elif current["status"] == "running":
                        current.update(status=label, ended_at=utc())
                    save()
                    # Salvage an interrupted/failed best checkpoint within the SAME deadline.
                    if not was_backup and current.get("log") and Path(current["run_dir"]).is_dir():
                        try:
                            run_backup(current, "-partial")
                        except BaseException as backup_error:
                            stop_child(process, remaining)
                            current.update(backup_status="failed", backup_error=repr(backup_error))
                return finish(label, 124 if timed_out else 130 if interrupted else 1, error)
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    if len(sys.argv) == 6 and sys.argv[1] == "--_backup-worker":
        backup_worker(*(Path(value) for value in sys.argv[2:]))
    else:
        raise SystemExit(main())
