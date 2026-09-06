"""Execute prepared DistMult residual trials under one explicit wall-clock budget.

Only validation-only jobs from the checked planner schema are accepted. Multiple
manifests run in order under ONE deadline, so profiling can gate the anchors.
Existing queues are never resumed or overwritten; no final-test trials launch.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
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


METHODS = ("baseline", "euclidean", "sl8")


def utc():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def simple_name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise argparse.ArgumentTypeError("must be a simple directory name")
    return value


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0 or not math.isfinite(number * 3600):
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def script_module(filename):
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location("_queue_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_config(base, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Manifest config paths must be safe relative paths")
    path = base / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError("Missing or symbolic-link manifest config")
    path.resolve().relative_to(base.resolve())
    return json.loads(path.read_text())


def load_manifest(path):
    path = path.resolve()
    manifest = json.loads(path.read_text())
    planner = script_module("plan_distmult_sl_hpo.py")
    runner = script_module("run_distmult_sl_author.py")
    spec = planner.checked_spec(path.parent / "spec.json")
    if spec.get("dataset") != "wn9_img" or spec.get("author_config") != runner.AUTHOR_CONFIG:
        raise ValueError("Executor accepts only the locked WN9 DistMult author configuration")
    digest = hashlib.sha256(canonical(spec).encode()).hexdigest()
    if (manifest.get("schema_version") != 1 or manifest.get("status") != "prepared_not_launched"
            or manifest.get("selection_metric") != "best_validation_mrr"
            or manifest.get("auto_launch") is not False
            or manifest.get("spec_sha256") != digest
            or manifest.get("experiment") != spec["experiment"]):
        raise ValueError("Manifest metadata does not match its prepared specification")
    stage = manifest.get("stage")
    expected = planner.planned_jobs(spec, stage)
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != len(expected) or not jobs:
        raise ValueError("Manifest job count differs from the deterministic planner")
    for job, reference in zip(jobs, expected):
        if any(job.get(key) != value for key, value in reference.items()):
            raise ValueError("Manifest jobs differ from the deterministic matched plan")
        simple_name(job["job_id"])
        overrides = safe_config(path.parent, job["author_overrides_path"])
        extension = safe_config(path.parent, job["extension_config_path"])
        if overrides != job["author_overrides"] or extension != job["extension_config"]:
            raise ValueError("Manifest config file content differs from its declared job")
        runner.validate_author_overrides(overrides)
        if runner.validate_extension_config(job["method"], extension) != extension:
            raise ValueError("Extension config must contain the complete resolved defaults")
    return {"source_path": str(path), "source_sha256": file_hash(path), "manifest": manifest,
            "spec": spec, "stage": stage, "spec_sha256": digest}


def runner_command(root, queue, job):
    config = queue / "configs" / job["job_id"]
    return [sys.executable, "-u", str(root / "scripts/run_distmult_sl_author.py"),
            "--repo", str(root / "upstream/vl-kge"), "--run-dir", str(queue / job["job_id"]),
            "--method", job["method"], "--validation-only",
            "--author-overrides", str(config / "author_overrides.json"),
            "--extension-config", str(config / "extension.json")]


def checked_result(run_dir, job):
    result = json.loads((run_dir / "result.json").read_text())
    if (result.get("status") != "completed_validation_only" or result.get("test_evaluated") is not False
            or "final_test" in result or result.get("method") != job["method"]
            or result.get("author_overrides") != job["author_overrides"]
            or result.get("extension_config") != job["extension_config"]):
        raise ValueError("Result is not the requested, completed, test-free validation trial")
    score, epoch = result.get("best_validation_mrr"), result.get("best_epoch")
    completed = result.get("completed_epochs")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Invalid validation MRR")
    if (isinstance(epoch, bool) or not isinstance(epoch, int) or isinstance(completed, bool)
            or not isinstance(completed, int) or not 1 <= epoch <= completed <= job["author_overrides"]["epochs"]):
        raise ValueError("Invalid completed/best epoch provenance")
    if not (run_dir / "best.pt").is_file():
        raise ValueError("Completed result has no best checkpoint")
    return result


def selection_records(jobs):
    records = []
    for job in jobs:
        if job["status"] == "completed":
            records.append({"candidate_id": job["candidate_id"], "method": job["method"],
                            "stage": job["stage"], "seed": job["author_overrides"]["seed"],
                            "spec_sha256": job["spec_sha256"], "status": "completed_validation_only",
                            "test_evaluated": False, "author_overrides": job["author_overrides"],
                            "best_validation_mrr": job["best_validation_mrr"], "best_epoch": job["best_epoch"],
                            "run_dir": job["run_dir"], "backup_status": job.get("backup_status")})
    return records


def stop_child(process):
    """Only signal the Popen child owned by this queue, never a GPU-wide PID list."""
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
        process.wait(timeout=5)


def backup_worker(source, log, destination, queue_state):
    """Invoked in a timed, owned subprocess so a slow NFS cannot hang the queue."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name("." + destination.name + ".partial")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(temporary)
    shutil.copytree(source, temporary)
    shutil.copy2(log, temporary / "train.log")
    shutil.copy2(queue_state, temporary / "queue_snapshot.json")
    os.rename(temporary, destination)


def source_fingerprints(root):
    paths = [root / "scripts/run_distmult_sl_author.py", root / "scripts/run_author.py",
             root / "scripts/run_distmult_sl_queue.py", root / "scripts/plan_distmult_sl_hpo.py",
             root / "geometry/distmult_sl.py", root / "geometry/__init__.py", root / "geometry/models.py",
             root / "artifacts/wn9-inputs.json"]
    author = root / "upstream/vl-kge/vlkge"
    if author.is_dir():
        paths += sorted(author.rglob("*.py"))
        paths += sorted((author / "configs").rglob("*.yaml"))
    for core in (root / "vendor/sl-manifold-core/src/sl_manifold", root.parent / "sl-manifold-core/src/sl_manifold"):
        if core.is_dir():
            paths += sorted(core.glob("*.py"))
            break
    return {str(path): file_hash(path) for path in paths if path.is_file()}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, nargs="+", required=True,
                        help="One or more prepared manifests, executed in the given order under one deadline")
    parser.add_argument("--queue-id", type=simple_name, required=True)
    parser.add_argument("--max-hours", type=positive_float, default=8.0, required=True,
                        help="Explicit finite total wall budget; recommended first batch: 8")
    parser.add_argument("--backup-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = args.root.resolve()
    if not (root / "scripts/run_distmult_sl_author.py").is_file():
        raise FileNotFoundError("Missing DistMult residual author launcher")
    plans = [load_manifest(path) for path in args.manifest]
    if len({plan["spec_sha256"] for plan in plans}) != 1 or len({plan["stage"] for plan in plans}) != len(plans):
        raise ValueError("Ordered manifests must share one specification and have distinct stages")
    stage_order = {"profile": 0, "anchors": 1, "coarse": 2}
    if [stage_order[plan["stage"]] for plan in plans] != sorted(stage_order[plan["stage"]] for plan in plans):
        raise ValueError("Manifests must run profile before anchors before coarse")
    queue = root / "runs" / args.queue_id
    backup = args.backup_root.resolve() / args.queue_id
    if backup == queue or queue in backup.parents or backup in queue.parents:
        raise ValueError("Backup destination must be separate from the local queue tree")
    jobs = []
    for plan in plans:
        for job in plan["manifest"]["jobs"]:
            jobs.append({**job, "stage": plan["stage"], "spec_sha256": plan["spec_sha256"],
                         "run_dir": str(queue / job["job_id"]), "status": "not_started"})
    if len({job["job_id"] for job in jobs}) != len(jobs):
        raise ValueError("Duplicate queue job IDs")
    fingerprints = source_fingerprints(root)
    (root / "runs").mkdir(exist_ok=True)
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Queue received signal {signum}")

    process = None
    current = None
    state = None
    started = time.monotonic()
    deadline = started + args.max_hours * 3600
    try:
        for sig in handlers:
            signal.signal(sig, interrupted)
        with (root / "runs/gpu0.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Do not touch an earlier successful or partial queue, or its backups.
            if queue.exists() or queue.is_symlink() or backup.exists() or backup.is_symlink():
                raise FileExistsError("Queue or persistent backup already exists; use a new queue ID")
            queue.mkdir(exist_ok=False)
            for job in jobs:
                config = queue / "configs" / job["job_id"]
                config.mkdir(parents=True)
                save_json(config / "author_overrides.json", job["author_overrides"])
                save_json(config / "extension.json", job["extension_config"])
            save_json(queue / "input_plans.json", plans)
            state = {"status": "running", "pid": os.getpid(), "started_at": utc(),
                     "max_hours": args.max_hours, "stages": [plan["stage"] for plan in plans],
                     "backup_root": str(backup), "source_sha256": fingerprints,
                     "selection_metric": "best_validation_mrr", "test_scoring_allowed": False,
                     "profile_gate": "success_and_numeric_health_only; no prediction that all anchor trials will fit the remaining budget",
                     "budget_scope": "all listed trials and per-job backups; shutdown may need up to 25 additional seconds",
                     "jobs": jobs, "validation_records": []}

            def save():
                state["elapsed_seconds"] = time.monotonic() - started
                state["validation_records"] = selection_records(jobs)
                save_json(queue / "queue.json", state)

            def finish(status, code, error=None):
                state.update(status=status, ended_at=utc())
                if error is not None:
                    state["error"] = repr(error)
                save()
                return code

            save()
            try:
                for job in jobs:
                    current = job
                    if time.monotonic() >= deadline:
                        return finish("budget_exhausted", 124)
                    if source_fingerprints(root) != fingerprints:
                        raise RuntimeError("Training source changed during the queue; refusing mixed-source HPO")
                    run_dir = Path(job["run_dir"])
                    if run_dir.exists() or run_dir.is_symlink():
                        raise FileExistsError(run_dir)
                    command = runner_command(root, queue, job)
                    log = queue / (job["job_id"] + ".log")
                    job.update(status="running", started_at=utc(), command=command, log=str(log))
                    save()
                    environment = os.environ.copy()
                    environment.setdefault("CUDA_VISIBLE_DEVICES", "0")
                    environment.update(PYTHONHASHSEED=str(job["author_overrides"]["seed"]), PYTHONUNBUFFERED="1")
                    with log.open("x") as stream:
                        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                                   cwd=root, env=environment, pass_fds=(lock.fileno(),))
                        job["pid"] = process.pid
                        save()
                        code = process.wait(timeout=max(0, deadline - time.monotonic()))
                    job["exit_code"] = code
                    if code:
                        raise RuntimeError(f"Trial exited {code}; inspect its log (startup may produce no status.json)")
                    result = checked_result(run_dir, job)
                    process = None
                    job.update(status="completed", ended_at=utc(), best_epoch=result["best_epoch"],
                               best_validation_mrr=result["best_validation_mrr"],
                               completed_epochs=result["completed_epochs"], backup_status="pending")
                    save()
                    if time.monotonic() >= deadline:
                        job["backup_status"] = "not_started_budget_exhausted"
                        return finish("budget_exhausted", 124)
                    backup_destination = backup / job["job_id"]
                    backup_command = [sys.executable, str(Path(__file__).resolve()), "--_backup-worker",
                                      str(run_dir), str(log), str(backup_destination), str(queue / "queue.json")]
                    job.update(backup_status="running", backup_path=str(backup_destination))
                    save()
                    with (queue / (job["job_id"] + ".backup.log")).open("x") as stream:
                        process = subprocess.Popen(backup_command, stdout=stream, stderr=subprocess.STDOUT,
                                                   cwd=root, pass_fds=(lock.fileno(),))
                        code = process.wait(timeout=max(0, deadline - time.monotonic()))
                    if code:
                        raise RuntimeError(f"Persistent backup failed with exit {code}; local result/checkpoint retained")
                    process = None
                    job["backup_status"] = "completed"
                    save()
                    print(f"COMPLETED {job['job_id']}: validation MRR={job['best_validation_mrr']:.8f}; backed up; test not evaluated", flush=True)
                return finish("completed", 0)
            except (KeyboardInterrupt, subprocess.TimeoutExpired) as error:
                for sig in handlers:
                    signal.signal(sig, signal.SIG_IGN)
                stop_child(process)
                timed_out = isinstance(error, subprocess.TimeoutExpired)
                label = "budget_exhausted" if timed_out else "interrupted"
                if current is not None:
                    if current.get("backup_status") == "running":
                        current["backup_status"] = label
                    elif current["status"] == "running":
                        current.update(status=label, ended_at=utc())
                return finish(label, 124 if timed_out else 130, error)
            except Exception as error:
                stop_child(process)
                label = "backup_failed" if current and current.get("backup_status") == "running" else "failed"
                if current is not None:
                    if label == "backup_failed":
                        current["backup_status"] = "failed"
                    elif current["status"] == "running":
                        current.update(status="failed", ended_at=utc(), error=repr(error))
                return finish(label, 1, error)
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    if len(sys.argv) == 6 and sys.argv[1] == "--_backup-worker":
        backup_worker(*(Path(value) for value in sys.argv[2:]))
    else:
        raise SystemExit(main())
