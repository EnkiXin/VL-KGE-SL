"""Bounded validation-only structural KGE queue; reuses WN9 queue safety helpers.

Plans list explicit smoke/profile/screen jobs. Dataset-dependent validation
scope, optimization steps and evaluation cadence are checked before acceptance.
No resume, test scoring, automatic promotion or extension of the UTC deadline.
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
import signal
import struct
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_wn9_geometry_v2_queue as common_queue

utc = common_queue.utc
parse_utc = common_queue.parse_utc
name = common_queue.name
digest = common_queue.digest
save_json = common_queue.save_json
backup_worker = common_queue.backup_worker
stop_child = common_queue.stop_child
TRAIN_RESERVE_SECONDS = common_queue.TRAIN_RESERVE_SECONDS
MODELS = common_queue.MODELS
PROTOCOL = "structural_directional_v1"
DATASETS = {
    "WN18RR": (40943, 11, 86835, 3034, 3134),
    "FB15k-237": (14541, 237, 272115, 17535, 20466),
}
DEFAULTS = {
    "kind": "pilot", "epochs": 15, "eval_every": 5, "patience": 50, "seed": 42,
    "lr": .01, "chart_radius": 1.5, "target_init_norm": .5,
    "initial_logit_scale": 1.0, "initial_offset": 0.0,
    "batch_size": 512, "negatives": 100, "score_chunk": 1024,
    "query_batch": 32, "candidate_chunk": 256, "grad_clip": 5.0,
    "log_backend": "gregory12", "log_order": 16, "validation_seed": 260906,
    "max_train_batches": None, "eval_limit": None,
}


def load_plan(path, deadline, root=None):
    root = Path.cwd() if root is None else root
    raw = json.loads(path.read_text())
    if isinstance(raw, list):
        raw = {"schema_version": 1, "jobs": raw}
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("Expected schema_version=1 and explicit jobs")
    if "deadline_utc" in raw and parse_utc(raw["deadline_utc"]) != deadline:
        raise ValueError("CLI deadline differs from the authorized plan deadline")
    common, entries = raw.get("common", {}), raw.get("jobs")
    keys = set(DEFAULTS) | {"data_root", "dataset"}
    if not isinstance(common, dict) or set(common) - keys:
        raise ValueError("Unknown common trainer settings")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Plan has no jobs")
    jobs = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) - (keys | {"job_id", "stage", "model"}):
            raise ValueError("Unknown job fields; arbitrary commands are forbidden")
        config = {**DEFAULTS, **common, **{k: v for k, v in entry.items() if k in keys}}
        model, stage = entry.get("model"), entry.get("stage")
        if model not in MODELS or stage not in ("smoke", "profile", "screen"):
            raise ValueError("Invalid structural model/stage")
        config["model"] = model
        if config.get("dataset") not in DATASETS:
            raise ValueError("Dataset must be WN18RR or FB15k-237")
        data_root = config.get("data_root")
        if not isinstance(data_root, str) or not data_root:
            raise ValueError("Explicit data_root is required")
        data_root = Path(data_root).expanduser()
        config["data_root"] = str((data_root if data_root.is_absolute() else root / data_root).resolve())
        if config["kind"] not in ("smoke", "pilot"):
            raise ValueError("Only validation-only smoke/pilot jobs are permitted")
        if stage == "screen" and config["kind"] != "pilot":
            raise ValueError("Screen requires full training and complete validation")
        for key in ("epochs", "eval_every", "patience", "batch_size", "negatives", "score_chunk", "query_batch", "candidate_chunk", "log_order"):
            if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if config["log_backend"] != "gregory12" or config["log_order"] != 16:
            raise ValueError("This queue uses Gregory12 only; log_order=16 is an inactive compatibility setting")
        for key in ("seed", "validation_seed"):
            if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        for key in ("lr", "chart_radius", "target_init_norm", "initial_logit_scale", "grad_clip"):
            if isinstance(config[key], bool) or not isinstance(config[key], (int, float)) or not math.isfinite(config[key]) or config[key] <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if config["target_init_norm"] >= config["chart_radius"]:
            raise ValueError("Initialization norm must be below the chart radius")
        if isinstance(config["initial_offset"], bool) or not isinstance(config["initial_offset"], (int, float)) or not math.isfinite(config["initial_offset"]):
            raise ValueError("initial_offset must be finite")
        for key in ("max_train_batches", "eval_limit"):
            v = config[key]
            if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v <= 0 or config["kind"] != "smoke"):
                raise ValueError(f"{key} is a positive smoke-only option")
        jobs.append({"job_id": name(entry.get("job_id")), "stage": stage, "config": config, "status": "not_started"})
    if len({job["job_id"] for job in jobs}) != len(jobs):
        raise ValueError("Duplicate job IDs")
    order = {"smoke": 0, "profile": 1, "screen": 2}
    if [order[j["stage"]] for j in jobs] != sorted(order[j["stage"]] for j in jobs):
        raise ValueError("Run smoke/profile before screen")
    return raw, jobs


def source_fingerprints(root, plan):
    result = common_queue.source_fingerprints(root, plan)
    for path in (root / "scripts/run_structural_geometry.py",
                 root / "scripts/run_structural_geometry_queue.py"):
        if path.is_file():
            result[str(path)] = digest(path)
    # The shared helper includes structural/geometry modules, WN9 runner,
    # old input/loss helpers, the author Python sources, and shared SL core.
    return result


def runner_command(root, run_dir, config):
    command = [sys.executable, "-u", str(root / "scripts/run_structural_geometry.py"),
               "--root", str(root), "--run-dir", str(run_dir), "--validation-only"]
    for key, value in config.items():
        if value is not None:
            command.extend(("--" + key.replace("_", "-"), str(value)))
    return command


def checked_result(run_dir, job):
    result = json.loads((run_dir / "result.json").read_text())
    cfg = job["config"]
    entities, relations, train_count, valid_count, test_count = DATASETS[cfg["dataset"]]
    if (result.get("status") != "completed_validation_only" or result.get("test_evaluated") is not False
            or "final_test" in result or result.get("dataset") != cfg["dataset"]
            or result.get("model") != cfg["model"] or result.get("evaluation_protocol") != PROTOCOL
            or result.get("training_negative_filter") != "train_only_directional"
            or result.get("validation_filter") != "all_splits_directional"
            or result.get("tie_policy") != "realistic_average"
            or result.get("negative_sampling") != "uniform_without_replacement_per_triple"
            or result.get("score_distance_power") != 1 or result.get("optimizer") != "Adagrad"):
        raise ValueError("Result violates the requested test-free structural protocol")
    actual = result.get("config", result.get("args"))
    if not isinstance(actual, dict) or any(actual.get(k) != v for k, v in cfg.items()):
        raise ValueError("Effective trainer configuration differs from the queue snapshot")
    if (result.get("log_backend") != "gregory12" or result.get("log_terms") != 12
            or result.get("entity_bias") is not False or result.get("coordinate_dim") != 63):
        raise ValueError("Result changed the Gregory12, linear 63-D no-entity-bias model contract")
    diagnostic_rows = (result.get("initial_diagnostics"), result.get("last_epoch", {}).get("diagnostics"),
                       result.get("best_checkpoint_diagnostics"))
    for diagnostic in diagnostic_rows:
        contract = diagnostic.get("model_contract", {}) if isinstance(diagnostic, dict) else {}
        if (not isinstance(diagnostic, dict) or diagnostic.get("log_backend") != "gregory12"
                or diagnostic.get("geometry") != cfg["model"] or diagnostic.get("sampled_health_passed") is not True
                or contract.get("log_backend") != "gregory12" or contract.get("log_terms") != 12
                or contract.get("entity_bias") is not False or contract.get("distance_power") != 1
                or contract.get("dimension") != 63):
            raise ValueError("Missing/unsafe diagnostic or mixed geometry/backend provenance")
        if cfg["model"] == "sl8" and diagnostic.get("principal_log", {}).get("backend") != "gregory12":
            raise ValueError("SL diagnostic backend is inconsistent")
    if (result.get("num_entities") != entities or result.get("num_relations") != relations
            or result.get("split_sizes") != [train_count, valid_count, test_count]
            or result.get("train_count") != train_count):
        raise ValueError("Pinned dataset entity/relation/split counts differ")
    expected_parameters = (entities + relations) * 63 + 2
    if result.get("parameters_trainable") != expected_parameters or result.get("parameters_total") != expected_parameters:
        raise ValueError("Structural parameter budget differs from direct 63-D tables plus two scalars")
    provenance = result.get("dataset_provenance", {})
    if (result.get("dataset_commit") != "2e440e0f9c687314d5ff67ead68ce985dc446e3a"
            or provenance.get("commit") != result["dataset_commit"] or provenance.get("dataset") != cfg["dataset"]):
        raise ValueError("Missing or unexpected pinned data provenance")
    score, epoch, completed = (result.get(k) for k in ("best_validation_mrr", "best_epoch", "completed_epochs"))
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Invalid validation MRR")
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (epoch, completed)) or not 1 <= epoch <= completed <= cfg["epochs"]:
        raise ValueError("Invalid epoch provenance")
    expected_valid = min(cfg["eval_limit"] or valid_count, valid_count)
    if (result.get("validation_count") != expected_valid or result.get("validation_total_count") != valid_count
            or result.get("validation_scope") != ("full" if expected_valid == valid_count else "subset")):
        raise ValueError("Dataset-dependent validation scope/count mismatch")
    indices = result.get("validation_indices")
    if (not isinstance(indices, list) or len(indices) != expected_valid or len(set(indices)) != len(indices)
            or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < valid_count for i in indices)):
        raise ValueError("Invalid ordered validation indices")
    index_hash = hashlib.sha256(struct.pack("<" + "q" * len(indices), *indices)).hexdigest()
    if result.get("subset_indices_hash") != index_hash:
        raise ValueError("Validation index hash mismatch")
    expected_validated = [e for e in range(1, completed + 1) if e % cfg["eval_every"] == 0 or e == cfg["epochs"]]
    if not expected_validated or result.get("validation_epochs") != expected_validated or epoch not in expected_validated:
        raise ValueError("Evaluation cadence differs from every-N plus final epoch")
    full_batches = math.ceil(train_count / cfg["batch_size"])
    batches = min(cfg["max_train_batches"] or full_batches, full_batches)
    train_complete = batches == full_batches
    if result.get("training_complete") is not train_complete:
        raise ValueError("Truncated training label is inaccurate")
    events = [json.loads(line) for line in (run_dir / "epochs.jsonl").read_text().splitlines() if line.strip()]
    if len(events) != completed or [e.get("epoch") for e in events] != list(range(1, completed + 1)):
        raise ValueError("Missing or duplicated epoch records")
    steps = 0
    for event in events:
        if event.get("train_batches") != batches or event.get("train_examples") != min(train_count, batches * cfg["batch_size"]):
            raise ValueError("Recorded training batches/examples differ from the plan")
        validation = event.get("validation")
        if event["epoch"] in expected_validated:
            if (not isinstance(validation, dict) or validation.get("evaluated_triples") != expected_valid
                    or validation.get("num_entities") != entities or validation.get("directional_queries") != 2 * expected_valid):
                raise ValueError("Validation event used an incomplete query/candidate scope")
        elif validation is not None:
            raise ValueError("Unexpected extra validation event")
        steps += event["train_batches"]
    if result.get("total_optimizer_steps") != steps:
        raise ValueError("Optimizer step total is inconsistent with the epoch log")
    if not (run_dir / "best.pt").is_file():
        raise ValueError("Completed validation result has no best.pt")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--queue-id", type=name, required=True)
    parser.add_argument("--deadline-utc", type=parse_utc, required=True)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--stop-after-current", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root, plan = args.root.resolve(), args.plan.resolve()
    if not (root / "scripts/run_structural_geometry.py").is_file():
        raise FileNotFoundError("Missing structural geometry trainer")
    raw_plan, jobs = load_plan(plan, args.deadline_utc, root)
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
                        if (previous_result and previous["config"]["dataset"] == job["config"]["dataset"]
                                and previous["config"]["validation_seed"] == job["config"]["validation_seed"]
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
