"""Single-use, validation-only follow-on gate under an unchanged UTC deadline."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_structural_geometry_queue as queue


def completed_records(root, directory, deadline, stage, count):
    state = json.loads((directory / "queue.json").read_text())
    if state.get("status") != "completed" or queue.parse_utc(state["deadline_utc"]) != deadline:
        raise ValueError("Prerequisite queue is not completed under the same deadline")
    old_sources = state["source_sha256"]
    plans = [Path(p) for p, sha in old_sources.items() if p.endswith(".json") and sha == state["plan_sha256"]]
    if len(plans) != 1 or queue.source_fingerprints(root, plans[0]) != old_sources:
        raise ValueError("Prerequisite source/plan fingerprint changed")
    raw, jobs = queue.load_plan(directory / "plan.json", deadline, root)
    if raw != json.loads(plans[0].read_text()) or len(jobs) != count or len(state["jobs"]) != count:
        raise ValueError("Prerequisite plan snapshot/count mismatch")
    records = {}
    for expected, job in zip(jobs, state["jobs"]):
        if any(expected[k] != job[k] for k in ("job_id", "stage", "config")) or job["stage"] != stage:
            raise ValueError("Prerequisite job differs from its frozen plan")
        run_dir, backup = directory / job["job_id"], Path(job.get("backup_path", ""))
        if job["status"] != "completed" or job.get("backup_status") != "completed" or Path(job["run_dir"]) != run_dir:
            raise ValueError("Prerequisite job is not completed and backed up")
        result = queue.checked_result(run_dir, job)
        if any(not (backup / f).is_file() or queue.digest(backup / f) != queue.digest(run_dir / f)
               for f in ("result.json", "best.pt")):
            raise ValueError("Prerequisite persistent backup differs or is missing")
        cfg = job["config"]
        if cfg["epochs"] != 1 or (stage == "profile" and (cfg["model"] != "sl8" or not result["training_complete"] or result["validation_scope"] != "full")):
            raise ValueError("Expected one complete SL profile epoch")
        if stage == "smoke" and (cfg["max_train_batches"] != 3 or cfg["eval_limit"] != 32):
            raise ValueError("Expected three-batch, 32-triple smoke")
        if cfg["model"] == "sl8" and any(d.get("sampled_reference_accuracy_passed") is not True for d in
                (result["initial_diagnostics"], result["last_epoch"]["diagnostics"], result["best_checkpoint_diagnostics"])):
            raise ValueError("SL sampled SciPy reference accuracy did not pass")
        key = (cfg["dataset"], cfg["model"])
        if key in records:
            raise ValueError("Duplicate prerequisite dataset/model")
        records[key] = (cfg, result)
    return records


def estimate(jobs, smoke, profile):
    expected = {(d, m) for d in queue.DATASETS for m in queue.MODELS}
    if len(jobs) != 6 or {(j["config"]["dataset"], j["config"]["model"]) for j in jobs} != expected or set(smoke) != expected or set(profile) != {(d, "sl8") for d in queue.DATASETS}:
        raise ValueError("Expected exactly six matched jobs and complete prerequisite coverage")
    ignored = {"epochs", "eval_every", "kind", "max_train_batches", "eval_limit"}
    costs = []
    for job in jobs:
        cfg = job["config"]
        if job["stage"] != "screen" or cfg["epochs"] != 20 or cfg["eval_every"] != 5 or cfg["chart_radius"] != 2:
            raise ValueError("Only the explicitly planned radius-2, 20-epoch six-job screen is authorized")
        old, result = (profile if cfg["model"] == "sl8" else smoke)[(cfg["dataset"], cfg["model"])]
        if any(old.get(k) != v for k, v in cfg.items() if k not in ignored):
            raise ValueError("Measured and planned training/scoring configurations differ")
        event = result["last_epoch"]
        train = event["train_seconds"] * math.ceil(result["train_count"] / cfg["batch_size"]) / event["train_batches"]
        valid = event["validation"]["elapsed_seconds"] * result["validation_total_count"] / result["validation_count"]
        if not all(isinstance(t, (float, int)) and not isinstance(t, bool) and math.isfinite(t) and t > 0 for t in (train, valid)):
            raise ValueError("Nonfinite/nonpositive measured timing")
        evaluations = len([e for e in range(1, cfg["epochs"] + 1) if e % cfg["eval_every"] == 0 or e == cfg["epochs"]])
        costs.append({"job_id": job["job_id"], "train_epoch_seconds": train, "full_validation_seconds": valid,
                      "scheduled_validations": evaluations, "seconds": cfg["epochs"] * train + evaluations * valid,
                      "timing_source": "complete_SL_epoch" if cfg["model"] == "sl8" else "short_smoke_extrapolation"})
    total = sum(c["seconds"] for c in costs)
    return {"jobs": costs, "base_seconds": total, "safety_multiplier": 1.2, "allowance_seconds": 120,
            "required_seconds": total * 1.2 + 120, "estimate_is_uncertain": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "profile-queue", "smoke-queue", "plan", "backup-root", "decision-file"):
        parser.add_argument("--" + key, type=lambda p: Path(p).expanduser().resolve(), required=True)
    parser.add_argument("--queue-id", type=queue.name, required=True)
    parser.add_argument("--deadline-utc", type=queue.parse_utc, required=True)
    args = parser.parse_args(argv)
    _, jobs = queue.load_plan(args.plan, args.deadline_utc, args.root)
    frozen = queue.source_fingerprints(args.root, args.plan)
    end = time.monotonic() + (args.deadline_utc - datetime.now(timezone.utc)).total_seconds()
    remaining = lambda: min(end - time.monotonic(), (args.deadline_utc - datetime.now(timezone.utc)).total_seconds())
    state = {"status": "waiting_for_profile", "created_at": queue.utc(), "deadline_utc": args.deadline_utc.isoformat(),
             "test_evaluated": False, "launcher_sha256": queue.digest(Path(__file__)), "source_sha256": frozen,
             "queue_id": args.queue_id, "profile_queue": str(args.profile_queue), "smoke_queue": str(args.smoke_queue)}
    args.decision_file.parent.mkdir(parents=True, exist_ok=True)
    with args.decision_file.open("x") as stream:
        json.dump(state, stream, indent=2, allow_nan=False)
    try:
        while remaining() > 120:
            profile_state = json.loads((args.profile_queue / "queue.json").read_text())
            if profile_state.get("status") != "running":
                break
            time.sleep(min(5, max(0, remaining() - 120)))
        if remaining() <= 120:
            state["status"] = "deadline_reached_before_dispatch"
            return 124
        profile = completed_records(args.root, args.profile_queue, args.deadline_utc, "profile", 2)
        smoke = completed_records(args.root, args.smoke_queue, args.deadline_utc, "smoke", 6)
        state["cost"] = estimate(jobs, smoke, profile)
        state["remaining_seconds_at_gate"] = remaining()
        if queue.source_fingerprints(args.root, args.plan) != frozen:
            raise ValueError("Source or screen plan changed while waiting")
        if state["cost"]["required_seconds"] > remaining():
            state["status"] = "stopped_cost_gate"
            return 2
        state.update(status="dispatched", dispatched_at=queue.utc())
        queue.save_json(args.decision_file, state)
        code = subprocess.call([sys.executable, "-u", str(args.root / "scripts/run_structural_geometry_queue.py"),
                                "--root", str(args.root), "--plan", str(args.plan), "--queue-id", args.queue_id,
                                "--deadline-utc", args.deadline_utc.isoformat(), "--backup-root", str(args.backup_root)], cwd=args.root)
        state.update(status="queue_returned", queue_exit_code=code)
        return code
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        return 1
    finally:
        state["updated_at"] = queue.utc()
        queue.save_json(args.decision_file, state)


if __name__ == "__main__":
    raise SystemExit(main())
