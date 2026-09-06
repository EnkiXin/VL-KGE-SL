"""Deadline-bound three-dataset SL28 search, optionally waiting for a legacy queue.

No preceding process is signalled. This controller owns only its new session
process groups and backs up incomplete runs as well as completed results. The
user-authorized default selects hyperparameters/checkpoints on TEST, so all
outputs are explicitly test_tuned_not_held_out, not unbiased held-out evidence.
"""

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


LRS = (0.03, 0.1, 0.01, 0.05, 0.003, 0.2)
STAGES = (("wn9_img", "pilot", 1), ("wikiart_mkg_v1", "pilot", 1), ("wikiart_mkg_v2", "pilot", 1),
          ("wn9_img", "formal", 2), ("wikiart_mkg_v1", "formal", 2), ("wikiart_mkg_v2", "formal", 2))
EPOCHS = {"wn9_img": (10, 200), "wikiart_mkg_v1": (5, 50), "wikiart_mkg_v2": (2, 20)}
TERMINAL = {"completed", "failed", "backup_failed", "budget_exhausted", "interrupted", "stopped"}
GEOMETRY = ["--embedding-dim", "768", "--matrix-dim", "28", "--entity-mapping", "linear",
            "--coordinate-scale", "0.1", "--chart-radius", "2", "--relation-radius", "1.5",
            "--relation-init-norm", "0.5", "--relation-mode", "left", "--score-form", "linear",
            "--initial-logit-scale", "3", "--initial-offset", "3", "--schatten-p", "2",
            "--log-sqrt-steps", "1", "--log-terms", "12", "--batch-size", "512", "--patience", "50"]


class ProcessSafetyError(Exception):
    """Abort all scheduling when a child/group cannot be safely verified/drained."""


def utc():
    return datetime.now(timezone.utc).isoformat()


def simple_name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise argparse.ArgumentTypeError("must be a simple directory name")
    return value


def deadline_timestamp(value, now=None):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("--deadline-utc must contain UTC timezone Z or +00:00")
    deadline = parsed.timestamp()
    now = time.time() if now is None else now
    if not now < deadline <= now + 24 * 3600:
        raise ValueError("--deadline-utc must be in the future and at most 24 hours away")
    return deadline


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--deadline-utc", required=True)
    parser.add_argument("--preceding-root", type=Path)
    parser.add_argument("--preceding-queue", type=simple_name)
    parser.add_argument("--selection-split", choices=("test", "validation"), default="test")
    parser.add_argument("--controller-id", type=simple_name, default="sl28-three-datasets")
    args = parser.parse_args(argv)
    try:
        args.deadline = deadline_timestamp(args.deadline_utc)
    except ValueError as error:
        parser.error(str(error))
    if (args.preceding_root is None) != (args.preceding_queue is None):
        parser.error("--preceding-root and --preceding-queue must be supplied together")
    for name in ("root", "backup_root"):
        setattr(args, name, getattr(args, name).resolve())
    if args.preceding_root is not None:
        args.preceding_root = args.preceding_root.resolve()
    if args.root == args.preceding_root:
        parser.error("--root must be separate from the frozen --preceding-root")
    if args.backup_root == args.root or args.root in args.backup_root.parents:
        parser.error("--backup-root must be outside --root")
    return args


def read_json(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")
    os.replace(temporary, path)


def proc_snapshot(proc_root=Path("/proc")):
    """Read Linux process identity without relying on PID existence alone."""
    if not proc_root.is_dir():
        raise RuntimeError("Linux /proc is required for safe preceding/child process checks")
    result = []
    for directory in proc_root.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            if fields[0] == "Z":
                continue
            item = dict(pid=int(directory.name), ppid=int(fields[1]), pgid=int(fields[2]),
                        sid=int(fields[3]), starttime=int(fields[19]), cwd=None, argv=[])
            try:
                item["cwd"] = str((directory / "cwd").resolve(strict=True))
                item["argv"] = [part.decode(errors="surrogateescape") for part in
                                (directory / "cmdline").read_bytes().split(b"\0") if part]
            except PermissionError:
                pass
            result.append(item)
        except (FileNotFoundError, ProcessLookupError):
            continue
    return result


def flag(argv, name):
    try:
        index = argv.index(name)
        return argv[index + 1]
    except (ValueError, IndexError):
        return None


def process_matches(item, root, queue_id=None, smoke_dir=None):
    """Require exact cwd, script and --root, plus the queue/run-dir boundary."""
    root = root.resolve()
    if item.get("cwd") != str(root):
        return False
    argv = item.get("argv", [])
    root_arg = flag(argv, "--root")
    if root_arg is None or (root / root_arg).resolve() != root:
        return False
    scripts = {(root / argument).resolve() for argument in argv if argument.endswith(".py")}
    if root / "scripts/run_geometry_queue.py" in scripts:
        return queue_id is not None and flag(argv, "--queue-id") == queue_id
    if root / "scripts/run_geometry.py" not in scripts:
        return False
    run_arg = flag(argv, "--run-dir")
    if run_arg is None:
        return False
    run_dir = (root / run_arg).resolve()
    if smoke_dir is not None and run_dir == smoke_dir.resolve():
        return True
    return queue_id is not None and root / "runs" / queue_id in run_dir.parents


def preceding_status(root, queue_id, snapshot=None):
    state = read_json(root / "runs" / queue_id / "queue.json")
    processes = proc_snapshot() if snapshot is None else snapshot
    active = [item["pid"] for item in processes if process_matches(item, root, queue_id)]
    return {"ready": bool(state and state.get("status") in TERMINAL and not active),
            "status": state.get("status") if state else "not_created", "active_pids": active,
            "queue_path": str(root / "runs" / queue_id / "queue.json")}


def stage_deadline(now, deadline, index):
    remaining = max(0.0, deadline - now)
    return min(deadline, now + remaining * STAGES[index][2] / sum(s[2] for s in STAGES[index:]))


def queue_command(args, dataset, kind, queue_id, hours, selected_lr=None):
    command = [sys.executable, "-u", str(args.root / "scripts/run_geometry_queue.py"),
               "--root", str(args.root), "--dataset", dataset, "--queue-id", queue_id,
               "--backup-root", str(args.backup_root / args.controller_id / "queues"),
               "--models", "slv2n28", "--max-hours", str(hours),
               "--pilot-epochs", str(EPOCHS[dataset][0]), "--formal-epochs", str(EPOCHS[dataset][1]),
               "--learning-rates", *map(str, LRS), "--selection-split", args.selection_split,
               "--negatives", "100" if dataset == "wn9_img" else "1",
               "--validate-every", "5" if dataset == "wn9_img" else "1", *GEOMETRY]
    if kind == "pilot":
        command += ["--pilots-only", "--continue-failed-pilots"]
    else:
        if selected_lr not in LRS:
            raise ValueError("Formal LR must come from a configured completed pilot")
        command += ["--formal-only-lr", str(selected_lr)]
    return command


def smoke_command(args, dataset, run_dir):
    return [sys.executable, "-u", str(args.root / "scripts/run_geometry.py"),
            "--root", str(args.root), "--dataset", dataset, "--model", "slv2",
            "--run-dir", str(run_dir), "--kind", "smoke", "--epochs", "1", "--seed", "42",
            "--lr", str(LRS[0]), "--max-train-batches", "1", "--eval-limit", "2",
            "--selection-split", args.selection_split, "--validate-every", "1",
            "--negatives", "100" if dataset == "wn9_img" else "1", *GEOMETRY]


class OwnedProcess:
    """A new session whose verified queue and runner members alone may be killed."""
    def __init__(self, process, root, queue_id=None, smoke_dir=None, scan=proc_snapshot):
        self.process, self.root = process, root
        self.queue_id, self.smoke_dir, self.scan = queue_id, smoke_dir, scan
        self.pgid = process.pid
        snapshot = scan()
        leader = next((item for item in snapshot if item["pid"] == process.pid), None)
        self.starttime = leader["starttime"] if leader else None

    def members(self):
        members = [item for item in self.scan() if item["pgid"] == self.pgid]
        for item in members:
            if (item["sid"] != self.pgid or not process_matches(item, self.root, self.queue_id, self.smoke_dir)
                    or (item["pid"] == self.pgid and self.starttime is not None
                        and item["starttime"] != self.starttime)):
                raise ProcessSafetyError(f"Refusing to signal process group with unverified member {item['pid']}")
        return members

    def stop(self, grace=20.0):
        if self.members():
            try:
                os.killpg(self.pgid, signal.SIGINT)
            except ProcessLookupError:
                pass
        end = time.monotonic() + grace
        while self.members() and time.monotonic() < end:
            self.process.poll()
            time.sleep(0.1)
        if self.members():
            try:
                os.killpg(self.pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.process.wait(timeout=5)
        # A killed orphan may briefly remain before the kernel removes /proc.
        end = time.monotonic() + 5
        while self.members() and time.monotonic() < end:
            time.sleep(0.05)
        if self.members():
            raise ProcessSafetyError("Owned process descendants did not exit; refusing next stage")


def run_owned(command, root, log, deadline, *, queue_id=None, smoke_dir=None, on_start=None):
    # Reserve cooperative drain time inside the allotted stage/global deadline.
    drain_at = deadline - 20.0
    if time.time() >= drain_at:
        return {"status": "budget_exhausted", "exit_code": 124, "started": False}
    environment = os.environ.copy()
    environment.update(PYTHONHASHSEED="42", PYTHONUNBUFFERED="1")
    environment.setdefault("CUDA_VISIBLE_DEVICES", "0")
    with log.open("x") as output:
        process = subprocess.Popen(command, cwd=root, stdout=output, stderr=subprocess.STDOUT,
                                   env=environment, start_new_session=True)
        owner = OwnedProcess(process, root, queue_id, smoke_dir)
        try:
            if on_start is not None:
                on_start(process.pid)
            while time.time() < drain_at:
                code = process.poll()
                if code is not None:
                    if owner.members():
                        owner.stop()
                    return {"status": "completed" if code == 0 else "failed", "exit_code": code,
                            "pid": process.pid, "started": True}
                time.sleep(min(1.0, max(0.0, drain_at - time.time())))
            owner.stop(grace=max(0.0, min(20.0, deadline - time.time())))
            return {"status": "budget_exhausted", "exit_code": 124, "pid": process.pid, "started": True}
        except BaseException:
            handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
            for sig in handlers:
                signal.signal(sig, signal.SIG_IGN)
            try:
                owner.stop()
            finally:
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)
            raise


def select_pilot(queue, dataset, selection_split="test"):
    """Select only completed pilots matching the explicitly requested tuning split."""
    state = read_json(queue / "queue.json") or {}
    candidates, rejected = [], []
    for job in state.get("jobs", []):
        if job.get("kind") != "pilot" or job.get("status") != "completed":
            continue
        try:
            run_dir = Path(job["run_dir"]).resolve()
            if queue.resolve() not in run_dir.parents or job.get("model") != "slv2n28":
                raise ValueError("foreign run path/model")
            result = read_json(run_dir / "result.json") or {}
            value, epoch = result.get("best_selection_mrr"), result.get("best_epoch")
            config = result.get("args", {})
            metric_key = "best_test_mrr" if selection_split == "test" else "best_validation_mrr"
            evaluation_key = "test_selection" if selection_split == "test" else "validation"
            evaluation = result.get("last_epoch", {}).get(evaluation_key, {}) or {}
            if (result.get("status") != "completed" or result.get("kind") != "pilot"
                    or result.get("dataset") != dataset or result.get("final_test") is not None
                    or result.get("selection_split") != selection_split
                    or result.get("test_used_for_selection") is not (selection_split == "test")
                    or (selection_split == "test" and result.get("evaluation_label") != "test_tuned_not_held_out")
                    or result.get(metric_key) != value
                    or isinstance(value, bool) or not isinstance(value, (float, int))
                    or not math.isfinite(value) or not 0 <= value <= 1
                    or isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1
                    or evaluation.get("complete") is not True or config.get("seed") != 42
                    or config.get("matrix_dim") != 28 or config.get("embedding_dim") != 768
                    or config.get("entity_mapping") != "linear" or config.get("lr") != job.get("lr")
                    or job.get("lr") not in LRS):
                raise ValueError("result is not an eligible completed SL28 pilot for this selection split")
            candidates.append({"lr": job["lr"], "best_selection_mrr": value, metric_key: value,
                               "selection_split": selection_split, "best_epoch": epoch,
                               "run_dir": str(run_dir), "dataset": dataset})
        except (ValueError, KeyError, TypeError) as error:
            rejected.append({"run_dir": job.get("run_dir"), "error": str(error)})
    candidates.sort(key=lambda item: (-item["best_selection_mrr"], LRS.index(item["lr"])))
    completed_lrs = {item["lr"] for item in candidates}
    return {"selected": candidates[0] if candidates else None, "eligible_pilots": candidates,
            "rejected_pilots": rejected, "search_incomplete": completed_lrs != set(LRS)}


def main(argv=None):
    args = parse_args(argv)
    for script in ("run_geometry.py", "run_geometry_queue.py"):
        if not (args.root / "scripts" / script).is_file():
            raise FileNotFoundError(f"Missing runner script: {script}")
    control = args.root / "runs" / args.controller_id
    backup = args.backup_root / args.controller_id
    control.parent.mkdir(parents=True, exist_ok=True)
    backup.parent.mkdir(parents=True, exist_ok=True)
    state = {"status": "waiting_preceding", "started_at": utc(), "pid": os.getpid(),
             "deadline_utc": args.deadline_utc, "architecture": "768 -> Linear(768,783) -> SL(28)",
             "learning_rates": LRS, "stages": [], "selection": {},
             "legacy_preceding_reference": ({"root": str(args.preceding_root), "queue": args.preceding_queue,
                                              "reused_for_new_experiment": False}
                                             if args.preceding_root is not None else None),
             "backup_root": str(backup), "selection_split": args.selection_split,
             "test_used_for_selection": args.selection_split == "test",
             "evaluation_label": "test_tuned_not_held_out" if args.selection_split == "test" else "validation_selected"}

    def save():
        state["updated_at"] = utc()
        save_json(control / "state.json", state)
        save_json(backup / "state.json", state)

    def copy_changed(source, destination):
        source, destination = Path(source), Path(destination)
        if destination.is_file() and not destination.is_symlink():
            before, after = source.stat(), destination.stat()
            if before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns:
                return str(destination)
        return shutil.copy2(source, destination)

    def backup_stage(queue_id=None):
        if queue_id is not None:
            source = args.root / "runs" / queue_id
            if source.exists():
                shutil.copytree(source, backup / "queues" / queue_id, dirs_exist_ok=True,
                                copy_function=copy_changed)
        shutil.copytree(control, backup / "controller", dirs_exist_ok=True, copy_function=copy_changed)

    def record_pid(stage, pid):
        stage["pid"] = pid
        save()

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Controller received signal {signum}")

    original_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    current_queue = None
    with (control.parent / "sl28-three-datasets.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        control.mkdir(exist_ok=False)
        backup.mkdir(exist_ok=False)
        for sig in original_handlers:
            signal.signal(sig, interrupted)
        try:
            save()
            while args.preceding_root is not None and time.time() < args.deadline:
                preceding = preceding_status(args.preceding_root, args.preceding_queue)
                state["preceding"] = preceding
                save()
                if preceding["ready"]:
                    break
                time.sleep(min(5.0, max(0.0, args.deadline - time.time())))
            if time.time() >= args.deadline:
                state["status"] = "deadline_reached_before_new_experiment"
                return 124
            for index, (dataset, kind, weight) in enumerate(STAGES):
                if time.time() >= args.deadline:
                    state["status"] = "deadline_reached"
                    return 124
                stage = {"dataset": dataset, "kind": kind, "weight": weight, "status": "starting",
                         "started_at": utc()}
                state["stages"].append(stage)
                selected = state["selection"].get(dataset, {}).get("selected")
                if kind == "formal" and selected is None:
                    stage.update(status="skipped_no_valid_pilot", ended_at=utc())
                    save()
                    continue
                end = stage_deadline(time.time(), args.deadline, index)
                stage["deadline_utc"] = datetime.fromtimestamp(end, timezone.utc).isoformat()
                current_queue = f"{args.controller_id}-{dataset}-{kind}"
                stage["queue_id"] = current_queue
                state["status"] = "running"
                save()
                try:
                    if kind == "pilot":
                        smoke_dir = control / f"smoke-{dataset}"
                        smoke = run_owned(smoke_command(args, dataset, smoke_dir), args.root,
                                          control / f"smoke-{dataset}.log", end, smoke_dir=smoke_dir,
                                          on_start=lambda pid: record_pid(stage, pid))
                        stage["smoke"] = smoke
                        result = read_json(smoke_dir / "result.json") or {}
                        if (smoke["status"] != "completed" or result.get("status") != "completed"
                                or result.get("final_test") is not None):
                            stage.update(status="smoke_failed", ended_at=utc())
                            continue
                    remaining = end - time.time()
                    if remaining <= 0:
                        stage["status"] = "budget_exhausted"
                        continue
                    command = queue_command(args, dataset, kind, current_queue, remaining / 3600,
                                            selected["lr"] if selected else None)
                    stage["command"] = command
                    save()
                    outcome = run_owned(command, args.root, control / f"{current_queue}.log", end,
                                        queue_id=current_queue, on_start=lambda pid: record_pid(stage, pid))
                    stage.update(outcome)
                    queue_state = read_json(args.root / "runs" / current_queue / "queue.json") or {}
                    stage["queue_status"] = queue_state.get("status")
                    if outcome["status"] == "completed" and queue_state.get("status") != "completed":
                        stage["status"] = "failed_missing_completed_queue"
                    if kind == "pilot":
                        state["selection"][dataset] = select_pilot(args.root / "runs" / current_queue, dataset,
                                                                 args.selection_split)
                except (OSError, ValueError, RuntimeError) as error:
                    stage.update(status="failed", error=repr(error))
                finally:
                    stage["ended_at"] = utc()
                    save()
                    backup_stage(current_queue)
                    current_queue = None
            state["status"] = ("completed" if all(stage["status"] == "completed" for stage in state["stages"])
                               else "finished_with_incomplete_or_failed_stages")
            return 0 if state["status"] == "completed" else 1
        except BaseException as error:
            state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=repr(error))
            return 130 if isinstance(error, KeyboardInterrupt) else 1
        finally:
            for sig in original_handlers:
                signal.signal(sig, signal.SIG_IGN)
            state["ended_at"] = utc()
            save()
            backup_stage(current_queue)
            queues = {}
            for stage in state["stages"]:
                if "queue_id" in stage:
                    queues[stage["queue_id"]] = read_json(args.root / "runs" / stage["queue_id"] / "queue.json")
            preceding = (read_json(args.preceding_root / "runs" / args.preceding_queue / "queue.json")
                         if args.preceding_root is not None else None)
            results = {"controller": state, "legacy_preceding_queue_reference": preceding, "dataset_queues": queues}
            save_json(control / "all-results.json", results)
            save_json(backup / "all-results.json", results)
            for sig, handler in original_handlers.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
