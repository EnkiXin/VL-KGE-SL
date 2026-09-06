"""One-GPU, stop-on-failure queue for the two author WN9 configurations."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Queue received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--queue-id", required=True)
    parser.add_argument("--backup-root", type=Path,
                        help="Optional persistent directory; copy each completed run here")
    args = parser.parse_args()
    root = args.root.resolve()
    if Path(args.queue_id).name != args.queue_id:
        raise ValueError("queue-id must be a simple directory name")
    queue = root / "runs" / args.queue_id
    queue.mkdir(parents=True, exist_ok=True)
    with (root / "runs" / "gpu0.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = queue / "queue.json"
        if state_path.exists():
            raise RuntimeError("Queue already exists; inspect its state instead of overwriting it")
        state = {"pid": os.getpid(), "status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "jobs": []}

        def save():
            temp = state_path.with_suffix(".tmp")
            temp.write_text(json.dumps(state, indent=2) + "\n")
            os.replace(temp, state_path)

        save()
        for slug in ["distmult", "complex"]:
            run_dir = queue / f"wn9-{slug}-clip-seed42"
            command = [sys.executable, "-u", str(root / "scripts/run_author.py"),
                       "--repo", str(root / "upstream/vl-kge"),
                       "--config", f"vlkge/configs/wn9_img/{slug}_clip.yaml",
                       "--run-dir", str(run_dir)]
            log = queue / f"{slug}.log"
            job = {"model": slug, "status": "running", "command": command, "log": str(log), "run_dir": str(run_dir)}
            state["jobs"].append(job)
            save()
            env = os.environ.copy()
            env.setdefault("CUDA_VISIBLE_DEVICES", "0")
            env.update(PYTHONHASHSEED="42", PYTHONUNBUFFERED="1")
            process = None
            try:
                with log.open("x") as output:
                    # The child retains the flock even if the parent is killed.
                    process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                               env=env, pass_fds=(lock.fileno(),))
                    job["pid"] = process.pid
                    save()
                    code = process.wait()
            except BaseException as error:
                if process is not None and process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                job.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                           error=repr(error))
                state["status"] = job["status"]
                save()
                raise
            job.update(exit_code=code, status="completed" if code == 0 else "failed")
            if code != 0:
                state["status"] = "failed"
                save()
                raise SystemExit(code)
            if args.backup_root is not None:
                backup = args.backup_root.resolve() / args.queue_id / run_dir.name
                try:
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(run_dir, backup)
                    shutil.copy2(log, backup / "train.log")
                    job["backup"] = str(backup)
                except Exception as error:
                    job["backup_error"] = repr(error)
                    state["status"] = "backup_failed"
                    save()
                    raise
            save()
        state.update(status="completed", ended_at=datetime.now(timezone.utc).isoformat())
        save()


if __name__ == "__main__":
    main()
