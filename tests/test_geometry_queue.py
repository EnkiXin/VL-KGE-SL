"""CPU-only queue contract, validation selection and safety tests."""

import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_geometry_queue.py"
SPEC = importlib.util.spec_from_file_location("geometry_queue", SCRIPT)
QUEUE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUEUE)

FAKE_RUNNER = '''import argparse, json, os, signal, sys, time
from pathlib import Path
p=argparse.ArgumentParser()
for key in ["root", "model", "run-dir", "epochs", "patience", "seed", "lr", "coordinate-scale", "chart-radius", "initial-logit-scale", "initial-offset", "kind"]:
    p.add_argument("--"+key, required=True)
a=p.parse_args()
d=Path(a.run_dir); d.mkdir()
(d/"args.json").write_text(json.dumps(vars(a)))
(d/"started").write_text(str(os.getpid()))
if os.environ.get("FAKE_WAIT"):
    time.sleep(30)
if os.environ.get("FAKE_FAIL"):
    sys.exit(7)
r={"status":"completed", "best_validation_mrr": 0.8 if float(a.lr)==0.03 else 0.7, "best_epoch":int(a.epochs)}
if a.kind=="formal" or os.environ.get("FAKE_PILOT_TEST"):
    r["final_test"]={"mrr":0.9,"hits":{"1":0.8}}
(d/"result.json").write_text(json.dumps(r))
print("fake run complete",flush=True)
'''


class GeometryQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "scripts").mkdir()
        (self.root / "scripts" / "run_geometry.py").write_text(FAKE_RUNNER)

    def tearDown(self):
        self.temp.cleanup()

    def command(self, *extra):
        return [sys.executable, str(SCRIPT), "--root", str(self.root), "--queue-id", "test",
                "--models", "euclidean", "sl8", "--learning-rates", "0.01", "0.03", *extra]

    def run_queue(self, *extra, env=None):
        return subprocess.run(self.command(*extra), capture_output=True, text=True,
                              env={**os.environ, **(env or {})}, timeout=15)

    def state(self):
        return json.loads((self.root / "runs" / "test" / "queue.json").read_text())

    def test_success_selects_validation_only_and_restarts_formal(self):
        backup = self.root / "persistent"
        result = self.run_queue("--backup-root", str(backup))
        self.assertEqual(result.returncode, 0, result.stderr)
        state = self.state()
        self.assertEqual(state["status"], "completed")
        self.assertEqual([job["kind"] for job in state["jobs"]], ["pilot"] * 4 + ["formal"] * 2)
        for choice in state["validation_selection"].values():
            self.assertEqual(choice["lr"], 0.03)
        for job in state["jobs"]:
            args = json.loads((Path(job["run_dir"]) / "args.json").read_text())
            self.assertEqual(args["seed"], "42")
            self.assertEqual(args["patience"], "50")
            self.assertEqual(args["epochs"], "10" if job["kind"] == "pilot" else "200")
            self.assertNotIn("--resume-from", job["command"])
            self.assertEqual("final_test" in job, job["kind"] == "formal")
            self.assertTrue((Path(job["backup"]) / "result.json").is_file())
            self.assertTrue((Path(job["backup"]) / "train.log").is_file())

    def test_failed_job_stops_without_retry(self):
        result = self.run_queue(env={"FAKE_FAIL": "1"})
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.state()["status"], "failed")
        self.assertEqual(len(self.state()["jobs"]), 1)
        self.assertEqual(self.state()["jobs"][0]["exit_code"], 7)

    def test_murp_local_metric_calibration_is_recorded_and_passed(self):
        result = self.run_queue("--models", "euclidean", "mure", "murp", "sl8",
                                "--learning-rates", "0.01")
        self.assertEqual(result.returncode, 0, result.stderr)
        state = self.state()
        self.assertEqual(state["local_metric_calibration"]["base_initial_logit_scale"], 100.0)
        self.assertEqual(state["local_metric_calibration"]["model_scale_factors"]["murp"], 0.25)
        for job in state["jobs"]:
            expected = 25.0 if job["model"] == "murp" else 100.0
            self.assertEqual(job["actual_initial_logit_scale"], expected)
            args = json.loads((Path(job["run_dir"]) / "args.json").read_text())
            self.assertEqual(float(args["initial_logit_scale"]), expected)

    def test_pilot_with_test_result_is_rejected(self):
        result = self.run_queue(env={"FAKE_PILOT_TEST": "1"})
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.state()["status"], "failed")
        self.assertIn("test set", self.state()["error"])
        self.assertEqual(len(self.state()["jobs"]), 1)

    def test_existing_empty_queue_is_not_reused(self):
        queue = self.root / "runs" / "test"
        queue.mkdir(parents=True)
        marker = queue / "user-file"
        marker.write_text("keep")
        result = self.run_queue()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(marker.read_text(), "keep")
        self.assertFalse((queue / "queue.json").exists())

    def test_gpu_lock_prevents_launch(self):
        (self.root / "runs").mkdir()
        with (self.root / "runs" / "gpu0.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.run_queue()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "runs" / "test").exists())

    def test_existing_broken_queue_symlink_is_not_followed(self):
        (self.root / "runs").mkdir()
        target = self.root / "must-not-create"
        (self.root / "runs" / "test").symlink_to(target)
        result = self.run_queue()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(target.exists())

    def test_bad_arguments_rejected_before_queue_creation(self):
        for extra in [("--queue-id", "../bad"), ("--queue-id", ".."),
                      ("--max-hours", "nan"), ("--max-hours", "0"),
                      ("--pilot-epochs", "0"), ("--learning-rates", "0.01", "0.01"),
                      ("--models", "sl8", "sl8")]:
            with self.subTest(extra=extra):
                result = self.run_queue(*extra)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.root / "runs").exists())

    def test_backup_collision_stops_queue_without_overwriting(self):
        backup = self.root / "persistent"
        target = backup / "test" / "pilot-euclidean-lr-0p01-seed42"
        target.mkdir(parents=True)
        (target / "user-file").write_text("keep")
        result = self.run_queue("--backup-root", str(backup))
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.state()["status"], "backup_failed")
        self.assertEqual(len(self.state()["jobs"]), 1)
        self.assertEqual((target / "user-file").read_text(), "keep")

    def test_backup_inside_source_is_rejected_before_launch(self):
        result = self.run_queue("--backup-root", str(self.root / "runs"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("outside the source queue", result.stderr)
        self.assertFalse((self.root / "runs").exists())

    def test_wall_budget_stops_running_child(self):
        result = self.run_queue("--max-hours", "0.00015", env={"FAKE_WAIT": "1"})
        self.assertEqual(result.returncode, 124, result.stderr)
        state = self.state()
        self.assertEqual(state["status"], "budget_exhausted")
        self.assertEqual(len(state["jobs"]), 1)
        self.assertEqual(state["jobs"][0]["status"], "budget_exhausted")
        self.assertLess(state["elapsed_seconds"], 5)

    def test_wall_budget_expired_before_first_launch(self):
        result = self.run_queue("--max-hours", "0.0000000001")
        self.assertEqual(result.returncode, 124, result.stderr)
        self.assertEqual(self.state()["status"], "budget_exhausted")
        self.assertEqual(self.state()["jobs"], [])

    def test_sigterm_gracefully_stops_owned_child(self):
        process = subprocess.Popen(self.command(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, env={**os.environ, "FAKE_WAIT": "1"})
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                markers = list((self.root / "runs").glob("test/*/started"))
                if markers:
                    break
                time.sleep(0.02)
            self.assertTrue(markers, "fake child never started")
            process.send_signal(signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 130, stderr)
            self.assertEqual(self.state()["status"], "interrupted")
            self.assertEqual(len(self.state()["jobs"]), 1)
            # The child has exited and no longer retains the inherited GPU lock.
            with (self.root / "runs" / "gpu0.lock").open("a+") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()

    def test_result_validation_rejects_nonfinite_mrr(self):
        target = self.root / "fake-result"
        target.mkdir()
        (target / "result.json").write_text(json.dumps({"status": "completed", "best_validation_mrr": float("nan"), "best_epoch": 1}))
        with self.assertRaisesRegex(ValueError, "best_validation_mrr"):
            QUEUE.read_result(target, "pilot")


if __name__ == "__main__":
    unittest.main()
