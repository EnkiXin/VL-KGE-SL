"""Bounded executor tests with synthetic child processes; no GPU or network."""

from contextlib import redirect_stderr, redirect_stdout
import fcntl
import importlib.util
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("distmult_queue_tested", ROOT / "scripts/run_distmult_sl_queue.py")
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)
planner = queue.script_module("plan_distmult_sl_hpo.py")


class DistMultQueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.root = self.base / "project"
        (self.root / "scripts").mkdir(parents=True)
        (self.root / "scripts/run_distmult_sl_author.py").write_text("# synthetic child fixture\n")
        self.spec = planner.checked_spec(ROOT / "experiments/distmult-sl-hpo-v1.json")
        planner.write_plan(self.spec, self.base / "profile", "profile")
        self.manifest = self.base / "profile/manifest.json"
        self.backup = self.base / "persistent"
        self.processes = []
        self.mode = "success"
        self.result_changes = {}
        outer = self

        class Child:
            def __init__(self, command, **kwargs):
                self.command, self.kwargs = command, kwargs
                self.backup = "--_backup-worker" in command
                self.pid = 80000 + len(outer.processes)
                self.returncode = None
                self.signals = []
                outer.processes.append(self)

            def poll(self):
                return self.returncode

            def send_signal(self, signum):
                self.signals.append(signum)
                self.returncode = -signum

            def kill(self):
                self.signals.append(signal.SIGKILL)
                self.returncode = -signal.SIGKILL

            def wait(self, timeout=None):
                if self.returncode is not None:
                    return self.returncode
                if outer.mode == "timeout" and not self.backup:
                    raise subprocess.TimeoutExpired(self.command, timeout)
                if outer.mode == "interrupt" and not self.backup:
                    raise KeyboardInterrupt("synthetic interruption")
                if self.backup:
                    if outer.mode == "backup_failure":
                        self.returncode = 9
                    else:
                        queue.backup_worker(*(Path(value) for value in self.command[-4:]))
                        self.returncode = 0
                    return self.returncode
                if outer.mode == "startup_failure":
                    self.returncode = 7
                    return self.returncode
                value = lambda flag: self.command[self.command.index(flag) + 1]
                run_dir = Path(value("--run-dir"))
                run_dir.mkdir()
                (run_dir / "best.pt").write_bytes(b"synthetic checkpoint, not a model")
                overrides = json.loads(Path(value("--author-overrides")).read_text())
                extension = json.loads(Path(value("--extension-config")).read_text())
                result = {"status": "completed_validation_only", "method": value("--method"),
                          "test_evaluated": False, "best_validation_mrr": 0.5,
                          "best_epoch": 1, "completed_epochs": overrides["epochs"],
                          "author_overrides": overrides, "extension_config": extension,
                          **outer.result_changes}
                queue.save_json(run_dir / "result.json", result)
                if outer.mode == "source_change":
                    (outer.root / "scripts/run_distmult_sl_author.py").write_text("# changed during HPO\n")
                self.returncode = 0
                return 0

        self.Child = Child

    def arguments(self, *extra):
        return ["--root", str(self.root), "--manifest", str(self.manifest),
                "--queue-id", "trial-v1", "--max-hours", "8", "--backup-root", str(self.backup), *extra]

    def execute(self, arguments=None):
        with patch.object(queue.subprocess, "Popen", self.Child), redirect_stdout(io.StringIO()):
            return queue.main(arguments or self.arguments())

    def state(self):
        return json.loads((self.root / "runs/trial-v1/queue.json").read_text())

    def test_success_profiles_three_methods_and_backs_up_every_checkpoint(self):
        self.assertEqual(self.execute(), 0)
        state = self.state()
        self.assertEqual(state["status"], "completed")
        training = [child for child in self.processes if not child.backup]
        self.assertEqual(len(training), 3)
        self.assertTrue(all("--validation-only" in child.command for child in training))
        self.assertTrue(all(child.command[2].endswith("run_distmult_sl_author.py") for child in training))
        self.assertTrue(all(child.kwargs.get("pass_fds") for child in self.processes))
        for job in state["jobs"]:
            self.assertEqual(job["backup_status"], "completed")
            copied = self.backup / "trial-v1" / job["job_id"]
            self.assertTrue((copied / "best.pt").is_file())
            self.assertTrue((copied / "train.log").is_file())
            self.assertTrue((copied / "queue_snapshot.json").is_file())
        self.assertEqual(len(state["validation_records"]), 3)
        self.assertTrue(all(row["test_evaluated"] is False and row["stage"] == "profile" for row in state["validation_records"]))

    def test_profile_and_anchors_share_one_deadline_and_preserve_stage_order(self):
        planner.write_plan(self.spec, self.base / "anchors", "anchors")
        arguments = self.arguments()
        arguments.insert(arguments.index("--queue-id"), str(self.base / "anchors/manifest.json"))
        self.assertEqual(self.execute(arguments), 0)
        state = self.state()
        self.assertEqual(state["stages"], ["profile", "anchors"])
        self.assertEqual(len(state["jobs"]), 21)
        self.assertEqual(state["max_hours"], 8.0)
        self.assertTrue(all(job["stage"] == "profile" for job in state["jobs"][:3]))

    def test_startup_failure_stops_before_other_methods_or_anchors(self):
        self.mode = "startup_failure"
        self.assertEqual(self.execute(), 1)
        self.assertEqual(len(self.processes), 1)
        self.assertEqual(self.state()["status"], "failed")
        self.assertFalse((self.root / "runs/trial-v1/profile-c000-baseline-s42").exists())

    def test_timeout_signals_only_owned_child_and_stops_queue(self):
        self.mode = "timeout"
        self.assertEqual(self.execute(), 124)
        self.assertEqual(self.state()["status"], "budget_exhausted")
        self.assertEqual(len(self.processes), 1)
        self.assertEqual(self.processes[0].signals, [signal.SIGINT])

    def test_keyboard_interrupt_gracefully_stops_owned_child(self):
        self.mode = "interrupt"
        self.assertEqual(self.execute(), 130)
        self.assertEqual(self.state()["status"], "interrupted")
        self.assertEqual(self.processes[0].signals, [signal.SIGINT])

    def test_backup_failure_preserves_local_completed_result_and_stops(self):
        self.mode = "backup_failure"
        self.assertEqual(self.execute(), 1)
        state = self.state()
        self.assertEqual(state["status"], "backup_failed")
        self.assertEqual(state["jobs"][0]["status"], "completed")
        self.assertEqual(state["jobs"][0]["backup_status"], "failed")
        self.assertTrue(Path(state["jobs"][0]["run_dir"]).joinpath("best.pt").exists())
        self.assertEqual(len(self.processes), 2)

    def test_test_contaminated_or_wrong_result_is_rejected(self):
        self.result_changes = {"final_test": None}
        self.assertEqual(self.execute(), 1)
        self.assertEqual(self.state()["status"], "failed")
        self.assertEqual(self.state()["validation_records"], [])
        self.assertEqual(len(self.processes), 1)

    def test_existing_queue_is_not_overwritten(self):
        self.assertEqual(self.execute(), 0)
        before = (self.root / "runs/trial-v1/queue.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.execute()
        self.assertEqual((self.root / "runs/trial-v1/queue.json").read_bytes(), before)

    def test_gpu_lock_contention_does_not_create_queue_or_launch_child(self):
        (self.root / "runs").mkdir()
        with (self.root / "runs/gpu0.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.execute()
        self.assertFalse((self.root / "runs/trial-v1").exists())
        self.assertEqual(self.processes, [])

    def test_manifest_tamper_or_config_path_escape_rejected_before_launch(self):
        manifest = json.loads(self.manifest.read_text())
        manifest["jobs"][0]["author_overrides_path"] = "../outside.json"
        self.manifest.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            self.execute()
        self.assertEqual(self.processes, [])
        self.assertFalse((self.root / "runs").exists())

    def test_source_change_stops_before_another_training_trial(self):
        self.mode = "source_change"
        self.assertEqual(self.execute(), 1)
        self.assertEqual(self.state()["status"], "failed")
        self.assertEqual(len([child for child in self.processes if not child.backup]), 1)

    def test_explicit_finite_budget_is_mandatory(self):
        arguments = self.arguments()
        index = arguments.index("--max-hours")
        for replacement in (None, "nan", "inf", "1e308", "0", "-1"):
            candidate = arguments[:]
            if replacement is None:
                del candidate[index:index + 2]
            else:
                candidate[index + 1] = replacement
            with self.subTest(replacement=replacement), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                queue.parse_args(candidate)

    def test_real_backup_worker_subprocess_copies_once_without_overwrite(self):
        source = self.base / "finished"
        source.mkdir()
        (source / "best.pt").write_bytes(b"small synthetic checkpoint")
        log, state = self.base / "train.log", self.base / "queue.json"
        log.write_text("finished validation-only\n")
        state.write_text('{"status":"running"}\n')
        destination = self.backup / "job"
        command = [sys.executable, str(ROOT / "scripts/run_distmult_sl_queue.py"), "--_backup-worker",
                   str(source), str(log), str(destination), str(state)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((destination / "best.pt").read_bytes(), (source / "best.pt").read_bytes())
        second = subprocess.run(command, capture_output=True, text=True, timeout=10)
        self.assertNotEqual(second.returncode, 0)
        self.assertEqual((destination / "best.pt").read_bytes(), b"small synthetic checkpoint")


if __name__ == "__main__":
    unittest.main()
