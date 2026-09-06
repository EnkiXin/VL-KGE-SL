"""Queue safety/provenance tests using owned synthetic children, no GPU."""

from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import signal
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("wn9_v2_queue", ROOT / "scripts/run_wn9_geometry_v2_queue.py")
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


class WN9V2QueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.root = self.base / "project"
        (self.root / "scripts").mkdir(parents=True)
        (self.root / "scripts/run_wn9_geometry_v2.py").write_text("# synthetic trainer\n")
        self.plan = self.base / "plan.json"
        self.deadline = (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat()
        self.raw = {"schema_version": 1, "deadline_utc": self.deadline,
                    "common": {"kind": "smoke", "epochs": 2, "max_train_batches": 3, "eval_limit": 32},
                    "jobs": [{"job_id": f"smoke-{model}", "stage": "smoke", "model": model}
                             for model in queue.MODELS]}
        queue.save_json(self.plan, self.raw)
        self.backup = self.base / "persistent"
        self.mode = "success"
        self.result_changes = {}
        self.children = []
        outer = self

        class Child:
            def __init__(self, command, **kwargs):
                self.command, self.kwargs = command, kwargs
                self.backup = "--_backup-worker" in command
                self.pid, self.returncode, self.signals = 80000 + len(outer.children), None, []
                outer.children.append(self)

            def poll(self):
                return self.returncode

            def send_signal(self, signum):
                self.signals.append(signum)
                self.returncode = -signum

            def kill(self):
                self.signals.append(signal.SIGKILL)
                self.returncode = -signal.SIGKILL

            def wait(self, timeout=None):
                self.last_timeout = timeout
                if self.returncode is not None:
                    return self.returncode
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
                (run_dir / "best.pt").write_bytes(b"synthetic checkpoint")
                if outer.mode == "timeout":
                    raise subprocess.TimeoutExpired(self.command, timeout)
                if outer.mode == "interrupt":
                    raise KeyboardInterrupt("synthetic interruption")
                config = {**queue.DEFAULTS, "model": value("--model")}
                for key, default in queue.DEFAULTS.items():
                    flag = "--" + key.replace("_", "-")
                    if flag in self.command:
                        config[key] = (int if default is None else type(default))(value(flag))
                count = min(config["eval_limit"] or 1337, 1337)
                indices = list(range(count))
                result = {"status": "completed_validation_only", "test_evaluated": False,
                          "model": config["model"], "evaluation_protocol": queue.PROTOCOL,
                          "training_negative_filter": "train_only_directional",
                          "validation_filter": "all_splits_directional", "tie_policy": "realistic_average",
                          "negative_sampling": "uniform_without_replacement_per_triple",
                          "score_distance_power": 1, "optimizer": "Adagrad", "dataset": "WN9-IMG",
                          "config": config, "best_validation_mrr": .4, "best_epoch": 1,
                          "completed_epochs": config["epochs"], "training_complete": config["kind"] == "pilot",
                          "validation_scope": "full" if count == 1337 else "subset",
                          "validation_count": count, "validation_total_count": 1337,
                          "validation_indices": indices,
                          "subset_indices_hash": hashlib.sha256(struct.pack("<" + "q" * count, *indices)).hexdigest(),
                          **outer.result_changes}
                queue.save_json(run_dir / "result.json", result)
                if outer.mode == "source_change":
                    (outer.root / "scripts/run_wn9_geometry_v2.py").write_text("# changed source\n")
                if outer.mode == "stop_after_current":
                    (outer.root / "runs/trial/STOP_AFTER_CURRENT").touch()
                self.returncode = 0
                return 0

        self.Child = Child

    def arguments(self):
        return ["--root", str(self.root), "--plan", str(self.plan), "--queue-id", "trial",
                "--deadline-utc", self.deadline, "--backup-root", str(self.backup)]

    def execute(self):
        with patch.object(queue.subprocess, "Popen", self.Child), redirect_stdout(io.StringIO()):
            return queue.main(self.arguments())

    def state(self):
        return json.loads((self.root / "runs/trial/queue.json").read_text())

    def test_three_methods_only_validation_and_mandatory_backups(self):
        self.assertEqual(self.execute(), 0)
        state = self.state()
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["counts"]["completed"], 3)
        self.assertEqual(len(state["validation_records"]), 3)
        self.assertEqual(queue.parse_utc(state["deadline_utc"]), queue.parse_utc(self.deadline))
        for child in self.children:
            self.assertTrue(child.kwargs["pass_fds"])
            self.assertGreater(child.last_timeout, 0)
            self.assertLess(child.last_timeout, 900)
            if not child.backup:
                self.assertTrue(child.command[2].endswith("run_wn9_geometry_v2.py"))
                self.assertIn("--validation-only", child.command)
        for job in state["jobs"]:
            self.assertEqual(job["backup_status"], "completed")
            self.assertEqual(job["validated_result"]["validation_scope"], "subset")
            destination = self.backup / "trial" / job["job_id"]
            self.assertTrue((destination / "best.pt").is_file())
            self.assertTrue((destination / "queue_snapshot.json").is_file())
            self.assertTrue((destination / "train.log").is_file())

    def test_smoke_then_full_screen_remain_distinct(self):
        self.raw["jobs"].append({"job_id": "screen-euclidean", "stage": "screen", "model": "euclidean",
                                 "kind": "pilot", "epochs": 30, "eval_limit": None, "max_train_batches": None})
        queue.save_json(self.plan, self.raw)
        self.assertEqual(self.execute(), 0)
        records = self.state()["validation_records"]
        self.assertEqual([r["stage"] for r in records], ["smoke"] * 3 + ["screen"])
        self.assertEqual(records[-1]["result"]["validation_scope"], "full")
        self.assertTrue(records[-1]["result"]["training_complete"])

    def test_startup_failure_without_status_stops_queue(self):
        self.mode = "startup_failure"
        self.assertEqual(self.execute(), 1)
        self.assertEqual(self.state()["status"], "failed")
        self.assertEqual(len(self.children), 1)
        self.assertEqual(self.state()["counts"]["not_started"], 2)

    def test_deadline_stops_only_owned_child_and_salvages_checkpoint(self):
        self.mode = "timeout"
        self.assertEqual(self.execute(), 124)
        state = self.state()
        self.assertEqual(state["status"], "deadline_reached")
        self.assertEqual(self.children[0].signals, [signal.SIGINT])
        self.assertEqual(len(self.children), 2)  # training plus bounded salvage
        self.assertTrue((self.backup / "trial/smoke-euclidean-partial/best.pt").is_file())
        self.assertEqual(state["validation_records"], [])

    def test_keyboard_interrupt_salvages_and_does_not_launch_next_trial(self):
        self.mode = "interrupt"
        self.assertEqual(self.execute(), 130)
        self.assertEqual(self.state()["status"], "interrupted")
        self.assertEqual(self.children[0].signals, [signal.SIGINT])
        self.assertEqual(len(self.children), 2)

    def test_stop_file_waits_for_completed_backup(self):
        self.mode = "stop_after_current"
        self.assertEqual(self.execute(), 0)
        state = self.state()
        self.assertEqual(state["status"], "stopped_after_current")
        self.assertEqual(state["counts"]["completed"], 1)
        self.assertEqual(state["jobs"][0]["backup_status"], "completed")
        self.assertEqual(len(self.children), 2)

    def test_backup_failure_preserves_local_result_and_fail_stops(self):
        self.mode = "backup_failure"
        self.assertEqual(self.execute(), 1)
        state = self.state()
        self.assertEqual(state["status"], "backup_failed")
        self.assertEqual(state["jobs"][0]["status"], "completed")
        self.assertTrue(Path(state["jobs"][0]["run_dir"]).joinpath("best.pt").is_file())
        self.assertEqual(len(self.children), 2)

    def test_test_contamination_protocol_scope_and_hash_are_rejected(self):
        self.assertEqual(self.execute(), 0)
        original_job = self.state()["jobs"][0]
        run = Path(original_job["run_dir"])
        original_result = json.loads((run / "result.json").read_text())
        for changes in ({"final_test": None}, {"test_evaluated": True}, {"evaluation_protocol": "old"},
                        {"tie_policy": "ascending_id"}, {"validation_scope": "full"},
                        {"subset_indices_hash": "bad"}, {"best_validation_mrr": float("inf")},
                        {"score_distance_power": 2}, {"dataset": "WN18RR"}, {"optimizer": "Adam"},
                        {"config": {**original_result["config"], "lr": 9.0}}):
            with self.subTest(changes=changes):
                (run / "result.json").write_text(json.dumps({**original_result, **changes}))
                with self.assertRaises(ValueError):
                    queue.checked_result(run, original_job)

    def test_command_parses_with_real_trainer_and_resolves_same_settings(self):
        spec = importlib.util.spec_from_file_location("wn9_v2_real_parser", ROOT / "scripts/run_wn9_geometry_v2.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        _, jobs = queue.load_plan(self.plan, queue.parse_utc(self.deadline))
        for job in jobs:
            command = queue.runner_command(self.root, self.base / "unused", job["config"])
            parsed = runner.parse_args(command[3:])
            self.assertTrue(parsed.validation_only)
            self.assertTrue(all(getattr(parsed, key) == value for key, value in job["config"].items()))
        default_parsed = runner.parse_args(["--root", str(self.root), "--run-dir", str(self.base / "unused"), "--model", "sl8"])
        self.assertTrue(all(getattr(default_parsed, key) == value for key, value in queue.DEFAULTS.items()))

    def test_existing_queue_and_lock_are_never_overwritten(self):
        self.assertEqual(self.execute(), 0)
        before = (self.root / "runs/trial/queue.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.execute()
        self.assertEqual((self.root / "runs/trial/queue.json").read_bytes(), before)

    def test_gpu_lock_contention_launches_nothing(self):
        (self.root / "runs").mkdir()
        with (self.root / "runs/gpu0.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.execute()
        self.assertFalse((self.root / "runs/trial").exists())
        self.assertEqual(self.children, [])

    def test_source_change_aborts_and_salvages_without_accepting_score(self):
        self.mode = "source_change"
        self.assertEqual(self.execute(), 1)
        self.assertEqual(self.state()["status"], "failed")
        self.assertEqual(self.state()["validation_records"], [])
        self.assertEqual(len(self.children), 2)

    def test_plan_unknown_fields_duplicate_ids_and_smoke_screen_rejected(self):
        for addition in ({"command": "anything"}, {"stage": "screen"}, {"model": "distmult"}, {"epochs": True}):
            raw = json.loads(json.dumps(self.raw))
            raw["jobs"][0].update(addition)
            queue.save_json(self.plan, raw)
            with self.subTest(addition=addition), self.assertRaises(ValueError):
                self.execute()
        self.raw["jobs"][1]["job_id"] = self.raw["jobs"][0]["job_id"]
        queue.save_json(self.plan, self.raw)
        with self.assertRaises(ValueError):
            self.execute()
        self.assertEqual(self.children, [])

    def test_fixed_deadline_cannot_be_extended_or_omitted(self):
        self.raw["deadline_utc"] = "2026-09-06T12:17:36Z"
        queue.save_json(self.plan, self.raw)
        with self.assertRaises(ValueError):
            self.execute()
        self.assertEqual(self.children, [])
        with self.assertRaises(Exception):
            queue.parse_utc("2026-09-06T12:17:36")

    def test_real_backup_subprocess_copies_once_and_ignores_partial_checkpoint(self):
        source = self.base / "finished"
        source.mkdir()
        (source / "best.pt").write_bytes(b"small checkpoint")
        (source / "best.pt.tmp").write_bytes(b"incomplete write")
        log, state = self.base / "train.log", self.base / "queue.json"
        log.write_text("synthetic log\n")
        state.write_text("{}")
        destination = self.base / "copied"
        command = [sys.executable, str(ROOT / "scripts/run_wn9_geometry_v2_queue.py"), "--_backup-worker",
                   str(source), str(log), str(destination), str(state)]
        subprocess.run(command, check=True, capture_output=True, timeout=10)
        self.assertTrue((destination / "best.pt").is_file())
        self.assertFalse((destination / "best.pt.tmp").exists())
        self.assertNotEqual(subprocess.run(command, capture_output=True, timeout=10).returncode, 0)


if __name__ == "__main__":
    unittest.main()
