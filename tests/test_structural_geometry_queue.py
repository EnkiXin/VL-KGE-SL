"""Queue safety/provenance tests using owned synthetic children, no GPU."""

from contextlib import redirect_stdout
import copy
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
SPEC = importlib.util.spec_from_file_location("structural_queue", ROOT / "scripts/run_structural_geometry_queue.py")
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


class StructuralQueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.root = self.base / "project"
        (self.root / "scripts").mkdir(parents=True)
        (self.root / "scripts/run_structural_geometry.py").write_text("# synthetic trainer\n")
        self.plan = self.base / "plan.json"
        self.deadline = (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat()
        self.raw = {"schema_version": 1, "deadline_utc": self.deadline,
                    "common": {"kind": "smoke", "epochs": 2, "max_train_batches": 3, "eval_limit": 32,
                               "dataset": "WN18RR", "data_root": str(self.base / "data")},
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
                (run_dir / "last.pt").write_bytes(b"synthetic last checkpoint")
                if outer.mode != "timeout_no_best":
                    (run_dir / "best.pt").write_bytes(b"synthetic checkpoint")
                if outer.mode in ("timeout", "timeout_no_best"):
                    raise subprocess.TimeoutExpired(self.command, timeout)
                if outer.mode == "interrupt":
                    raise KeyboardInterrupt("synthetic interruption")
                config = {**queue.DEFAULTS, "model": value("--model"), "dataset": value("--dataset"), "data_root": value("--data-root")}
                for key, default in queue.DEFAULTS.items():
                    flag = "--" + key.replace("_", "-")
                    if flag in self.command:
                        config[key] = (int if default is None else type(default))(value(flag))
                entities, relations, train_count, valid_count, test_count = queue.DATASETS[config["dataset"]]
                count = min(config["eval_limit"] or valid_count, valid_count)
                offset = 17 if config["dataset"] == "FB15k-237" and count < valid_count else 0
                indices = list(range(offset, offset + count))
                backend = config["log_backend"]
                terms = 12 if backend == "gregory12" else config["log_order"]
                diagnostic = {"geometry": config["model"], "log_backend": backend,
                              "sampled_health_passed": True,
                              "model_contract": {"log_backend": backend, "log_terms": terms, "entity_bias": False, "distance_power": 1, "dimension": 63}}
                if config["model"] == "sl8":
                    diagnostic["principal_log"] = {"backend": backend}
                result = {"status": "completed_validation_only", "test_evaluated": False,
                          "model": config["model"], "evaluation_protocol": queue.PROTOCOL,
                          "training_negative_filter": "train_only_directional",
                          "validation_filter": "all_splits_directional", "tie_policy": "realistic_average",
                          "negative_sampling": "uniform_without_replacement_per_triple",
                          "score_distance_power": 1, "optimizer": "Adagrad", "dataset": config["dataset"],
                          "log_backend": backend, "log_terms": terms,
                          "initial_diagnostics": diagnostic,
                          "last_epoch": {"diagnostics": diagnostic},
                          "best_checkpoint_diagnostics": diagnostic,
                          "config": config, "best_validation_mrr": .4, "best_epoch": config["epochs"],
                          "completed_epochs": config["epochs"], "training_complete": config["max_train_batches"] is None or config["max_train_batches"] >= (train_count + config["batch_size"] - 1) // config["batch_size"],
                          "validation_scope": "full" if count == valid_count else "subset",
                          "validation_count": count, "validation_total_count": valid_count,
                          "validation_indices": indices,
                          "subset_indices_hash": hashlib.sha256(struct.pack("<" + "q" * count, *indices)).hexdigest(),
                          **outer.result_changes}
                full_batches = (train_count + config["batch_size"] - 1) // config["batch_size"]
                batches = min(config["max_train_batches"] or full_batches, full_batches)
                validation_epochs = [e for e in range(1, config["epochs"] + 1) if e % config["eval_every"] == 0 or e == config["epochs"]]
                events = [{"epoch": e, "train_batches": batches,
                           "train_examples": min(train_count, batches * config["batch_size"]),
                           "validation": {"evaluated_triples": count, "directional_queries": 2 * count, "num_entities": entities}
                           if e in validation_epochs else None} for e in range(1, config["epochs"] + 1)]
                (run_dir / "epochs.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
                result.update(num_entities=entities, num_relations=relations, train_count=train_count,
                              split_sizes=[train_count, valid_count, test_count], validation_epochs=validation_epochs,
                              total_optimizer_steps=batches * config["epochs"], entity_bias=False, coordinate_dim=63,
                              parameters_trainable=(entities + relations) * 63 + 2, parameters_total=(entities + relations) * 63 + 2,
                              dataset_commit="2e440e0f9c687314d5ff67ead68ce985dc446e3a",
                              dataset_provenance={"dataset":config["dataset"],"commit":"2e440e0f9c687314d5ff67ead68ce985dc446e3a"})
                queue.save_json(run_dir / "result.json", result)
                if outer.mode == "source_change":
                    (outer.root / "scripts/run_structural_geometry.py").write_text("# changed source\n")
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
                self.assertTrue(child.command[2].endswith("run_structural_geometry.py"))
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
                        {"score_distance_power": 2}, {"dataset": "WN9-IMG"}, {"optimizer": "Adam"},
                        {"log_backend": "gauss_legendre"}, {"log_terms": 16},
                        {"config": {**original_result["config"], "lr": 9.0}}):
            with self.subTest(changes=changes):
                (run / "result.json").write_text(json.dumps({**original_result, **changes}))
                with self.assertRaises(ValueError):
                    queue.checked_result(run, original_job)

    def test_gl_is_rejected_in_this_gregory_only_queue(self):
        self.raw["common"].update(log_backend="gauss_legendre", log_order=32)
        queue.save_json(self.plan, self.raw)
        with self.assertRaises(ValueError):
            self.execute()
        self.assertEqual(self.children, [])

    def test_missing_or_cross_backend_model_diagnostics_are_rejected(self):
        self.assertEqual(self.execute(), 0)
        job = self.state()["jobs"][2]  # SL requires its actual principal-log backend too.
        run = Path(job["run_dir"])
        original = json.loads((run / "result.json").read_text())
        for field in ("initial_diagnostics", "last_epoch", "best_checkpoint_diagnostics"):
            for damage in ("missing", "wrong_backend", "wrong_order", "wrong_principal", "failed_health"):
                result = copy.deepcopy(original)
                if damage == "missing":
                    result.pop(field)
                else:
                    diagnostic = result[field]["diagnostics"] if field == "last_epoch" else result[field]
                    if damage == "wrong_backend":
                        diagnostic["log_backend"] = "gauss_legendre"
                    elif damage == "wrong_order":
                        diagnostic["model_contract"]["log_terms"] = 16
                    elif damage == "wrong_principal":
                        diagnostic["principal_log"]["backend"] = "gauss_legendre"
                    else:
                        diagnostic["sampled_health_passed"] = False
                queue.save_json(run / "result.json", result)
                with self.subTest(field=field, damage=damage), self.assertRaises(ValueError):
                    queue.checked_result(run, job)

    def test_unrecognized_backend_rejected_before_queue_creation(self):
        self.raw["common"]["log_backend"] = "gregory16"
        queue.save_json(self.plan, self.raw)
        with self.assertRaises(ValueError):
            queue.load_plan(self.plan, queue.parse_utc(self.deadline))

    def test_command_parses_with_real_trainer_and_resolves_same_settings(self):
        spec = importlib.util.spec_from_file_location("structural_real_parser", ROOT / "scripts/run_structural_geometry.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        _, jobs = queue.load_plan(self.plan, queue.parse_utc(self.deadline))
        for job in jobs:
            command = queue.runner_command(self.root, self.base / "unused", job["config"])
            parsed = runner.parse_args(command[3:])
            self.assertTrue(parsed.validation_only)
            self.assertTrue(all((str(getattr(parsed, key)) if isinstance(getattr(parsed, key), Path)
                                 else getattr(parsed, key)) == value for key, value in job["config"].items()))
        default_parsed = runner.parse_args(["--root", str(self.root), "--run-dir", str(self.base / "unused"), "--model", "sl8", "--dataset", "WN18RR", "--data-root", str(self.base / "data")])
        self.assertTrue(all(getattr(default_parsed, key) == value for key, value in queue.DEFAULTS.items()))

    def test_actual_smoke_and_profile_plans_parse_with_real_trainer(self):
        spec = importlib.util.spec_from_file_location("structural_real_plan_parser", ROOT / "scripts/run_structural_geometry.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        for filename, count in (("structural_geometry_smoke.json", 6), ("structural_geometry_profile.json", 2)):
            _, jobs = queue.load_plan(ROOT / "experiments" / filename,
                                     queue.parse_utc("2026-09-06T12:17:36Z"), ROOT)
            self.assertEqual(len(jobs), count)
            for job in jobs:
                parsed = runner.parse_args(queue.runner_command(ROOT, self.base / "unused", job["config"])[3:])
                self.assertEqual(str(parsed.data_root), job["config"]["data_root"])
                self.assertEqual(parsed.dataset, job["config"]["dataset"])
                self.assertEqual(parsed.log_backend, "gregory12")

    def test_both_datasets_keep_different_subset_hashes_without_false_mismatch(self):
        extra = [{**job, "job_id": "fb-" + job["job_id"], "dataset": "FB15k-237"} for job in self.raw["jobs"]]
        self.raw["jobs"] += extra
        queue.save_json(self.plan, self.raw)
        self.assertEqual(self.execute(), 0)
        rows = self.state()["jobs"]
        self.assertEqual(len(rows), 6)
        self.assertNotEqual(rows[0]["validated_result"]["subset_indices_hash"], rows[3]["validated_result"]["subset_indices_hash"])
        self.assertEqual(rows[0]["validated_result"]["validation_total_count"], 3034)
        self.assertEqual(rows[3]["validated_result"]["validation_total_count"], 17535)

    def test_optimizer_steps_and_validation_cadence_are_verified(self):
        self.assertEqual(self.execute(), 0)
        job = self.state()["jobs"][0]
        run_dir = Path(job["run_dir"])
        original = json.loads((run_dir / "result.json").read_text())
        for changes in ({"total_optimizer_steps": 1}, {"validation_epochs": [1, 2]},
                        {"training_complete": True}, {"split_sizes": [1, 3034, 3134]},
                        {"parameters_trainable": 999}, {"entity_bias": True}):
            queue.save_json(run_dir / "result.json", {**original, **changes})
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                queue.checked_result(run_dir, job)
        queue.save_json(run_dir / "result.json", original)
        events = [json.loads(line) for line in (run_dir / "epochs.jsonl").read_text().splitlines()]
        events[0]["train_batches"] = 1
        (run_dir / "epochs.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n")
        with self.assertRaises(ValueError):
            queue.checked_result(run_dir, job)

    def test_interrupt_before_first_validation_salvages_last_checkpoint(self):
        self.mode = "timeout_no_best"
        self.assertEqual(self.execute(), 124)
        copied = self.backup / "trial/smoke-euclidean-partial"
        self.assertTrue((copied / "last.pt").is_file())
        self.assertFalse((copied / "best.pt").exists())
        self.assertEqual(self.state()["validation_records"], [])

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
        command = [sys.executable, str(ROOT / "scripts/run_structural_geometry_queue.py"), "--_backup-worker",
                   str(source), str(log), str(destination), str(state)]
        subprocess.run(command, check=True, capture_output=True, timeout=10)
        self.assertTrue((destination / "best.pt").is_file())
        self.assertFalse((destination / "best.pt.tmp").exists())
        self.assertNotEqual(subprocess.run(command, capture_output=True, timeout=10).returncode, 0)


if __name__ == "__main__":
    unittest.main()
