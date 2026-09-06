"""No-GPU tests for the one-shot, same-deadline profile-to-screen gate."""
import copy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("structural_followon", ROOT / "scripts/start_structural_screen_after_profile.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


class FollowOnTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.deadline = datetime.now(timezone.utc) + timedelta(hours=5)
        self.plan = self.root / "screen.json"
        raw = {"schema_version": 1, "deadline_utc": self.deadline.isoformat(),
               "common": {"data_root": "data", "chart_radius": 2, "epochs": 20, "eval_every": 5},
               "jobs": [{"job_id": f"{d}-{m}", "dataset": d, "model": m, "stage": "screen"}
                        for d in gate.queue.DATASETS for m in gate.queue.MODELS]}
        gate.queue.save_json(self.plan, raw)
        _, self.jobs = gate.queue.load_plan(self.plan, self.deadline, self.root)
        self.smoke, self.profile = {}, {}
        for job in self.jobs:
            cfg = job["config"]
            key = (cfg["dataset"], cfg["model"])
            counts = gate.queue.DATASETS[cfg["dataset"]]
            old = dict(cfg, epochs=1, eval_every=1, kind="smoke", max_train_batches=3, eval_limit=32)
            result = {"train_count": counts[2], "validation_total_count": counts[3], "validation_count": 32,
                      "last_epoch": {"train_seconds": .6, "train_batches": 3, "validation": {"elapsed_seconds": .3}}}
            self.smoke[key] = (old, result)
            if cfg["model"] == "sl8":
                full = copy.deepcopy(result)
                full["validation_count"] = counts[3]
                full["last_epoch"] = {"train_seconds": 60, "train_batches": gate.math.ceil(counts[2] / cfg["batch_size"]),
                                      "validation": {"elapsed_seconds": 120}}
                self.profile[key] = (dict(old, kind="pilot", max_train_batches=None, eval_limit=None), full)
        self.profile_dir = self.root / "profile"
        self.profile_dir.mkdir()
        gate.queue.save_json(self.profile_dir / "queue.json", {"status": "completed"})
        self.decision = self.root / "decision.json"
        self.argv = ["--root", str(self.root), "--profile-queue", str(self.profile_dir),
                     "--smoke-queue", str(self.root / "smoke"), "--plan", str(self.plan),
                     "--queue-id", "screen", "--deadline-utc", self.deadline.isoformat(),
                     "--backup-root", str(self.root / "backup"), "--decision-file", str(self.decision)]

    def run_main(self, fingerprints=None):
        with patch.object(gate, "completed_records", side_effect=[self.profile, self.smoke]), \
             patch.object(gate.queue, "source_fingerprints", side_effect=fingerprints, return_value={"frozen": "same"}), \
             patch.object(gate.subprocess, "call", return_value=0) as child:
            code = gate.main(self.argv)
        return code, child

    def test_cost_and_four_evaluations(self):
        cost = gate.estimate(self.jobs, self.smoke, self.profile)
        self.assertEqual(len(cost["jobs"]), 6)
        self.assertTrue(all(j["scheduled_validations"] == 4 for j in cost["jobs"]))
        self.assertAlmostEqual(cost["required_seconds"], cost["base_seconds"] * 1.2 + 120)
        sl = [j for j in cost["jobs"] if j["timing_source"] == "complete_SL_epoch"]
        self.assertTrue(all(j["seconds"] == 20 * 60 + 4 * 120 for j in sl))

    def test_matched_config_and_authorized_scope(self):
        for field, value in (("chart_radius", 1.5), ("epochs", 15), ("lr", .03), ("score_chunk", 2048)):
            jobs = copy.deepcopy(self.jobs)
            jobs[0]["config"][field] = value
            with self.assertRaises(ValueError):
                gate.estimate(jobs, self.smoke, self.profile)

    def test_nan_timing_rejected(self):
        self.profile[("WN18RR", "sl8")][1]["last_epoch"]["train_seconds"] = float("nan")
        with self.assertRaises(ValueError):
            gate.estimate(self.jobs, self.smoke, self.profile)

    def test_dispatch_keeps_original_deadline(self):
        code, child = self.run_main()
        self.assertEqual(code, 0)
        command = child.call_args.args[0]
        self.assertIn(self.deadline.isoformat(), command)
        self.assertEqual(command[2], str(self.root / "scripts/run_structural_geometry_queue.py"))
        self.assertEqual(json.loads(self.decision.read_text())["status"], "queue_returned")

    def test_cost_gate_does_not_launch(self):
        for _, result in self.profile.values():
            result["last_epoch"]["train_seconds"] = 10000
        code, child = self.run_main()
        self.assertEqual(code, 2)
        child.assert_not_called()
        self.assertEqual(json.loads(self.decision.read_text())["status"], "stopped_cost_gate")

    def test_source_change_blocks_dispatch(self):
        code, child = self.run_main([{"frozen": "one"}, {"frozen": "two"}])
        self.assertEqual(code, 1)
        child.assert_not_called()

    def test_decision_never_overwritten(self):
        self.decision.write_text("existing")
        with self.assertRaises(FileExistsError):
            self.run_main()
        self.assertEqual(self.decision.read_text(), "existing")

    def test_prerequisite_source_tampering_rejected(self):
        state = {"status": "completed", "deadline_utc": self.deadline.isoformat(),
                 "source_sha256": {str(self.plan): gate.queue.digest(self.plan)}, "plan_sha256": gate.queue.digest(self.plan)}
        gate.queue.save_json(self.profile_dir / "queue.json", state)
        with patch.object(gate.queue, "source_fingerprints", return_value={"changed": "hash"}):
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                gate.completed_records(self.root, self.profile_dir, self.deadline, "profile", 2)

    def prepare_profile_artifacts(self):
        original = self.root / "profile-original.json"
        raw = {"schema_version": 1, "deadline_utc": self.deadline.isoformat(),
               "jobs": [{"job_id": f"profile-{d}", "stage": "profile", **cfg}
                        for (d, _), (cfg, _) in self.profile.items()]}
        gate.queue.save_json(original, raw)
        gate.queue.save_json(self.profile_dir / "plan.json", raw)
        _, jobs = gate.queue.load_plan(original, self.deadline, self.root)
        results = []
        for job in jobs:
            run, backup = self.profile_dir / job["job_id"], self.root / "persistent" / job["job_id"]
            for directory in (run, backup):
                directory.mkdir(parents=True)
                (directory / "result.json").write_text('{"fixture": true}')
                (directory / "best.pt").write_bytes(b"same synthetic checkpoint")
            job.update(status="completed", backup_status="completed", run_dir=str(run), backup_path=str(backup))
            result = copy.deepcopy(self.profile[(job["config"]["dataset"], "sl8")][1])
            result.update(training_complete=True, validation_scope="full")
            diagnostic = {"sampled_reference_accuracy_passed": True}
            result["initial_diagnostics"] = dict(diagnostic)
            result["last_epoch"]["diagnostics"] = dict(diagnostic)
            result["best_checkpoint_diagnostics"] = dict(diagnostic)
            results.append(result)
        state = {"status": "completed", "deadline_utc": self.deadline.isoformat(), "jobs": jobs,
                 "source_sha256": gate.queue.source_fingerprints(self.root, original), "plan_sha256": gate.queue.digest(original)}
        gate.queue.save_json(self.profile_dir / "queue.json", state)
        return results, jobs

    def test_completed_records_reuses_checker_and_accepts_backups(self):
        results, jobs = self.prepare_profile_artifacts()
        with patch.object(gate.queue, "checked_result", side_effect=results) as checker:
            records = gate.completed_records(self.root, self.profile_dir, self.deadline, "profile", 2)
        self.assertEqual(len(records), 2)
        self.assertEqual(checker.call_count, 2)

    def test_reference_accuracy_false_blocks_prerequisite(self):
        results, _ = self.prepare_profile_artifacts()
        results[0]["last_epoch"]["diagnostics"]["sampled_reference_accuracy_passed"] = False
        with patch.object(gate.queue, "checked_result", side_effect=results):
            with self.assertRaisesRegex(ValueError, "accuracy"):
                gate.completed_records(self.root, self.profile_dir, self.deadline, "profile", 2)

    def test_backup_corruption_blocks_prerequisite(self):
        results, jobs = self.prepare_profile_artifacts()
        (Path(jobs[0]["backup_path"]) / "best.pt").write_bytes(b"different")
        with patch.object(gate.queue, "checked_result", side_effect=results):
            with self.assertRaisesRegex(ValueError, "backup"):
                gate.completed_records(self.root, self.profile_dir, self.deadline, "profile", 2)

    def test_real_parent_plan_is_supported(self):
        path = ROOT / "experiments/structural_geometry_screen.json"
        raw = json.loads(path.read_text())
        _, jobs = gate.queue.load_plan(path, gate.queue.parse_utc(raw["deadline_utc"]), ROOT)
        self.assertEqual(len(jobs), 6)
        self.assertTrue(all(j["config"]["epochs"] == 20 and j["config"]["chart_radius"] == 2 for j in jobs))


if __name__ == "__main__":
    unittest.main()
