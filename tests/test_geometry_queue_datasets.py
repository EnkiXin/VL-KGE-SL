"""CPU-only dataset routing, staged queue execution, and failure boundaries."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_geometry_queue.py"
SPEC = importlib.util.spec_from_file_location("geometry_queue_datasets", SCRIPT)
QUEUE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUEUE)

FAKE_RUNNER = '''import json, os, sys
from pathlib import Path
it = iter(sys.argv[1:])
a = {key.lstrip("-").replace("-", "_"): next(it) for key in it}
fail = (os.environ.get("FAIL_ALL") == "1" or
        a["lr"] == os.environ.get("FAIL_LR") or
        a["kind"] == os.environ.get("FAIL_KIND"))
print("fake runner diagnostic", flush=True)
if fail and os.environ.get("FAIL_BEFORE_MKDIR") == "1":
    raise SystemExit(7)
d = Path(a["run_dir"]); d.mkdir()
(d / "args.json").write_text(json.dumps(a))
if fail:
    (d / "partial.txt").write_text("partial checkpoint marker")
    raise SystemExit(7)
r = {"status": "completed", "best_validation_mrr": float(a["lr"]), "best_epoch": int(a["epochs"])}
if a.get("selection_split") == "test":
    score = 0.9 - float(a["lr"])
    r.pop("best_validation_mrr")
    r.update(selection_split="test", test_used_for_selection=True,
             evaluation_label="test_tuned_not_held_out", best_selection_mrr=score,
             best_test_mrr=score, test_selection={"mrr": score})
if a["kind"] == "formal" or os.environ.get("PILOT_TEST") == "1":
    r["final_test"] = {"mrr": 0.8, "hits": {"10": 0.9}}
    if a.get("selection_split") == "test":
        r["final_test"].update(mrr=r["best_test_mrr"], selection_split="test",
                               test_used_for_selection=True, evaluation_label="test_tuned_not_held_out")
if os.environ.get("BAD_RESULT") == "1":
    r["best_test_mrr" if a.get("selection_split") == "test" else "best_validation_mrr"] = float("nan")
if os.environ.get("BAD_SELECTION_METADATA") == "1":
    r.pop("evaluation_label", None)
if os.environ.get("BAD_SELECTION_SCORE") == "1":
    r["best_selection_mrr"] = 0.123
if os.environ.get("BAD_FINAL_LABEL") == "1":
    r.get("final_test", {}).pop("evaluation_label", None)
(d / "result.json").write_text(json.dumps(r))
'''


class DatasetArgumentTests(unittest.TestCase):
    def args(self, *extra):
        return QUEUE.parse_args(["--root", "/unused", "--queue-id", "q", *extra])

    def test_dataset_epoch_defaults_and_validation_cadence(self):
        for dataset, epochs in [("wn9_img", (10, 200)), ("wikiart_mkg_v1", (5, 50)),
                                ("wikiart_mkg_v2", (2, 20))]:
            with self.subTest(dataset=dataset):
                args = self.args("--dataset", dataset)
                self.assertEqual((args.pilot_epochs, args.formal_epochs), epochs)
                self.assertEqual(args.validate_every, 1)
        self.assertEqual(self.args().dataset, "wn9_img")

    def test_explicit_epoch_overrides_preserved(self):
        args = self.args("--dataset", "wikiart_mkg_v2", "--pilot-epochs", "7", "--formal-epochs", "40")
        self.assertEqual((args.pilot_epochs, args.formal_epochs), (7, 40))

    def test_wn9_and_optional_runner_flags_absent_by_default(self):
        args = self.args()
        self.assertEqual(args.selection_split, "validation")
        command = QUEUE.runner_command(args, Path("/unused"), Path("/unused/r"), "sl8", .01, "pilot")
        for flag in ("--dataset", "--selection-split", "--batch-size", "--negatives", "--score-chunk",
                     "--candidate-chunk", "--query-batch"):
            self.assertNotIn(flag, command)

    def test_explicit_test_selection_is_forwarded(self):
        args = self.args("--selection-split", "test")
        command = QUEUE.runner_command(args, Path("/unused"), Path("/unused/r"), "sl8", .01, "pilot")
        self.assertEqual(command[command.index("--selection-split") + 1], "test")

    def test_nonwn9_and_explicit_runner_options_forwarded(self):
        options = {"batch-size": 64, "negatives": 7, "score-chunk": 128,
                   "candidate-chunk": 256, "query-batch": 4}
        extra = [item for key, value in options.items() for item in ("--" + key, str(value))]
        args = self.args("--dataset", "wikiart_mkg_v1", *extra)
        command = QUEUE.runner_command(args, Path("/unused"), Path("/unused/r"), "sl8", .01, "pilot")
        self.assertEqual(command[command.index("--dataset") + 1], "wikiart_mkg_v1")
        for flag, value in options.items():
            self.assertEqual(command[command.index("--" + flag) + 1], str(value))

    def test_invalid_options_and_mutual_exclusion(self):
        for extra in [("--dataset", "unknown"), ("--formal-only-lr", "0"),
                      ("--formal-only-lr", "nan"), ("--pilots-only", "--formal-only-lr", ".1")]:
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                self.args(*extra)
        for name in QUEUE.OPTIONAL_RUNNER_INTS:
            with self.subTest(name=name), self.assertRaises(SystemExit):
                self.args("--" + name.replace("_", "-"), "0")


class DatasetQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "scripts").mkdir()
        (self.root / "scripts" / "run_geometry.py").write_text(FAKE_RUNNER)

    def tearDown(self):
        self.temp.cleanup()

    def run_queue(self, *extra, env=None):
        command = [sys.executable, str(SCRIPT), "--root", str(self.root), "--queue-id", "q",
                   "--models", "euclidean", "sl8", "--learning-rates", ".01", ".03", *extra]
        return subprocess.run(command, capture_output=True, text=True,
                              env={**os.environ, **(env or {})}, timeout=15)

    def state(self):
        return json.loads((self.root / "runs" / "q" / "queue.json").read_text())

    def test_pilots_only_selects_without_formal_or_test(self):
        result = self.run_queue("--dataset", "wikiart_mkg_v1", "--pilots-only", "--negatives", "9")
        self.assertEqual(result.returncode, 0, result.stderr)
        state = self.state()
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["dataset"], "wikiart_mkg_v1")
        self.assertEqual(state["selection_split"], "validation")
        self.assertFalse(state["test_used_for_selection"])
        self.assertNotIn("test_selection", state)
        self.assertEqual(state["selection"], state["validation_selection"])
        self.assertEqual(state["runner_overrides"], {"negatives": 9})
        self.assertTrue(state["pilots_only"])
        self.assertEqual(len(state["jobs"]), 4)
        self.assertEqual([choice["lr"] for choice in state["validation_selection"].values()], [.03, .03])
        for job in state["jobs"]:
            self.assertEqual(job["kind"], "pilot")
            self.assertEqual(job["dataset"], "wikiart_mkg_v1")
            self.assertNotIn("final_test", job)
            args = json.loads((Path(job["run_dir"]) / "args.json").read_text())
            self.assertEqual(args["epochs"], "5")
            self.assertEqual(args["dataset"], "wikiart_mkg_v1")
            self.assertEqual(args["negatives"], "9")

    def test_formal_only_uses_explicit_lr_without_pilot_selection(self):
        result = self.run_queue("--dataset", "wikiart_mkg_v2", "--formal-only-lr", ".07")
        self.assertEqual(result.returncode, 0, result.stderr)
        state = self.state()
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["validation_selection"], {})
        self.assertEqual(state["formal_only_lr"], .07)
        self.assertEqual(len(state["jobs"]), 2)
        for job in state["jobs"]:
            self.assertEqual(job["kind"], "formal")
            self.assertEqual(job["lr"], .07)
            self.assertEqual(job["dataset"], "wikiart_mkg_v2")
            self.assertIn("final_test", job)
            args = json.loads((Path(job["run_dir"]) / "args.json").read_text())
            self.assertEqual(args["epochs"], "20")

    def test_nonwn9_full_queue_uses_dataset_defaults(self):
        result = self.run_queue("--dataset", "wikiart_mkg_v2", "--models", "sl8")
        self.assertEqual(result.returncode, 0, result.stderr)
        jobs = self.state()["jobs"]
        self.assertEqual([job["kind"] for job in jobs], ["pilot", "pilot", "formal"])
        self.assertEqual([job["best_epoch"] for job in jobs], [2, 2, 20])

    def test_failed_pilot_requires_opt_in(self):
        result = self.run_queue(env={"FAIL_LR": "0.01"})
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(self.state()["status"], "failed")
        self.assertEqual(len(self.state()["jobs"]), 1)

    def test_skip_nonzero_pilots_back_up_partials_and_select_completed(self):
        backup = self.root / "persistent"
        result = self.run_queue("--continue-failed-pilots", "--backup-root", str(backup),
                                env={"FAIL_LR": "0.03"})
        self.assertEqual(result.returncode, 0, result.stderr)
        state = self.state()
        self.assertEqual(state["status"], "completed")
        self.assertEqual([choice["lr"] for choice in state["validation_selection"].values()], [.01, .01])
        failed = [job for job in state["jobs"] if job["status"] == "failed"]
        self.assertEqual(len(failed), 2)
        for job in failed:
            self.assertEqual(job["exit_code"], 7)
            self.assertTrue(job["skipped_failed_pilot"])
            self.assertEqual(job["backup_status"], "completed")
            self.assertTrue((Path(job["backup"]) / "partial.txt").is_file())
            self.assertIn("fake runner diagnostic", (Path(job["backup"]) / "train.log").read_text())

    def test_failed_pilot_before_mkdir_still_backs_up_log(self):
        backup = self.root / "persistent"
        result = self.run_queue("--continue-failed-pilots", "--backup-root", str(backup),
                                "--models", "sl8", "--pilots-only",
                                env={"FAIL_LR": "0.03", "FAIL_BEFORE_MKDIR": "1"})
        self.assertEqual(result.returncode, 0, result.stderr)
        job = self.state()["jobs"][1]
        self.assertEqual(job["status"], "failed")
        self.assertTrue((Path(job["backup"]) / "train.log").is_file())

    def test_all_pilots_failed_cannot_launch_formal(self):
        result = self.run_queue("--continue-failed-pilots", env={"FAIL_ALL": "1"})
        self.assertEqual(result.returncode, 1, result.stderr)
        state = self.state()
        self.assertEqual(state["status"], "failed")
        self.assertIn("No completed pilots", state["error"])
        self.assertEqual(len(state["jobs"]), 4)
        self.assertTrue(all(job["kind"] == "pilot" for job in state["jobs"]))

    def test_pilot_test_contamination_is_fatal_with_continue_enabled(self):
        result = self.run_queue("--continue-failed-pilots", env={"PILOT_TEST": "1"})
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(self.state()["status"], "failed")
        self.assertIn("test set", self.state()["error"])
        self.assertEqual(len(self.state()["jobs"]), 1)

    def test_invalid_metrics_are_fatal_with_continue_enabled(self):
        result = self.run_queue("--continue-failed-pilots", env={"BAD_RESULT": "1"})
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(self.state()["status"], "failed")
        self.assertIn("best_validation_mrr", self.state()["error"])
        self.assertEqual(len(self.state()["jobs"]), 1)

    def test_formal_failure_is_not_skipped(self):
        result = self.run_queue("--continue-failed-pilots", "--formal-only-lr", ".03",
                                env={"FAIL_KIND": "formal"})
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(self.state()["status"], "failed")
        self.assertEqual(len(self.state()["jobs"]), 1)

    def test_failed_pilot_backup_collision_remains_fatal(self):
        backup = self.root / "persistent"
        target = backup / "q" / "pilot-euclidean-lr-0p01-seed42"
        target.mkdir(parents=True)
        (target / "keep.txt").write_text("user file")
        result = self.run_queue("--continue-failed-pilots", "--backup-root", str(backup),
                                env={"FAIL_LR": "0.01"})
        self.assertEqual(result.returncode, 1, result.stderr)
        state = self.state()
        self.assertEqual(state["status"], "backup_failed")
        self.assertEqual(state["jobs"][0]["backup_status"], "failed")
        self.assertEqual(len(state["jobs"]), 1)
        self.assertEqual((target / "keep.txt").read_text(), "user file")

    def test_explicit_test_selection_uses_test_scores_and_keeps_labels(self):
        result = self.run_queue("--selection-split", "test")
        self.assertEqual(result.returncode, 0, result.stderr)
        state = self.state()
        self.assertEqual(state["selection_split"], "test")
        self.assertTrue(state["test_used_for_selection"])
        self.assertEqual(state["evaluation_label"], "test_tuned_not_held_out")
        self.assertIn("test MRR", state["selection_policy"])
        self.assertNotIn("validation_selection", state)
        self.assertEqual(state["selection"], state["test_selection"])
        # Fake test scores prefer .01, while legacy validation scores prefer .03.
        self.assertEqual([choice["lr"] for choice in state["selection"].values()], [.01, .01])
        for job in state["jobs"]:
            self.assertNotIn("best_validation_mrr", job)
            self.assertEqual(job["best_selection_mrr"], job["best_test_mrr"])
            self.assertEqual(job["evaluation_label"], "test_tuned_not_held_out")
            self.assertTrue(job["test_used_for_selection"])
            if job["kind"] == "formal":
                self.assertEqual(job["lr"], .01)
                self.assertEqual(job["final_test"]["evaluation_label"], "test_tuned_not_held_out")
                self.assertTrue(job["final_test"]["test_used_for_selection"])
            else:
                self.assertNotIn("final_test", job)
                self.assertIn("test_selection", job)

    def test_test_selected_pilots_only_do_not_claim_formal_test(self):
        result = self.run_queue("--selection-split", "test", "--pilots-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        state = self.state()
        self.assertEqual(state["status"], "completed")
        self.assertEqual(len(state["jobs"]), 4)
        self.assertTrue(all(job["kind"] == "pilot" and "final_test" not in job for job in state["jobs"]))

    def test_test_selected_formal_only_does_not_fabricate_pilot_selection(self):
        result = self.run_queue("--selection-split", "test", "--formal-only-lr", ".07")
        self.assertEqual(result.returncode, 0, result.stderr)
        state = self.state()
        self.assertEqual(state["selection"], {})
        self.assertEqual(state["test_selection"], {})
        self.assertNotIn("validation_selection", state)
        self.assertTrue(all(job["lr"] == .07 and job["kind"] == "formal" for job in state["jobs"]))

    def test_test_selection_missing_metadata_is_fatal(self):
        result = self.run_queue("--selection-split", "test", "--continue-failed-pilots",
                                env={"BAD_SELECTION_METADATA": "1"})
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("test_tuned_not_held_out", self.state()["error"])
        self.assertEqual(len(self.state()["jobs"]), 1)

    def test_test_selection_nonfinite_metric_is_fatal(self):
        result = self.run_queue("--selection-split", "test", "--continue-failed-pilots",
                                env={"BAD_RESULT": "1"})
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("best_test_mrr", self.state()["error"])

    def test_generic_selection_score_cannot_disagree_with_test_metric(self):
        result = self.run_queue("--selection-split", "test", env={"BAD_SELECTION_SCORE": "1"})
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("best_selection_mrr", self.state()["error"])

    def test_test_selected_formal_requires_label_on_final_test(self):
        result = self.run_queue("--selection-split", "test", "--formal-only-lr", ".07",
                                env={"BAD_FINAL_LABEL": "1"})
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("Formal test-selected result", self.state()["error"])


class SelectionResultTests(unittest.TestCase):
    def test_validation_mode_rejects_test_selection_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = {"status": "completed", "best_validation_mrr": .8, "best_epoch": 2,
                      "selection_split": "test", "best_test_mrr": .9,
                      "test_used_for_selection": True, "evaluation_label": "test_tuned_not_held_out"}
            (root / "result.json").write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "unexpectedly used the test set"):
                QUEUE.read_result(root, "pilot")

    def test_test_selected_pilot_accepts_selection_record_but_not_final_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = {"status": "completed", "best_test_mrr": .8, "best_selection_mrr": .8,
                      "best_epoch": 2, "selection_split": "test", "test_used_for_selection": True,
                      "evaluation_label": "test_tuned_not_held_out", "test_selection": {"mrr": .8},
                      "final_test": None}
            (root / "result.json").write_text(json.dumps(result))
            self.assertEqual(QUEUE.read_result(root, "pilot", "test")["best_test_mrr"], .8)
            result["final_test"] = {"mrr": .8}
            (root / "result.json").write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "Pilot unexpectedly reported final_test"):
                QUEUE.read_result(root, "pilot", "test")


if __name__ == "__main__":
    unittest.main()
