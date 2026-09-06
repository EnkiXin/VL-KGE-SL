"""Controller tests with fake queues and /proc snapshots; no GPU or SSH."""

from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import signal
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("three_dataset_controller", ROOT / "scripts/run_sl28_three_datasets.py")
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)


class ThreeDatasetControllerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.root = self.base / "new-root"
        self.backup = self.base / "backup"
        (self.root / "scripts").mkdir(parents=True)
        for name in ("run_geometry.py", "run_geometry_queue.py"):
            (self.root / "scripts" / name).write_text("# fake runner, never executed\n")
        self.deadline = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        self.argv = ["--root", str(self.root), "--backup-root", str(self.backup),
                     "--deadline-utc", self.deadline]
        self.args = controller.parse_args(self.argv)

    def process(self, pid=100, queue="q", runner=False, root=None, pgid=None):
        root = self.root if root is None else root
        script = "run_geometry.py" if runner else "run_geometry_queue.py"
        command = ["python", "-u", str(root / "scripts" / script), "--root", str(root)]
        if runner:
            command += ["--run-dir", str(root / "runs" / queue / "pilot")]
        else:
            command += ["--queue-id", queue]
        return dict(pid=pid, ppid=1, pgid=pid if pgid is None else pgid,
                    sid=pid if pgid is None else pgid, starttime=500, cwd=str(root), argv=command)

    def pilot(self, queue, lr, score, dataset="wn9_img", split="test", status="completed"):
        run_dir = queue / f"pilot-{lr}"
        run_dir.mkdir(parents=True)
        key = "best_test_mrr" if split == "test" else "best_validation_mrr"
        event_key = "test_selection" if split == "test" else "validation"
        result = dict(status=status, kind="pilot", dataset=dataset, selection_split=split,
                      test_used_for_selection=split == "test", best_selection_mrr=score,
                      evaluation_label="test_tuned_not_held_out" if split == "test" else "validation_selected",
                      best_epoch=1, final_test=None, **{key: score},
                      last_epoch={event_key: {"complete": True}},
                      args=dict(seed=42, lr=lr, matrix_dim=28, embedding_dim=768, entity_mapping="linear"))
        (run_dir / "result.json").write_text(json.dumps(result))
        return dict(kind="pilot", status=status, model="slv2n28", lr=lr, run_dir=str(run_dir)), result

    def test_deadline_required_utc_future_and_bounded(self):
        now = datetime(2026, 9, 6, tzinfo=timezone.utc).timestamp()
        self.assertEqual(controller.deadline_timestamp("2026-09-07T00:00:00Z", now), now + 86400)
        for value in ("2026-09-08T00:00:00Z", "2026-09-06T00:00:00Z",
                      "2026-09-06T01:00:00", "2026-09-06T10:00:00+09:00"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                controller.deadline_timestamp(value, now)
        with self.assertRaises(SystemExit):
            controller.parse_args(self.argv + ["--preceding-root", str(self.base / "old")])

    def test_six_stage_weights_share_absolute_deadline(self):
        self.assertEqual([stage[:2] for stage in controller.STAGES],
                         [(ds, kind) for kind in ("pilot", "formal")
                          for ds in ("wn9_img", "wikiart_mkg_v1", "wikiart_mkg_v2")])
        now = 1000.0
        durations = []
        for index in range(6):
            end = controller.stage_deadline(now, 1900.0, index)
            durations.append(end - now)
            now = end
        self.assertEqual(durations, [100, 100, 100, 200, 200, 200])
        self.assertEqual(controller.stage_deadline(2000, 1900, 0), 1900)

    def test_commands_keep_user_math_and_author_dataset_sizes(self):
        for dataset in controller.EPOCHS:
            command = controller.queue_command(self.args, dataset, "pilot", "q", 1)
            self.assertEqual(controller.flag(command, "--selection-split"), "test")
            self.assertEqual(controller.flag(command, "--embedding-dim"), "768")
            self.assertEqual(controller.flag(command, "--matrix-dim"), "28")
            self.assertEqual(controller.flag(command, "--entity-mapping"), "linear")
            self.assertEqual(controller.flag(command, "--chart-radius"), "2")
            self.assertEqual(controller.flag(command, "--batch-size"), "512")
            self.assertEqual(controller.flag(command, "--negatives"), "100" if dataset == "wn9_img" else "1")
            self.assertEqual(controller.flag(command, "--validate-every"), "5" if dataset == "wn9_img" else "1")
            self.assertIn("--pilots-only", command)
            self.assertIn("--continue-failed-pilots", command)
            offset = command.index("--learning-rates") + 1
            self.assertEqual(command[offset:offset+6], list(map(str, controller.LRS)))
            formal = controller.queue_command(self.args, dataset, "formal", "formal", 1, 0.03)
            self.assertEqual(controller.flag(formal, "--formal-only-lr"), "0.03")
            self.assertNotIn("--pilots-only", formal)
            smoke = controller.smoke_command(self.args, dataset, self.root / "smoke")
            self.assertEqual(controller.flag(smoke, "--max-train-batches"), "1")
            self.assertEqual(controller.flag(smoke, "--eval-limit"), "2")

    def test_preceding_terminal_state_not_enough_until_runner_exits(self):
        queue = self.root / "runs/q"
        queue.mkdir(parents=True)
        (queue / "queue.json").write_text(json.dumps({"status": "completed"}))
        child = self.process(pid=101, runner=True, pgid=100)
        self.assertFalse(controller.preceding_status(self.root, "q", [child])["ready"])
        self.assertTrue(controller.preceding_status(self.root, "q", [])["ready"])
        (queue / "queue.json").write_text(json.dumps({"status": "running"}))
        self.assertFalse(controller.preceding_status(self.root, "q", [])["ready"])

    def test_process_identity_checks_all_boundaries(self):
        item = self.process()
        self.assertTrue(controller.process_matches(item, self.root, "q"))
        self.assertFalse(controller.process_matches(item, self.root, "q-other"))
        item["cwd"] = str(self.base)
        self.assertFalse(controller.process_matches(item, self.root, "q"))
        runner = self.process(runner=True)
        self.assertTrue(controller.process_matches(runner, self.root, "q"))
        runner["argv"][-1] = str(self.root / "runs/q-similar/pilot")
        self.assertFalse(controller.process_matches(runner, self.root, "q"))
        foreign = self.process(root=self.base / "foreign")
        self.assertFalse(controller.process_matches(foreign, self.root, "q"))

    def test_owned_group_refuses_unverified_members_and_pid_reuse(self):
        members = [self.process()]
        process = SimpleNamespace(pid=100)
        owner = controller.OwnedProcess(process, self.root, "q", scan=lambda: members)
        self.assertEqual(len(owner.members()), 1)
        members.append(self.process(pid=101, pgid=100, root=self.base / "other"))
        with patch.object(controller.os, "killpg") as kill, self.assertRaises(controller.ProcessSafetyError):
            owner.stop(grace=0)
        kill.assert_not_called()
        members.pop()
        members[0]["starttime"] = 999
        with self.assertRaises(controller.ProcessSafetyError):
            owner.members()

    def test_owned_group_drains_orphan_runner_even_after_leader_exit(self):
        members = [self.process(pid=101, runner=True, pgid=100)]
        process = SimpleNamespace(pid=100, wait=lambda timeout: 0, poll=lambda: 0)
        owner = controller.OwnedProcess(process, self.root, "q", scan=lambda: members)
        calls = []
        def kill(pgid, signum):
            calls.append((pgid, signum))
            if signum == signal.SIGKILL:
                members.clear()
        with patch.object(controller.os, "killpg", side_effect=kill):
            owner.stop(grace=0)
        self.assertEqual(calls, [(100, signal.SIGINT), (100, signal.SIGKILL)])
        self.assertEqual(owner.members(), [])

    def test_selection_ignores_wrong_split_partial_results_and_uses_order_for_ties(self):
        queue = self.root / "runs/q"
        jobs = []
        for lr, score, split, status in ((0.03, 0.8, "test", "completed"),
                                         (0.1, 0.8, "test", "completed"),
                                         (0.01, 0.99, "validation", "completed"),
                                         (0.05, 0.99, "test", "interrupted")):
            job, _ = self.pilot(queue, lr, score, split=split, status=status)
            jobs.append(job)
        (queue / "queue.json").write_text(json.dumps({"status": "budget_exhausted", "jobs": jobs}))
        selection = controller.select_pilot(queue, "wn9_img")
        self.assertEqual(selection["selected"]["lr"], 0.03)
        self.assertEqual(selection["selected"]["best_selection_mrr"], 0.8)
        self.assertTrue(selection["search_incomplete"])
        self.assertEqual(len(selection["rejected_pilots"]), 1)
        path = Path(jobs[0]["run_dir"]) / "result.json"
        result = json.loads(path.read_text())
        result["last_epoch"]["test_selection"]["complete"] = False
        path.write_text(json.dumps(result))
        self.assertEqual(controller.select_pilot(queue, "wn9_img")["selected"]["lr"], 0.1)

    def fake_run(self, command, root, log, deadline, **kwargs):
        log.write_text("fake run log\n")
        dataset = controller.flag(command, "--dataset")
        if "--kind" in command:
            run_dir = Path(controller.flag(command, "--run-dir"))
            run_dir.mkdir(parents=True)
            (run_dir / "result.json").write_text(json.dumps({"status": "completed", "final_test": None}))
        else:
            queue = root / "runs" / controller.flag(command, "--queue-id")
            queue.mkdir()
            jobs = []
            if "--pilots-only" in command:
                for lr in controller.LRS:
                    job, _ = self.pilot(queue, lr, 0.8 if lr == 0.1 else 0.7, dataset)
                    jobs.append(job)
            else:
                self.assertEqual(controller.flag(command, "--formal-only-lr"), "0.1")
                jobs.append({"kind": "formal", "status": "completed", "final_test": {
                    "mrr": 0.9, "evaluation_label": "test_tuned_not_held_out"}})
            (queue / "queue.json").write_text(json.dumps({"status": "completed", "jobs": jobs}))
        return {"status": "completed", "exit_code": 0, "started": True}

    def test_fake_end_to_end_runs_all_three_anew_and_backs_up(self):
        with patch.object(controller, "run_owned", side_effect=self.fake_run) as launch:
            self.assertEqual(controller.main(self.argv), 0)
        self.assertEqual(launch.call_count, 9)  # Three smokes, three searches, three formal queues.
        control = self.root / "runs/sl28-three-datasets"
        results = json.loads((control / "all-results.json").read_text())
        state = results["controller"]
        self.assertEqual(state["evaluation_label"], "test_tuned_not_held_out")
        self.assertTrue(state["test_used_for_selection"])
        self.assertEqual(len(state["stages"]), 6)
        self.assertEqual(len(results["dataset_queues"]), 6)
        self.assertIsNone(results["legacy_preceding_queue_reference"])
        self.assertTrue((self.backup / "sl28-three-datasets/all-results.json").is_file())
        self.assertTrue((self.backup / "sl28-three-datasets/queues/sl28-three-datasets-wn9_img-pilot/queue.json").is_file())
        with self.assertRaises(FileExistsError):
            controller.main(self.argv)

    def test_smoke_failure_continues_other_datasets_but_skips_its_formal(self):
        def runner(command, root, log, deadline, **kwargs):
            if "--kind" in command and controller.flag(command, "--dataset") == "wikiart_mkg_v1":
                log.write_text("failed smoke\n")
                return {"status": "failed", "exit_code": 1, "started": True}
            return self.fake_run(command, root, log, deadline, **kwargs)
        with patch.object(controller, "run_owned", side_effect=runner):
            self.assertEqual(controller.main(self.argv), 1)
        state = json.loads((self.root / "runs/sl28-three-datasets/state.json").read_text())
        self.assertEqual(state["stages"][1]["status"], "smoke_failed")
        self.assertEqual(state["stages"][4]["status"], "skipped_no_valid_pilot")
        self.assertEqual(state["stages"][5]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
