"""Queue contract for geometry v2 model specs (CPU-only, fake runner)."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_geometry_queue.py"
SPEC = importlib.util.spec_from_file_location("geometry_queue_v2", SCRIPT)
QUEUE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUEUE)
RUNNER_SPEC = importlib.util.spec_from_file_location("geometry_runner_v2", ROOT / "scripts" / "run_geometry.py")
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER)

FAKE_RUNNER = '''import argparse, json, sys
from pathlib import Path
p = argparse.ArgumentParser()
known, unknown = p.parse_known_args()
args = {}
it = iter(unknown)
for key in it:
    args[key.lstrip("-").replace("-", "_")] = next(it)
d = Path(args["run_dir"]); d.mkdir()
(d / "args.json").write_text(json.dumps(args))
r = {"status": "completed", "best_validation_mrr": 0.8 if float(args["lr"]) == 0.03 else 0.7,
     "best_epoch": int(args["epochs"])}
if args["kind"] == "formal":
    r["final_test"] = {"mrr": 0.9, "hits": {"1": 0.8}}
(d / "result.json").write_text(json.dumps(r))
'''


class ResolveModelTests(unittest.TestCase):
    def test_specs(self):
        self.assertEqual(QUEUE.resolve_model("slv2"), ("slv2", None))
        self.assertEqual(QUEUE.resolve_model("slv2n28"), ("slv2", 28))
        self.assertEqual(QUEUE.resolve_model("euclidv2n783"), ("euclidv2", 783))
        self.assertEqual(QUEUE.resolve_model("sl8"), ("sl8", None))
        self.assertEqual(QUEUE.resolve_model("murp"), ("murp", None))


class QueueV2Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "scripts").mkdir()
        (self.root / "scripts" / "run_geometry.py").write_text(FAKE_RUNNER)

    def tearDown(self):
        self.temp.cleanup()

    def run_queue(self, *extra):
        command = [sys.executable, str(SCRIPT), "--root", str(self.root), "--queue-id", "v2", *extra]
        return subprocess.run(command, capture_output=True, text=True, env=dict(os.environ), timeout=30)

    def test_v2_specs_pass_matrix_dim_and_options(self):
        result = self.run_queue("--models", "slv2", "euclidv2", "slv2n28", "sl8",
                                "--learning-rates", "0.01", "0.03", "--matrix-dim", "8",
                                "--chart-radius", "2.0", "--initial-logit-scale", "3", "--initial-offset", "3",
                                "--relation-radius", "1.5", "--relation-init-norm", "0.5",
                                "--log-sqrt-steps", "1", "--patience", "40")
        self.assertEqual(result.returncode, 0, result.stderr)
        state = json.loads((self.root / "runs" / "v2" / "queue.json").read_text())
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["patience"], 40)
        self.assertEqual(state["geometry_v2"]["entity_mapping"], "linear")
        self.assertEqual(state["geometry_v2"]["embedding_dim"], 768)
        self.assertEqual(state["geometry_v2"]["resolved_models"]["slv2n28"], {"runner_model": "slv2", "matrix_dim": 28})
        for job in state["jobs"]:
            args = json.loads((Path(job["run_dir"]) / "args.json").read_text())
            if job["model"] == "slv2n28":
                self.assertEqual(args["model"], "slv2")
                self.assertEqual(args["matrix_dim"], "28")
            elif job["model"] == "slv2":
                self.assertEqual(args["matrix_dim"], "8")
                self.assertEqual(args["relation_radius"], "1.5")
                self.assertEqual(args["score_form"], "linear")
            elif job["model"] == "euclidv2":
                self.assertEqual(args["model"], "euclidv2")
                self.assertEqual(args["log_sqrt_steps"], "1")
            else:
                self.assertEqual(job["model"], "sl8")
                self.assertNotIn("matrix_dim", args)
                self.assertNotIn("entity_mapping", args)
            if job["model"] != "sl8":
                self.assertEqual(args["entity_mapping"], "linear")
                self.assertEqual(args["embedding_dim"], "768")
            self.assertEqual(args["chart_radius"], "2.0")
            self.assertEqual(args["patience"], "40")
        formal = [job for job in state["jobs"] if job["kind"] == "formal"]
        self.assertEqual([job["model"] for job in formal], ["slv2", "euclidv2", "slv2n28", "sl8"])
        self.assertTrue(all(job["lr"] == 0.03 for job in formal))

    def test_unknown_spec_rejected(self):
        result = self.run_queue("--models", "slv3", "--learning-rates", "0.01")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "runs" / "v2").exists())

    def test_fixed_pad_is_recorded_and_passed_for_v2_models(self):
        result = self.run_queue("--models", "slv2n28", "euclidv2", "--matrix-dim", "28",
                                "--entity-mapping", "fixed_pad", "--learning-rates", "0.01",
                                "--validate-every", "5")
        self.assertEqual(result.returncode, 0, result.stderr)
        state = json.loads((self.root / "runs" / "v2" / "queue.json").read_text())
        self.assertEqual(state["geometry_v2"]["entity_mapping"], "fixed_pad")
        for job in state["jobs"]:
            args = json.loads((Path(job["run_dir"]) / "args.json").read_text())
            self.assertEqual(args["entity_mapping"], "fixed_pad")
            self.assertEqual(args["matrix_dim"], "28")
            self.assertEqual(args["validate_every"], "5")

    def test_fixed_pad_rejects_compression_and_v1_before_creating_queue(self):
        for models, extra in [(["slv2"], []), (["slv2n27"], ["--matrix-dim", "28"]),
                              (["sl8"], ["--matrix-dim", "28"]),
                              (["slv2n28", "murp"], [])]:
            with self.subTest(models=models, extra=extra):
                result = self.run_queue("--models", *models, "--entity-mapping", "fixed_pad", *extra)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("fixed_pad", result.stderr)
                self.assertFalse((self.root / "runs").exists())

    def test_direct_783_is_recorded_and_passed(self):
        result = self.run_queue("--models", "slv2n28", "--embedding-dim", "783",
                                "--entity-mapping", "direct", "--learning-rates", "0.01")
        self.assertEqual(result.returncode, 0, result.stderr)
        state = json.loads((self.root / "runs" / "v2" / "queue.json").read_text())
        self.assertEqual(state["geometry_v2"]["entity_mapping"], "direct")
        self.assertEqual(state["geometry_v2"]["embedding_dim"], 783)
        for job in state["jobs"]:
            args = json.loads((Path(job["run_dir"]) / "args.json").read_text())
            self.assertEqual(args["entity_mapping"], "direct")
            self.assertEqual(args["embedding_dim"], "783")
            self.assertEqual(args["matrix_dim"], "28")

    def test_direct_rejects_mismatched_dimensions_and_v1_before_launch(self):
        for extra in [("--models", "slv2n28"),
                      ("--models", "slv2", "--embedding-dim", "783"),
                      ("--models", "slv2n29", "--embedding-dim", "783"),
                      ("--models", "sl8", "--matrix-dim", "28", "--embedding-dim", "783")]:
            with self.subTest(extra=extra):
                result = self.run_queue("--entity-mapping", "direct", *extra)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("direct", result.stderr)
                self.assertFalse((self.root / "runs").exists())

    def test_non_direct_keeps_author_embedding_width(self):
        for mapping in ("linear", "fixed_pad"):
            with self.subTest(mapping=mapping):
                result = self.run_queue("--models", "slv2n28", "--embedding-dim", "783",
                                        "--entity-mapping", mapping)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("embedding_dim=768", result.stderr)
                self.assertFalse((self.root / "runs").exists())


class RunnerMappingTests(unittest.TestCase):
    def args(self, *extra):
        return RUNNER.parse_args(["--root", "/unused", "--run-dir", "/unused/run",
                                  "--model", "slv2", "--chart-radius", "2", *extra])

    def test_default_remains_linear(self):
        self.assertEqual(self.args().entity_mapping, "linear")
        self.assertEqual(self.args().embedding_dim, 768)

    def test_fixed_pad_valid_and_invalid_dimensions(self):
        self.assertEqual(self.args("--entity-mapping", "fixed_pad", "--matrix-dim", "28").entity_mapping,
                         "fixed_pad")
        for extra in [("--matrix-dim", "27"), ("--model", "sl8", "--matrix-dim", "28")]:
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                self.args("--entity-mapping", "fixed_pad", *extra)

    def test_direct_requires_matching_embedding_width(self):
        args = self.args("--entity-mapping", "direct", "--matrix-dim", "28", "--embedding-dim", "783")
        self.assertEqual(args.entity_mapping, "direct")
        self.assertEqual(args.embedding_dim, 783)
        for extra in [("--matrix-dim", "28"),
                      ("--model", "sl8", "--matrix-dim", "28", "--embedding-dim", "783"),
                      ("--matrix-dim", "28", "--embedding-dim", "784")]:
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                self.args("--entity-mapping", "direct", *extra)


class LaunchScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "scripts").mkdir()
        self.script = self.root / "scripts" / "run_v2_wn9.sh"
        self.script.write_text((ROOT / "scripts" / "run_v2_wn9.sh").read_text())
        self.python = self.root / "fake_python"
        self.python.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
from pathlib import Path
args = sys.argv[1:]
if args[0] == "-c":
    os.execv(sys.executable, [sys.executable, *args])
with Path("launcher_calls.jsonl").open("a") as stream:
    stream.write(json.dumps(args) + "\\n")
if "scripts/run_geometry_queue.py" in args:
    Path("launcher_args.json").write_text(json.dumps(args))
''')
        self.python.chmod(0o755)
        self.queue_id = "geometry-v2-unit-sl28-proj783"

    def tearDown(self):
        self.temp.cleanup()

    def launch(self, **extra):
        env = {**os.environ, "PYTHON": str(self.python), "STAMP": "unit",
               "BACKUP_ROOT": str(self.root / "backup"), **extra}
        # Environment overrides from real launches must not leak into contract tests.
        for name in ("LRS", "PILOT_EPOCHS", "FORMAL_EPOCHS", "MAX_HOURS", "VALIDATE_EVERY_HIGH"):
            if name not in extra:
                env.pop(name, None)
        return subprocess.run(["bash", str(self.script)], env=env, capture_output=True, text=True, timeout=10)

    def test_default_launch_is_only_sl28_projection_768_to_783(self):
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        marker = self.root / "launcher_args.json"
        deadline = time.monotonic() + 3
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(marker.exists(), "background queue was not invoked")
        args = json.loads(marker.read_text())
        for option, expected in [("--models", "slv2n28"), ("--matrix-dim", "28"),
                                 ("--embedding-dim", "768"), ("--entity-mapping", "linear"), ("--max-hours", "24"),
                                 ("--pilot-epochs", "10"), ("--formal-epochs", "200"),
                                 ("--validate-every", "5"), ("--queue-id", self.queue_id)]:
            self.assertEqual(args[args.index(option) + 1], expected)
        self.assertEqual(args[args.index("--learning-rates") + 1:args.index("--max-hours")],
                         ["0.003", "0.01", "0.03", "0.05", "0.1", "0.2"])
        self.assertNotIn("slv2", args)
        self.assertNotIn("euclidv2", args)
        self.assertNotIn("direct", args)
        self.assertNotIn("fixed_pad", args)
        calls = [json.loads(line) for line in (self.root / "launcher_calls.jsonl").read_text().splitlines()]
        self.assertEqual(sum("scripts/run_geometry_queue.py" in call for call in calls), 1)
        self.assertTrue(any("tests.test_geometry_v2_fixed_pad" in call for call in calls))
        self.assertTrue(any("tests.test_geometry_v2_direct" in call for call in calls))
        self.assertTrue(any("tests.test_geometry_v2_projection783" in call for call in calls))

    def test_existing_outputs_are_never_overwritten(self):
        for relative in [f"runs/{self.queue_id}.log", f"runs/{self.queue_id}", f"backup/{self.queue_id}"]:
            with self.subTest(relative=relative):
                target = self.root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("keep")
                result = self.launch()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Refusing to overwrite", result.stderr)
                self.assertEqual(target.read_text(), "keep")
                target.unlink()
                self.assertFalse((self.root / "launcher_calls.jsonl").exists())

    def test_more_than_24_hours_is_rejected_before_launch(self):
        result = self.launch(MAX_HOURS="25")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("MAX_HOURS", result.stderr)
        self.assertFalse((self.root / "launcher_args.json").exists())


if __name__ == "__main__":
    unittest.main()
