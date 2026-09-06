"""Lightweight tests requiring no GPU or author dataset."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RunnerSafety(unittest.TestCase):
    def test_nonpositive_smoke_rejected_before_run_creation(self):
        for epochs in ["0", "-1"]:
            with tempfile.TemporaryDirectory() as directory:
                out = Path(directory) / "must-not-exist"
                run = subprocess.run([sys.executable, str(ROOT / "scripts/run_author.py"),
                                      "--repo", directory, "--config", "absent.yaml",
                                      "--run-dir", str(out), "--smoke-epochs", epochs],
                                     capture_output=True, text=True)
                self.assertNotEqual(run.returncode, 0)
                self.assertIn("must be a positive integer", run.stderr)
                self.assertFalse(out.exists())

    def test_queue_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run = subprocess.run([sys.executable, str(ROOT / "scripts/run_queue.py"),
                                  "--root", directory, "--queue-id", "../bad"],
                                 capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("simple directory name", run.stderr)


if __name__ == "__main__":
    unittest.main()
