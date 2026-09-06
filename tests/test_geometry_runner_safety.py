"""CLI guards must reject invalid/subsampled formal runs before writing output."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class GeometryRunnerSafety(unittest.TestCase):
    def test_rejects_bad_args_before_creating_run(self):
        cases = [["--epochs", "0"], ["--lr", "nan"], ["--initial-offset", "inf"],
                 ["--kind", "formal", "--eval-limit", "32"],
                 ["--kind", "pilot", "--max-train-batches", "1"],
                 ["--score-chunk", "0"]]
        for extra in cases:
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "absent"
                run = subprocess.run([sys.executable, str(ROOT / "scripts/run_geometry.py"),
                                      "--root", directory, "--model", "sl8", "--run-dir", str(output),
                                      *extra], capture_output=True, text=True)
                self.assertNotEqual(run.returncode, 0)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
