"""Tiny CPU training and immutable data/protocol checks; no network/GPU."""

from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("structural_runner", ROOT / "scripts/run_structural_geometry.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
runner.setup_imports(ROOT)
from structural_kge import data
from structural_kge.models import StructuralKGE


class StructuralRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def args(self, directory, model="euclidean", extras=()):
        return runner.parse_args(["--root", str(ROOT), "--data-root", str(directory),
                                  "--run-dir", str(directory), "--dataset", "WN18RR",
                                  "--model", model, "--epochs", "3", "--eval-every", "2",
                                  "--batch-size", "3", "--negatives", "2", "--score-chunk", "8",
                                  "--query-batch", "2", "--candidate-chunk", "4", *extras])

    def state(self, args):
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        return {"config": config, "log_backend": args.log_backend,
                "log_terms": 12 if args.log_backend == "gregory12" else args.log_order,
                "test_evaluated": False}

    def triples(self):
        return (torch.tensor([[0, 0, 1], [1, 1, 2], [2, 0, 3], [3, 1, 4]]),
                torch.tensor([[0, 1, 3], [2, 1, 4]]), torch.tensor([[8, 1, 7]]))

    def test_validation_cadence_always_includes_last_epoch(self):
        self.assertEqual([e for e in range(1, 16) if runner.should_evaluate(e, 15, 5)], [5, 10, 15])
        self.assertEqual([e for e in range(1, 13) if runner.should_evaluate(e, 12, 5)], [5, 10, 12])
        self.assertTrue(runner.should_evaluate(1, 1, 5))

    def test_loss_and_gradients_are_bit_exact_to_author(self):
        from vlkge.helpers import compute_logistic_loss
        torch.manual_seed(7)
        positives = torch.randn(4, requires_grad=True)
        negatives = torch.randn(4, 3, requires_grad=True)
        original = compute_logistic_loss(positives, negatives)
        copied = runner.logistic_loss(positives, negatives)
        self.assertTrue(torch.equal(original, copied))
        original_grad = torch.autograd.grad(original, (positives, negatives))
        copied_grad = torch.autograd.grad(copied, (positives, negatives))
        self.assertTrue(all(torch.equal(left, right) for left, right in zip(original_grad, copied_grad)))

    def test_all_geometries_train_and_checkpoint_with_identical_rng_and_cadence(self):
        negative_hashes = []
        for geometry in ("euclidean", "hyperbolic", "sl8"):
            with self.subTest(geometry=geometry), tempfile.TemporaryDirectory() as directory:
                torch.manual_seed(42)
                args = self.args(directory, geometry)
                model = StructuralKGE(9, 2, geometry=geometry, coordinate_radius=1.5,
                                      init_scale=0.5, score_chunk=8, log_backend="gregory12")
                train, valid, test = self.triples()
                filters = runner.directional_filters(torch.cat((train, valid, test)))
                evaluated = []
                original_evaluate = runner.evaluate_strict

                def tracked_evaluate(model, triples, *positional, **kwargs):
                    evaluated.append(triples.clone())
                    return original_evaluate(model, triples, *positional, **kwargs)

                with redirect_stdout(io.StringIO()), patch.object(runner, "evaluate_strict", tracked_evaluate):
                    state = runner.training_loop(model, train, valid, filters, args, self.state(args),
                                                 Path(directory), "cpu", cpu_test=True)
                self.assertEqual(state["status"], "completed_validation_only")
                self.assertFalse(state["test_evaluated"])
                self.assertNotIn("final_test", state)
                self.assertEqual(state["validation_epochs"], [2, 3])
                self.assertEqual(state["total_optimizer_steps"], 6)
                self.assertTrue(state["training_complete"])
                self.assertEqual(state["validation_scope"], "full")
                self.assertTrue(all(torch.equal(t, valid) for t in evaluated))
                self.assertTrue(state["best_checkpoint_diagnostics"]["sampled_health_passed"])
                events = [json.loads(line) for line in (Path(directory) / "epochs.jsonl").read_text().splitlines()]
                self.assertIsNone(events[0]["validation"])
                self.assertIsNotNone(events[1]["validation"])
                self.assertEqual(sum(event["train_batches"] for event in events), 6)
                negative_hashes.append([event["first_batch_negative_sha256"] for event in events])
                best = torch.load(Path(directory) / "best.pt", map_location="cpu", weights_only=False)
                last = torch.load(Path(directory) / "last.pt", map_location="cpu", weights_only=False)
                self.assertEqual(best["epoch"], state["best_epoch"])
                self.assertEqual(last["epoch"], 3)
                self.assertEqual(last["log_terms"], 12)
                self.assertIn("negative_rng_state", last)
                self.assertIn("optimizer_state_dict", last)
        self.assertTrue(all(hashes == negative_hashes[0] for hashes in negative_hashes))

    def test_smoke_is_explicitly_partial_and_final_epoch_is_evaluated(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            args = self.args(directory, extras=("--kind", "smoke", "--epochs", "1",
                                               "--eval-every", "5", "--max-train-batches", "1", "--eval-limit", "1"))
            model = StructuralKGE(9, 2, coordinate_radius=1.5)
            train, valid, test = self.triples()
            state = runner.training_loop(model, train, valid, runner.directional_filters(torch.cat((train, valid, test))),
                                         args, self.state(args), Path(directory), "cpu", cpu_test=True)
            self.assertFalse(state["training_complete"])
            self.assertEqual(state["validation_scope"], "subset")
            self.assertEqual(state["total_optimizer_steps"], 1)
            self.assertEqual(state["validation_epochs"], [1])

    def test_last_checkpoint_exists_before_first_validation(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            args = self.args(directory)
            model = StructuralKGE(9, 2, coordinate_radius=1.5)
            train, valid, test = self.triples()
            with patch.object(runner, "evaluate_strict", side_effect=KeyboardInterrupt("unit-test stop")):
                with self.assertRaises(KeyboardInterrupt):
                    runner.training_loop(model, train, valid, runner.directional_filters(torch.cat((train, valid, test))),
                                         args, self.state(args), Path(directory), "cpu", cpu_test=True)
            last = torch.load(Path(directory) / "last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(last["epoch"], 1)
            self.assertIsNone(last["validation"])
            self.assertFalse((Path(directory) / "best.pt").exists())

    def test_production_loop_refuses_cpu_without_explicit_test_seam(self):
        with self.assertRaisesRegex(RuntimeError, "unit-test"):
            runner.training_loop(None, None, None, None, None, {}, None, "cpu")

    def test_invalid_cli_fails_before_creating_output(self):
        for extras in (["--eval-every", "0"], ["--lr", "nan"], ["--kind", "formal"],
                       ["--eval-limit", "32"], ["--max-train-batches", "3"],
                       ["--dataset", "WN9-IMG"], ["--coordinate-scale", "1"], ["--log-backend", "gregory16"]):
            with self.subTest(extras=extras), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "absent"
                process = subprocess.run([sys.executable, str(ROOT / "scripts/run_structural_geometry.py"),
                                          "--root", str(ROOT), "--data-root", directory, "--dataset", "WN18RR",
                                          "--run-dir", str(output), "--model", "sl8", *extras], capture_output=True, text=True)
                self.assertNotEqual(process.returncode, 0)
                self.assertFalse(output.exists())


class StructuralDataTests(unittest.TestCase):
    def test_released_dictionary_ids_are_not_reordered_or_train_limited(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "WN18RR"
            folder.mkdir()
            (folder / "entities.dict").write_text("2\ttrain_tail\n0\theld_out_only\n1\ttrain_head\n")
            (folder / "relations.dict").write_text("0\tr\n")
            (folder / "train.txt").write_text("train_head\tr\ttrain_tail\n")
            (folder / "valid.txt").write_text("held_out_only\tr\ttrain_tail\n")
            (folder / "test.txt").write_text("train_tail\tr\theld_out_only\n")
            loaded = data.load_dataset(directory, "WN18RR", verify=False)
            self.assertEqual(loaded.num_entities, 3)
            self.assertEqual(loaded.train.tolist(), [[1, 0, 2]])
            self.assertEqual(loaded.valid.tolist(), [[0, 0, 2]])
            self.assertNotIn((0, 0), loaded.training_filter().tails)
            self.assertIn((0, 0), loaded.ranking_filter().tails)

    def test_pinned_blob_check_detects_tampering_even_with_rewritten_sha256(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            original = b"0\tentity\n"
            (folder / "entities.dict").write_bytes(original)
            source = {"blobs": {"entities.dict": data.git_blob_sha1(original)}}
            manifest = {"dataset": "WN18RR", "commit": data.SOURCE_COMMIT,
                        "files": {"entities.dict": {"sha256": hashlib.sha256(original).hexdigest()}}}
            (folder / "provenance.json").write_text(json.dumps(manifest))
            with patch.dict(data.SOURCES, {"WN18RR": source}):
                data.verify_dataset(folder, "WN18RR")
                changed = b"0\ttampered\n"
                (folder / "entities.dict").write_bytes(changed)
                manifest["files"]["entities.dict"]["sha256"] = hashlib.sha256(changed).hexdigest()
                (folder / "provenance.json").write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "Pinned source mismatch"):
                    data.verify_dataset(folder, "WN18RR")


if __name__ == "__main__":
    unittest.main()
