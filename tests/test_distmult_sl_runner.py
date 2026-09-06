"""CPU-only protocol/routing tests for the untouched-author SL launcher."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import functools
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
CORE = next(path for path in (ROOT / "vendor/sl-manifold-core/src", ROOT.parent / "sl-manifold-core/src")
            if (path / "sl_manifold/core.py").is_file())
sys.path.insert(0, str(ROOT / "upstream/vl-kge"))
sys.path.insert(0, str(CORE))
from vlkge.helpers import negative_sampling_uniform

spec = importlib.util.spec_from_file_location("sl_author_launcher_test", ROOT / "scripts/run_distmult_sl_author.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def original_namespace(repo, out):
    # Build from the actual YAML/parser, not a local-only completed-run artifact.
    from vlkge.scripts import train as author_entry
    previous_argv = sys.argv[:]
    try:
        sys.argv = ["author-test-config", "--config", str(ROOT / "upstream/vl-kge" / launcher.AUTHOR_CONFIG)]
        record = vars(author_entry.parse_args()).copy()
    finally:
        sys.argv = previous_argv
    for key, relative in {
        "data_path": "vlkge/data/wn9_img/wn9_img_triples.csv",
        "visual_features_path": "vlkge/data/wn9_img/features/wn9_img_vf_clip.pkl",
        "textual_features_path": "vlkge/data/wn9_img/features/wn9_img_tf_clip.pkl",
    }.items():
        record[key] = str(repo / relative)
    record["save_path"] = str(out / "best.pt")
    return SimpleNamespace(**record)


class FakeOriginal(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.num_entities = 6
        self.num_relations = 1


class FakeExtension(FakeOriginal):
    diagnostic_value = 0.2

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.received_calibration = None

    def calibrate(self, triples):
        self.received_calibration = triples.clone()
        return {"calibrated": True, "frozen": True, "count": len(triples), "center": 0.1, "scale": 0.2}

    def diagnostics(self):
        return {"parameters_finite": True, "raw_gate": self.diagnostic_value,
                "relative_score_matrices": {"branch_risk_fraction": 0.0,
                                            "max_gregory_frobenius_remainder_bound": 1e-8}}


class FakeAuthorRuntime:
    """Preserve the author's wrapper nesting and reload-before-test ordering."""

    def __init__(self, args, *, make_checkpoint=True, startup_error=False):
        self.args = args
        self.calls = []
        self.writes = []
        self.best_reloaded = False
        self.make_checkpoint = make_checkpoint
        self.startup_error = startup_error
        self.created_model = None
        frame = pd.DataFrame([[0, 0, 1], [1, 0, 2], [2, 0, 3], [4, 0, 5]],
                             columns=["head_id", "relation_id", "tail_id"])
        self.train_loader = SimpleNamespace(dataset=SimpleNamespace(data=frame))
        self.val_loader = object()
        self.test_loader = object()
        self.sampled_loader = object()
        self.models = SimpleNamespace(DistMult=FakeOriginal)
        self.entry = SimpleNamespace(parse_args=lambda: original_namespace(args.repo, args.run_dir))
        self.helpers = SimpleNamespace(train=self.train, evaluate_kge=self.evaluate,
                                       negative_sampling_uniform=negative_sampling_uniform)
        self.runner = SimpleNamespace(ACTIVE_RUN_DIR=None, __file__=str(ROOT / "scripts/run_author.py"),
                                      save_json=self.save, main=self.main)
        self.torch = SimpleNamespace(load=torch.load,
                                     cuda=SimpleNamespace(max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0))

    def save(self, path, record):
        self.writes.append((path.name, json.loads(json.dumps(record, default=str, allow_nan=False))))
        path.write_text(json.dumps(record, default=str, allow_nan=False))

    def evaluate(self, model, dataloader, filter_map=None):
        split = "test" if dataloader is self.test_loader else "validation" if dataloader is self.val_loader else "sampled_train"
        self.calls.append(split)
        if split == "test":
            if not self.best_reloaded:
                raise AssertionError("author must reload best before final test")
            return 0.25, {1: 0.1, 3: 0.3, 10: 1.0}, {0: {"MRR": 0.25}}
        return 0.5, {1: 0.2, 3: 0.7, 10: 1.0}, {0: {"MRR": 0.5}}

    def train(self, model, train_loader, val_loader, test_loader, sampled_train_loader, save_path=None):
        self.helpers.evaluate_kge(model, sampled_train_loader)
        val = self.helpers.evaluate_kge(model, val_loader)
        if self.make_checkpoint:
            torch.save({"epoch": 1, "score": val[0], "model_state_dict": model.state_dict()}, save_path)
            model.load_state_dict(torch.load(save_path, weights_only=True)["model_state_dict"])
            self.best_reloaded = True
        self.helpers.evaluate_kge(model, test_loader)

    def main(self):
        out = self.args.run_dir
        out.mkdir(exist_ok=False)
        self.runner.ACTIVE_RUN_DIR = out
        if self.startup_error:
            raise RuntimeError("startup failure before original observer try block")
        state = {"status": "starting", "kind": "formal", "track": "author_release_protocol", "evaluations": 0}
        self.runner.save_json(out / "status.json", state)
        self.runner.save_json(out / "source_sha256.json", {"vlkge/helpers.py": "abc"})
        self.runner.save_json(out / "input_manifest.json", {"files": []})
        resolved = self.entry.parse_args()
        self.runner.save_json(out / "resolved_config.json", vars(resolved))
        self.resolved = resolved
        model = self.models.DistMult(num_entities=6, num_relations=1)
        self.created_model = model
        original_evaluate = self.helpers.evaluate_kge
        counts = {}

        @functools.wraps(original_evaluate)
        def observer(*args, **kwargs):
            loader = args[1]
            split = "test" if loader is self.test_loader else "validation" if loader is self.val_loader else "sampled_train"
            result = original_evaluate(*args, **kwargs)
            counts[split] = counts.get(split, 0) + 1
            event = {"split": split, "evaluation_index": counts[split], "mrr": result[0]}
            with (out / "evaluations.jsonl").open("a") as stream:
                stream.write(json.dumps(event) + "\n")
            state.update(status="running", last_evaluation=event, evaluations=state["evaluations"] + 1)
            self.runner.save_json(out / "status.json", state)
            if split == "test":
                state["final_test"] = event
            return result

        self.helpers.evaluate_kge = observer
        try:
            self.helpers.train(model, self.train_loader, self.val_loader, self.test_loader,
                               self.sampled_loader, save_path=out / "best.pt")
            state.update(status="completed", best_epoch=1, best_validation_mrr=0.5)
            self.runner.save_json(out / "result.json", state)
            self.runner.save_json(out / "status.json", state)
        except BaseException as error:
            state.update(status="failed", error=repr(error), traceback="fake author observer traceback")
            self.runner.save_json(out / "status.json", state)
            raise


class DistMultSLRunnerTests(unittest.TestCase):
    def make_args(self, directory, method="baseline", validation_only=True):
        return SimpleNamespace(repo=Path(directory).resolve() / "repo", run_dir=Path(directory).resolve() / "run",
                               config=launcher.AUTHOR_CONFIG, method=method,
                               extension_config=launcher.validate_extension_config(method, {}),
                               author_overrides={}, extension_config_file=None, author_overrides_file=None,
                               validation_only=validation_only, smoke_epochs=None, calibration_positives=4)

    def execute(self, runtime, extension_class=FakeExtension):
        # Calibration uses real CPU torch; completion uses harmless CUDA peak stubs.
        cuda = runtime.torch.cuda
        facade = SimpleNamespace(**{name: getattr(torch, name) for name in (
            "tensor", "long", "Generator", "randperm", "random", "no_grad", "stack", "cat", "load")}, cuda=cuda)
        originals = (runtime.models.DistMult, runtime.helpers.train, runtime.helpers.evaluate_kge,
                     runtime.entry.parse_args, runtime.runner.save_json, sys.argv[:], Path.cwd())
        with redirect_stdout(io.StringIO()):
            launcher.run_with_hooks(runtime.args, runtime.runner, runtime.models, runtime.helpers,
                                    runtime.entry, facade, extension_class=extension_class,
                                    source_hashes={"project/geometry/distmult_sl.py": "test-model-hash"})
        self.assertIs(runtime.models.DistMult, originals[0])
        self.assertEqual(runtime.helpers.train, originals[1])
        self.assertEqual(runtime.helpers.evaluate_kge, originals[2])
        self.assertIs(runtime.entry.parse_args, originals[3])
        self.assertEqual(runtime.runner.save_json, originals[4])
        self.assertEqual(sys.argv, originals[5])
        self.assertEqual(Path.cwd(), originals[6])
        self.assertIsNone(runtime.runner.ACTIVE_RUN_DIR)
        return json.loads((runtime.args.run_dir / "result.json").read_text())

    def test_validation_only_intercepts_after_reload_without_test_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = FakeAuthorRuntime(self.make_args(directory))
            result = self.execute(runtime)
            self.assertTrue(runtime.best_reloaded)
            self.assertEqual(runtime.calls, ["sampled_train", "validation"])
            self.assertEqual(result["status"], "completed_validation_only")
            self.assertFalse(result["test_evaluated"])
            self.assertNotIn("final_test", result)
            self.assertNotIn("error", result)
            self.assertFalse(result["final_test_interception"]["synthetic_metrics_used"])
            self.assertEqual(result["best_validation_mrr"], 0.5)
            self.assertFalse(any(record.get("status") == "failed" for _, record in runtime.writes))
            self.assertIs(type(runtime.created_model), FakeOriginal)
            self.assertIsNone(result["calibration"])

    def test_formal_really_evaluates_test_and_preserves_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = FakeAuthorRuntime(self.make_args(directory, validation_only=False))
            result = self.execute(runtime)
            self.assertEqual(runtime.calls, ["sampled_train", "validation", "test"])
            self.assertEqual(result["status"], "completed")
            self.assertTrue(result["test_evaluated"])
            self.assertEqual(result["final_test"]["mrr"], 0.25)

    def test_extension_calibrates_only_train_and_retains_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_args(directory, "sl8")
            runtime = FakeAuthorRuntime(args)
            rng = torch.get_rng_state().clone()
            result = self.execute(runtime)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            supplied = runtime.created_model.received_calibration
            train_set = set(map(tuple, runtime.train_loader.dataset.data.to_numpy().tolist()))
            self.assertEqual(len(supplied), 8)
            self.assertTrue(all(tuple(row) in train_set for row in supplied[:4].tolist()))
            self.assertTrue(all(tuple(row) not in train_set for row in supplied[4:].tolist()))
            self.assertFalse(result["calibration"]["validation_or_test_used"])
            self.assertEqual(runtime.created_model.kwargs["geometry"], "sl8")
            self.assertEqual(result["last_diagnostics"]["raw_gate"], 0.2)
            diagnostic_rows = [json.loads(row) for row in (args.run_dir / "diagnostics.jsonl").read_text().splitlines()]
            self.assertEqual(len(diagnostic_rows), 1)
            self.assertEqual(diagnostic_rows[0]["split"], "validation")
            sources = json.loads((args.run_dir / "source_sha256.json").read_text())
            self.assertEqual(sources["extension/project/geometry/distmult_sl.py"], "test-model-hash")
            self.assertEqual(json.loads((args.run_dir / "input_manifest.json").read_text()), {"files": []})

    def test_overrides_are_applied_before_observer_records_effective_args(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_args(directory)
            args.author_overrides = {"lr": 0.03, "num_neg_samples": 50, "epochs": 12, "seed": 2020}
            runtime = FakeAuthorRuntime(args)
            result = self.execute(runtime)
            resolved = json.loads((args.run_dir / "resolved_config.json").read_text())
            self.assertEqual(resolved["lr"], 0.03)
            self.assertEqual(resolved["model"], "DistMult")
            self.assertEqual(resolved["embedding_dim"], 768)
            self.assertEqual(result["author_args_before_overrides"]["lr"], 0.1)
            self.assertEqual(result["author_args_before_overrides"]["seed"], 42)
            self.assertEqual(result["author_overrides"], args.author_overrides)

    def test_failed_interception_is_not_reported_completed(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = FakeAuthorRuntime(self.make_args(directory), make_checkpoint=False)
            with self.assertRaisesRegex(RuntimeError, "best checkpoint"):
                self.execute(runtime)
            result = json.loads((runtime.args.run_dir / "status.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertFalse((runtime.args.run_dir / "result.json").exists())
            self.assertIs(runtime.models.DistMult, FakeOriginal)

    def test_startup_failure_is_recorded_and_existing_output_is_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = FakeAuthorRuntime(self.make_args(directory), startup_error=True)
            with self.assertRaisesRegex(RuntimeError, "startup failure"):
                self.execute(runtime)
            self.assertEqual(json.loads((runtime.args.run_dir / "status.json").read_text())["status"], "failed")
            before = (runtime.args.run_dir / "status.json").read_bytes()
            with self.assertRaises(FileExistsError):
                self.execute(runtime)
            self.assertEqual((runtime.args.run_dir / "status.json").read_bytes(), before)

    def test_nonfinite_diagnostics_leave_truthful_failure_record(self):
        before = FakeExtension.diagnostic_value
        try:
            FakeExtension.diagnostic_value = float("nan")
            with tempfile.TemporaryDirectory() as directory:
                runtime = FakeAuthorRuntime(self.make_args(directory, "sl8"))
                with self.assertRaises(FloatingPointError):
                    self.execute(runtime)
                record = json.loads((runtime.args.run_dir / "status.json").read_text())
                self.assertEqual(record["status"], "failed")
                self.assertEqual(record["last_diagnostics"]["raw_gate"], "nan")
                self.assertFalse(record["test_evaluated"])
        finally:
            FakeExtension.diagnostic_value = before

    def test_invalid_hpo_arguments_rejected_before_output_creation(self):
        invalid = [{"embedding_dim": 63}, {"model": "ComplEx"}, {"dataset": "other"},
                   {"num_negatives": 10}, {"top_k": [1]}, {"evaluate_every": 5},
                   {"lr": float("nan")}, {"lr": True}, {"batch_size": 0}, {"seed": -1}]
        for override in invalid:
            with self.subTest(override=override), tempfile.TemporaryDirectory() as directory:
                out = Path(directory) / "run"
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    launcher.parse_args(["--repo", directory, "--run-dir", str(out), "--method", "baseline",
                                         "--author-overrides", json.dumps(override)])
                self.assertFalse(out.exists())
        with self.assertRaises(ValueError):
            launcher.validate_extension_config("baseline", {"residual_weight": 1})
        with self.assertRaises(ValueError):
            launcher.validate_extension_config("sl8", {"geometry": "euclidean"})

    def test_sources_include_actual_model_launcher_observer_and_shared_core(self):
        core = CORE
        hashes = launcher.extension_source_hashes(ROOT, core)
        for label, path in {
            "project/geometry/distmult_sl.py": ROOT / "geometry/distmult_sl.py",
            "project/scripts/run_author.py": ROOT / "scripts/run_author.py",
            "project/scripts/run_distmult_sl_author.py": ROOT / "scripts/run_distmult_sl_author.py",
            "shared_sl/sl_manifold/core.py": core / "sl_manifold/core.py",
        }.items():
            self.assertEqual(hashes[label], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_health_checks_reject_branch_or_large_tail_but_not_small_calibration_std(self):
        launcher._validate_health({"scale": 1e-6})
        for value in ({"parameters_finite": False}, {"branch_risk_fraction": 0.01},
                      {"max_gregory_frobenius_remainder_bound": 0.01}):
            with self.assertRaises(FloatingPointError):
                launcher._validate_health(value)

    def test_rejection_capacity_guard_prevents_unbounded_author_loop(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = FakeAuthorRuntime(self.make_args(directory))
            filters = {(0, 0): {0, 1, 2, 3, 4}}
            with self.assertRaisesRegex(ValueError, "would hang"):
                launcher.validate_negative_capacity(runtime.train_loader, filters, 6, 2)
            result = launcher.validate_negative_capacity(runtime.train_loader, filters, 6, 1)
            self.assertEqual(result["minimum_available_for_training_query"], 1)

    def test_real_author_train_loss_evaluator_and_sl_checkpoint_interception_cpu(self):
        from collections import defaultdict
        from torch.utils.data import DataLoader
        from vlkge import helpers as real_helpers
        from vlkge.dataloader import KGDataset
        from vlkge.models.distmult import DistMult
        sys.path.insert(0, str(ROOT))
        sys.path.insert(0, str(CORE))
        from geometry.distmult_sl import DistMultSL
        original_train, original_eval = real_helpers.train, real_helpers.evaluate_kge
        original_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(2)
            with tempfile.TemporaryDirectory() as directory:
                args = self.make_args(directory, "sl8")
                args.extension_config["score_chunk"] = 32
                args.author_overrides = {"epochs": 1, "num_neg_samples": 2, "batch_size": 4}
                runtime = FakeAuthorRuntime(args)
                mapping = {f"e{i}": i for i in range(6)}
                relation_mapping = {"r0": 0, "r1": 1}
                splits = [
                    [[0, 0, 1], [1, 0, 2], [2, 1, 3], [4, 1, 5]],
                    [[3, 0, 4], [5, 1, 0]], [[0, 1, 5], [4, 0, 2]],
                ]
                frames = [pd.DataFrame([[f"e{h}", f"r{r}", f"e{t}"] for h, r, t in rows],
                                       columns=["head", "relation", "tail"]) for rows in splits]
                generator = torch.Generator().manual_seed(42)
                datasets = [KGDataset(frame, mapping, relation_mapping) for frame in frames]
                runtime.train_loader = DataLoader(datasets[0], batch_size=4, shuffle=True, generator=generator)
                runtime.val_loader = DataLoader(datasets[1], batch_size=2, shuffle=False)
                runtime.test_loader = DataLoader(datasets[2], batch_size=2, shuffle=False)
                runtime.sampled_loader = DataLoader(datasets[0], batch_size=4, shuffle=False)
                filters = defaultdict(set)
                for rows in splits:
                    for h, r, t in rows:
                        filters[(h, r)].add(t)
                        filters[(t, r)].add(h)
                features = torch.randn(6, 768, generator=torch.Generator().manual_seed(999))
                model_kwargs = dict(num_entities=6, num_relations=2, embedding_dim=768,
                                    use_structural=True, use_visual=True, use_textual=True,
                                    visual_features=features, textual_features=features.flip(0),
                                    visual_entity_to_index={i: i for i in range(6)},
                                    textual_entity_to_index={i: i for i in range(6)},
                                    fusion_mode="average", device=torch.device("cpu"))
                runtime.models = SimpleNamespace(DistMult=DistMult)
                runtime.helpers = real_helpers

                @functools.wraps(original_eval)
                def counted_evaluate(*ev_args, **ev_kwargs):
                    loader = ev_args[1] if len(ev_args) > 1 else ev_kwargs["dataloader"]
                    runtime.calls.append("test" if loader is runtime.test_loader else
                                         "validation" if loader is runtime.val_loader else "sampled_train")
                    return original_eval(*ev_args, **ev_kwargs)

                real_helpers.evaluate_kge = counted_evaluate

                def real_main():
                    out = args.run_dir
                    out.mkdir(exist_ok=False)
                    runtime.runner.ACTIVE_RUN_DIR = out
                    state = {"status": "starting", "kind": "formal", "evaluations": 0}
                    runtime.runner.save_json(out / "status.json", state)
                    resolved = runtime.entry.parse_args()
                    model = runtime.models.DistMult(**model_kwargs)
                    runtime.created_model = model
                    original_load = model.load_state_dict

                    def observed_load(*load_args, **load_kwargs):
                        result = original_load(*load_args, **load_kwargs)
                        runtime.best_reloaded = True
                        return result

                    model.load_state_dict = observed_load
                    inner_evaluate = real_helpers.evaluate_kge
                    counts = {}

                    @functools.wraps(inner_evaluate)
                    def observer(*ev_args, **ev_kwargs):
                        loader = ev_args[1] if len(ev_args) > 1 else ev_kwargs["dataloader"]
                        split = "test" if loader is runtime.test_loader else "validation" if loader is runtime.val_loader else "sampled_train"
                        result = inner_evaluate(*ev_args, **ev_kwargs)
                        counts[split] = counts.get(split, 0) + 1
                        event = {"split": split, "evaluation_index": counts[split], "mrr": result[0]}
                        with (out / "evaluations.jsonl").open("a") as stream:
                            stream.write(json.dumps(event) + "\n")
                        state.update(status="running", last_evaluation=event, evaluations=state["evaluations"] + 1)
                        runtime.runner.save_json(out / "status.json", state)
                        return result

                    real_helpers.evaluate_kge = observer
                    try:
                        real_helpers.train(
                            model, runtime.train_loader, runtime.val_loader, runtime.test_loader,
                            runtime.sampled_loader, torch.optim.Adagrad(model.parameters(), lr=resolved.lr),
                            None, filters, {}, num_epochs=resolved.epochs,
                            num_neg_samples=resolved.num_neg_samples, device=torch.device("cpu"),
                            save_path=str(out / "best.pt"), generator=generator,
                            patience=50, evaluate_every=1, bidirectional_eval=True,
                        )
                        raise AssertionError("Expected validation-only sentinel before return")
                    except BaseException as error:
                        state.update(status="failed", error=repr(error), traceback="test observer caught author exit")
                        runtime.runner.save_json(out / "status.json", state)
                        raise

                runtime.runner.main = real_main
                with redirect_stderr(io.StringIO()):
                    result = self.execute(runtime, DistMultSL)
                self.assertTrue(runtime.best_reloaded)
                self.assertEqual(runtime.calls, ["sampled_train", "validation"])
                self.assertEqual(result["status"], "completed_validation_only")
                self.assertEqual(result["best_epoch"], 1)
                self.assertEqual(result["negative_sampling_capacity"]["requested_unique_negatives"], 2)
                self.assertEqual(result["calibration"]["training_positive_count"], 4)
                self.assertEqual(result["last_diagnostics"]["calibration_count"], 8)
                self.assertIn("compatibility_on_train_probe", result["last_diagnostics"])
        finally:
            real_helpers.train, real_helpers.evaluate_kge = original_train, original_eval
            torch.set_num_threads(original_threads)


if __name__ == "__main__":
    unittest.main()
