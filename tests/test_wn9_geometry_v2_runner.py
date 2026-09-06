"""CPU-only protocol/sampler tests; no dataset download or GPU training."""

import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("wn9_v2_runner", ROOT / "scripts/run_wn9_geometry_v2.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class TableModel(torch.nn.Module):
    def __init__(self, table):
        super().__init__()
        self.register_buffer("table", table)
        self.num_entities, self.num_relations, _ = table.shape
        self.calls = []

    def forward(self, h, r, t):
        self.calls.append((h.clone(), r.clone(), t.clone()))
        return self.table[h, r, t]


def reference_ranks(model, triples, filters):
    result = {"head": [], "tail": []}
    ids = torch.arange(model.num_entities)
    for h, r, t in triples.tolist():
        for direction in ("tail", "head"):
            heads = ids if direction == "head" else torch.full_like(ids, h)
            tails = ids if direction == "tail" else torch.full_like(ids, t)
            values = model(heads, torch.full_like(ids, r), tails)
            fixed, target = (t, h) if direction == "head" else (h, t)
            eligible = [i for i in range(len(ids))
                        if i != target and i not in filters[direction].get((fixed, r), set())]
            better = sum(float(values[i]) > float(values[target]) for i in eligible)
            ties = sum(float(values[i]) == float(values[target]) for i in eligible)
            result[direction].append(1.0 + better + 0.5 * ties)
    return result


class StrictDataTests(unittest.TestCase):
    def test_directional_filters_do_not_merge_roles(self):
        triples = torch.tensor([[0, 0, 1], [2, 0, 0], [0, 1, 3]])
        filters = runner.directional_filters(triples)
        self.assertEqual(filters["tail"][(0, 0)], {1})
        self.assertEqual(filters["head"][(0, 0)], {2})
        self.assertEqual(filters["tail"][(0, 1)], {3})

    def test_train_only_negative_filter_and_distinct_sampling(self):
        train = torch.tensor([[0, 0, 1]])
        held_out = (0, 0, 2)
        sampler = runner.TrainNegativeSampler(5, runner.directional_filters(train), 42)
        positives = train.repeat(64, 1)
        negatives = sampler.sample(positives, 4).reshape(64, 4, 3)
        seen_held_out = False
        for row in negatives:
            self.assertEqual(len({tuple(t) for t in row.tolist()}), 4)
            self.assertNotIn(tuple(train[0].tolist()), [tuple(t) for t in row.tolist()])
            # Both roles may be corrupted, but never both in the same triple.
            self.assertTrue(all(h == 0 or t == 1 for h, _, t in row.tolist()))
            seen_held_out |= held_out in [tuple(t) for t in row.tolist()]
        self.assertTrue(seen_held_out, "Validation facts must not silently filter training negatives")

    def test_sampler_reproducibility_and_finite_capacity_guard(self):
        train = torch.tensor([[0, 0, 1], [1, 0, 2], [2, 0, 3]])
        filters = runner.directional_filters(train)
        left = runner.TrainNegativeSampler(5, filters, 7)
        right = runner.TrainNegativeSampler(5, filters, 7)
        for _ in range(3):
            self.assertTrue(torch.equal(left.sample(train, 3), right.sample(train, 3)))
        with self.assertRaisesRegex(ValueError, "admissible"):
            left.sample(train, 5)

    def test_complement_mapping_exhausts_exact_valid_ids(self):
        # Consecutive forbidden IDs exercise duplicate adjusted search keys.
        train = torch.tensor([[0, 0, 1], [0, 0, 2], [0, 0, 4],
                              [1, 0, 0], [2, 0, 0], [4, 0, 0]])
        filters = runner.directional_filters(train)
        sampler = runner.TrainNegativeSampler(6, filters, 8)
        positive = torch.tensor([[0, 0, 0]])
        for _ in range(10):
            sample = sampler.sample(positive, 3)
            column = 0 if not torch.equal(sample[:, 0], torch.zeros(3, dtype=torch.long)) else 2
            self.assertEqual(set(sample[:, column].tolist()), {0, 3, 5})

    def test_fixed_subset_is_sorted_seeded_and_hashes_exact_indices(self):
        triples = torch.arange(90).reshape(30, 3)
        selected, metadata = runner.select_validation(triples, 7, 260906)
        selected_again, again = runner.select_validation(triples, 7, 260906)
        self.assertTrue(torch.equal(selected, selected_again))
        self.assertEqual(metadata, again)
        self.assertEqual(metadata["validation_indices"], sorted(metadata["validation_indices"]))
        self.assertEqual(metadata["validation_scope"], "subset")
        self.assertEqual(metadata["validation_count"], 7)
        all_selected, full = runner.select_validation(triples)
        self.assertTrue(torch.equal(all_selected, triples))
        self.assertEqual(full["validation_scope"], "full")
        self.assertNotEqual(full["subset_indices_hash"], metadata["subset_indices_hash"])


class StrictEvaluationTests(unittest.TestCase):
    def setUp(self):
        ids = torch.arange(6)
        table = ((2 * ids[:, None, None] + ids[None, None, :] + torch.arange(2)[None, :, None]) % 4).float()
        self.model = TableModel(table)
        self.triples = torch.tensor([[0, 0, 4], [5, 1, 0], [2, 0, 3], [1, 1, 5]])
        self.filters = runner.directional_filters(torch.cat((self.triples, torch.tensor([[0, 0, 2], [1, 0, 4]]))))

    def test_matches_literal_reference_for_all_chunks(self):
        expected = reference_ranks(self.model, self.triples, self.filters)
        for query_batch in (1, 3, 32):
            for candidate_chunk in (1, 4, 256):
                with self.subTest(query_batch=query_batch, candidate_chunk=candidate_chunk):
                    result = runner.evaluate_strict(self.model, self.triples, self.filters, "cpu", query_batch, candidate_chunk)
                    self.assertEqual(result["ranks_head"], expected["head"])
                    self.assertEqual(result["ranks_tail"], expected["tail"])
                    self.assertEqual(result["tie_policy"], "realistic_average")
                    self.assertEqual(result["directional_queries"], 8)

    def test_exact_ties_average_and_target_self_excluded_even_without_mask(self):
        model = TableModel(torch.zeros(5, 1, 5))
        triples = torch.tensor([[4, 0, 2]])
        no_filters = {"head": {}, "tail": {}}
        result = runner.evaluate_strict(model, triples, no_filters, "cpu")
        self.assertEqual(result["ranks_head"], [3.0])
        self.assertEqual(result["ranks_tail"], [3.0])
        self.assertAlmostEqual(result["mrr"], 1 / 3)
        filters = {"head": {(2, 0): {0}}, "tail": {(4, 0): {0, 1}}}
        result = runner.evaluate_strict(model, triples, filters, "cpu")
        self.assertEqual(result["ranks_head"], [2.5])
        self.assertEqual(result["ranks_tail"], [2.0])

    def test_nonfinite_rejected_before_filtering_and_mode_rng_restored(self):
        self.model.train()
        before = torch.get_rng_state().clone()
        runner.evaluate_strict(self.model, self.triples, self.filters, "cpu")
        self.assertTrue(self.model.training)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.model.table[0, 0, 2] = float("nan")
        with self.assertRaises(FloatingPointError):
            runner.evaluate_strict(self.model, self.triples, self.filters, "cpu")
        self.assertTrue(self.model.training)

    def test_target_only_scored_in_candidate_pass(self):
        self.model.calls.clear()
        runner.evaluate_strict(self.model, self.triples[:1], self.filters, "cpu", 1, 4)
        self.assertEqual([len(call[0]) for call in self.model.calls], [4, 2, 4, 2])

    def test_preflight_has_distance_and_gradient_stats_and_preserves_mode(self):
        model = TableModel(-torch.ones(5, 1, 5) * 2)
        model.train()
        triples = torch.tensor([[0, 0, 1], [1, 0, 2]])
        result = runner.initial_score_statistics(model, triples, triples, "cpu", alpha=1, offset=0)
        self.assertTrue(model.training)
        self.assertEqual(result["positive"]["score"]["median"], -2)
        self.assertEqual(result["negative"]["distance"]["median"], 2)
        self.assertAlmostEqual(result["negative"]["sigmoid_probability_mean"], 1 / (1 + math.exp(2)), places=6)


class RunnerGuardTests(unittest.TestCase):
    def test_invalid_options_fail_before_output_creation(self):
        cases = (["--epochs", "0"], ["--lr", "nan"], ["--kind", "formal"],
                 ["--eval-limit", "32"], ["--max-train-batches", "3"],
                 ["--initial-offset", "inf"], ["--target-init-norm", "2"],
                 ["--model", "murp"], ["--log-order", "0"])
        for extra in cases:
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "absent"
                completed = subprocess.run([sys.executable, str(ROOT / "scripts/run_wn9_geometry_v2.py"),
                                            "--root", directory, "--run-dir", str(output),
                                            "--model", "sl8", *extra], capture_output=True, text=True)
                self.assertNotEqual(completed.returncode, 0)
                self.assertFalse(output.exists())

    def test_smoke_limits_and_locked_defaults(self):
        args = runner.parse_args(["--root", ".", "--run-dir", "not-created", "--model", "sl8",
                                  "--kind", "smoke", "--max-train-batches", "3", "--eval-limit", "32"])
        self.assertEqual(args.initial_logit_scale, 1)
        self.assertEqual(args.target_init_norm, 0.5)
        self.assertEqual(args.coordinate_scale, 1)
        self.assertEqual(args.negatives, 100)
        self.assertEqual(args.log_backend, "gregory12")
        self.assertTrue(args.validation_only)

    def test_explicit_gl_backend_remains_available(self):
        args = runner.parse_args(["--root", ".", "--run-dir", "not-created", "--model", "sl8",
                                  "--log-backend", "gauss_legendre", "--log-order", "32"])
        self.assertEqual(args.log_backend, "gauss_legendre")
        self.assertEqual(args.log_order, 32)

    def test_model_backend_diagnostic_must_match_requested_track(self):
        for backend, order, terms in (("gregory12", 16, 12), ("gauss_legendre", 32, 32)):
            diagnostic = {"geometry": "sl8", "log_backend": backend,
                          "model_contract": {"log_backend": backend, "log_terms": terms},
                          "principal_log": {"backend": backend}}
            runner.check_diagnostic_backend(diagnostic, backend, order, "sl8")
            for replacement in ({"log_backend": "unknown"},
                                {"model_contract": {"log_backend": backend, "log_terms": 7}},
                                {"principal_log": {"backend": "unknown"}}, {"geometry": "euclidean"}):
                with self.subTest(backend=backend, replacement=replacement), self.assertRaises(RuntimeError):
                    runner.check_diagnostic_backend({**diagnostic, **replacement}, backend, order, "sl8")

    def test_v2_health_does_not_reinstate_old_cayley_series_threshold(self):
        runner.check_health({"cayley_norm": 2, "max_gregory_frobenius_remainder_bound": 100})
        with self.assertRaises(FloatingPointError):
            runner.check_health({"sampled_health_passed": False})
        with self.assertRaises(FloatingPointError):
            runner.check_health({"nested": [float("nan")]})

    def test_cpu_mini_update_uses_same_sampler_loss_and_evaluator_interfaces(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.entities = torch.nn.Embedding(6, 3)
                self.relations = torch.nn.Embedding(1, 3)
                self.num_entities, self.num_relations = 6, 1

            def forward(self, h, r, t):
                return -torch.linalg.vector_norm(self.entities(h) + self.relations(r) - self.entities(t), dim=-1)

        torch.manual_seed(42)
        model = TinyModel()
        train = torch.tensor([[0, 0, 1], [1, 0, 2], [2, 0, 3]])
        filters = runner.directional_filters(train)
        sampler = runner.TrainNegativeSampler(6, filters, 42)
        optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
        initial = model.entities.weight.detach().clone()
        for _ in range(2):
            negatives = sampler.sample(train, 2)
            positive_scores = model(*train.T)
            negative_scores = model(*negatives.T).reshape(len(train), 2)
            # Literal author loss: equal weight per score, not per class.
            loss = torch.cat((torch.nn.functional.softplus(-positive_scores[:, None]),
                              torch.nn.functional.softplus(negative_scores)), dim=1).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
        self.assertFalse(torch.equal(initial, model.entities.weight))
        result = runner.evaluate_strict(model, train, filters, "cpu", 2, 3)
        self.assertTrue(math.isfinite(result["mrr"]))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
