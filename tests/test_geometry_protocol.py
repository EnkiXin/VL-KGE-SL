"""Synthetic author-protocol parity and chunking safety checks (CPU only)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geometry.evaluation import evaluate_full


class TableModel(torch.nn.Module):
    def __init__(self, table):
        super().__init__()
        self.register_buffer("table", table)
        self.num_entities, self.num_relations, _ = table.shape
        self.calls = []

    def forward(self, head, relation, tail):
        self.calls.append((head.clone(), relation.clone(), tail.clone()))
        return self.table[head, relation, tail]


def author_reference(model, triples, filter_map):
    """Literal author candidate ordering, masking, and stable sort ranking."""
    ranks = {"head": [], "tail": []}
    candidates = torch.arange(model.num_entities)
    for head, relation, tail in triples.tolist():
        for direction in ("tail", "head"):
            heads = candidates if direction == "head" else torch.full_like(candidates, head)
            tails = candidates if direction == "tail" else torch.full_like(candidates, tail)
            scores = model(heads, torch.full_like(candidates, relation), tails).clone()
            fixed, target = (head, tail) if direction == "tail" else (tail, head)
            for entity in filter_map.get((fixed, relation), set()):
                if entity != target:
                    scores[entity] = float("-inf")
            order = torch.argsort(scores, descending=True, stable=True)
            ranks[direction].append(int((order == target).nonzero()[0, 0]) + 1)
    return ranks


class GeometryProtocolTests(unittest.TestCase):
    def setUp(self):
        # Deliberately asymmetric head/tail scores and many exact ties.
        ids = torch.arange(6)
        table = ((ids[:, None, None] * 2 + ids[None, None, :] + torch.arange(2)[None, :, None]) % 4).float()
        self.model = TableModel(table)
        self.triples = torch.tensor([[0, 0, 4], [5, 1, 0], [2, 0, 3], [1, 1, 5], [4, 0, 1]])
        self.filter_map = {
            (0, 0): {1, 2, 4}, (4, 0): {0, 1, 3},
            (5, 1): {0, 1, 2}, (0, 1): {2, 5},
            (2, 0): {0, 3, 5}, (3, 0): {1, 2},
            (1, 1): {0, 5}, (1, 0): {2, 4},
        }

    def test_chunked_matches_author_stable_sort_both_directions(self):
        reference = author_reference(self.model, self.triples, self.filter_map)
        self.assertNotEqual(reference["head"], reference["tail"])
        for query_batch in (1, 2, 16):
            for candidate_chunk in (1, 2, 4, 100):
                with self.subTest(query_batch=query_batch, candidate_chunk=candidate_chunk):
                    result = evaluate_full(self.model, self.triples, self.filter_map, "cpu", query_batch, candidate_chunk)
                    self.assertEqual(result["ranks_head"], reference["head"])
                    self.assertEqual(result["ranks_tail"], reference["tail"])
                    self.assertEqual(result["mrr"], (sum(1/r for r in reference["head"]) + sum(1/r for r in reference["tail"])) / 10)
                    self.assertEqual(result["directional_queries"], 10)
                    self.assertTrue(result["complete"])
                    json.dumps(result, allow_nan=False)

    def test_all_ties_keep_target_and_use_ascending_id(self):
        model = TableModel(torch.zeros(6, 1, 6))
        result = evaluate_full(model, torch.tensor([[4, 0, 3]]), {(4, 0): {0, 2, 3}, (3, 0): {1, 4}}, "cpu")
        self.assertEqual(result["ranks_tail"], [2])  # Only entity 1 remains before target 3.
        self.assertEqual(result["ranks_head"], [4])  # Entities 0,2,3 precede target 4.

    def test_nonfinite_is_rejected_even_if_candidate_would_be_filtered(self):
        self.model.table[0, 0, 2] = float("nan")
        self.model.train()
        with self.assertRaises(FloatingPointError):
            evaluate_full(self.model, self.triples[:1], self.filter_map, "cpu")
        self.assertTrue(self.model.training)

    def test_target_is_scored_only_in_candidate_pass(self):
        self.model.calls.clear()
        evaluate_full(self.model, self.triples[:1], self.filter_map, "cpu", 1, 4)
        self.assertEqual(len(self.model.calls), 4)  # 2 chunks × 2 directions.
        self.assertEqual([len(call[0]) for call in self.model.calls], [4, 2, 4, 2])
        head_calls = self.model.calls[2:]
        self.assertEqual(torch.cat([call[0] for call in head_calls]).tolist(), list(range(6)))
        self.assertTrue(all((call[2] == 4).all() for call in head_calls))

    def test_mode_rng_and_capped_metadata(self):
        self.model.train()
        rng = torch.get_rng_state().clone()
        result = evaluate_full(self.model, self.triples, self.filter_map, "cpu", max_queries=2)
        self.assertTrue(self.model.training)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertFalse(result["complete"])
        self.assertEqual(result["evaluated_triples"], 2)
        self.assertEqual(result["total_triples"], 5)
        self.model.eval()
        evaluate_full(self.model, self.triples[:1], self.filter_map, "cpu")
        self.assertFalse(self.model.training)

    def test_invalid_arguments(self):
        for kwargs in ({"query_batch": 0}, {"candidate_chunk": -1}, {"max_queries": 0}, {"max_queries": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                evaluate_full(self.model, self.triples, self.filter_map, "cpu", **kwargs)
        for triples in (torch.empty(0, 3, dtype=torch.long), self.triples.float(), torch.tensor([[6, 0, 0]]), torch.tensor([[0, -1, 0]])):
            with self.assertRaises(ValueError):
                evaluate_full(self.model, triples, self.filter_map, "cpu")


if __name__ == "__main__":
    unittest.main()
