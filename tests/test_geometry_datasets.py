"""Author-release WikiArt policies, candidate ranking parity, and WN9 defaults."""

import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "upstream/vl-kge"), str(ROOT / "vendor/sl-manifold-core/src")]

import pandas as pd
import torch
from torch.utils.data import DataLoader

from geometry.datasets import (author_profile, loader_options, mark_training_entities,
                               epoch_training_frame, sample_negatives, check_v1_negative_capacity,
                               verified_manifest, AUTHOR_COMMIT)
from geometry.evaluation import evaluate_full
from geometry.models_v2 import VLGeometryV2
from vlkge import helpers
from vlkge.dataloader import KnowledgeGraphDataLoader, KGDataset
from vlkge.models.vlkge import VLKGEBase
from tests.test_geometry_protocol import TableModel

SPEC = importlib.util.spec_from_file_location("geometry_dataset_runner", ROOT / "scripts/run_geometry.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class DatasetProfileTests(unittest.TestCase):
    def test_selection_split_contract_never_labels_test_as_validation(self):
        self.assertEqual(RUNNER.selection_policy("validation")["tensor_index"], 1)
        test = RUNNER.selection_policy("test")
        self.assertEqual(test["tensor_index"], 2)
        self.assertEqual(test["event_key"], "test_selection")
        self.assertEqual(test["best_metric_key"], "best_test_mrr")
        self.assertEqual(test["metadata"], dict(selection_split="test", test_used_for_selection=True,
                                              evaluation_label="test_tuned_not_held_out"))
        args = RUNNER.parse_args(["--root", "/unused", "--run-dir", "/unused/out", "--model", "sl8",
                                  "--selection-split", "test"])
        self.assertEqual(args.selection_split, "test")

    def test_manifest_checks_dataset_commit_and_content(self):
        dataset = "wikiart_mkg_v1"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            prefix = f"vlkge/data/{dataset}/"
            records = []
            for relative in (f"{prefix}{dataset}_triples.csv", f"{prefix}features/{dataset}_vf_clip.pkl",
                             f"{prefix}features/{dataset}_tf_clip.pkl"):
                target = repo / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"test")
                records.append(dict(path=relative, bytes=4, sha256=hashlib.sha256(b"test").hexdigest()))
            path = root / "artifacts" / f"{dataset}-inputs.json"
            path.parent.mkdir()
            payload = dict(upstream_commit=AUTHOR_COMMIT, dataset=dataset, files=records)
            path.write_text(json.dumps(payload))
            self.assertEqual(verified_manifest(root, repo, dataset), payload)
            for field, value in (("dataset", "wn9_img"), ("upstream_commit", "wrong")):
                path.write_text(json.dumps({**payload, field: value}))
                with self.assertRaises(ValueError):
                    verified_manifest(root, repo, dataset)
            path.write_text(json.dumps(payload))
            (repo / records[0]["path"]).write_bytes(b"fail")
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                verified_manifest(root, repo, dataset)

    def test_runner_dataset_defaults_and_explicit_overrides(self):
        base = ["--root", "/unused", "--run-dir", "/unused/run", "--model", "slv2", "--chart-radius", "2"]
        for dataset, expected in [("wn9_img", (10, 100, .01, 50)),
                                  ("wikiart_mkg_v1", (50, 1, .1, None)),
                                  ("wikiart_mkg_v2", (20, 1, .1, None))]:
            args = RUNNER.parse_args(base + ["--dataset", dataset])
            self.assertEqual((args.epochs, args.negatives, args.lr, args.patience), expected)
            args = RUNNER.parse_args(base + ["--dataset", dataset, "--epochs", "7", "--negatives", "2",
                                             "--lr", ".03", "--patience", "6"])
            self.assertEqual((args.epochs, args.negatives, args.lr, args.patience), (7, 2, .03, 6))

    def test_author_profiles_preserve_literal_inverse_release_behavior(self):
        repo = ROOT / "upstream/vl-kge"
        wn = author_profile(repo, "wn9_img")
        self.assertFalse(wn["inductive"])
        for dataset in ("wikiart_mkg_v1", "wikiart_mkg_v2"):
            profile = author_profile(repo, dataset)
            self.assertTrue(profile["inductive"])
            self.assertTrue(profile["modality_asymmetry"])
            self.assertFalse(profile["bidirectional_eval"])
            self.assertTrue(profile["use_per_relation_candidates"])
            self.assertEqual(profile["num_neg_samples"], 1)
        v2 = author_profile(repo, "wikiart_mkg_v2")
        self.assertEqual(v2["add_inverse_relations"], {"isCreatedByArtist:hasCreatedArtwork": None})
        self.assertIsNone(loader_options(v2)["add_inverse_relations"])
        self.assertIn("adds no inverse", " ".join(v2["protocol_notes"]))

    def test_v2_loader_excludes_only_eval_and_uses_split_artist_pools(self):
        records = []
        for mode in ("train", "val", "test"):
            records.extend([(mode + "_art", "isCreatedByArtist", mode + "_artist", mode),
                            (mode + "_artist", "isInfluencedBy", mode + "_teacher", mode),
                            (mode + "_art", "isRelatedToArtwork", mode + "_other", mode)])
        frame = pd.DataFrame(records, columns=["head", "relation", "tail", "mode"])
        profile = author_profile(ROOT / "upstream/vl-kge", "wikiart_mkg_v2")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "triples.csv"
            frame.to_csv(path, index=False)
            data = KnowledgeGraphDataLoader(path, "wikiart_mkg_v2", **loader_options(profile))
        entities, relations = data.get_entities_and_relations()
        self.assertNotIn("hasCreatedArtwork", relations)
        train, val, test = data.split_data()
        self.assertEqual([len(train), len(val), len(test)], [3, 2, 2])
        artist_relation = relations["isInfluencedBy"]
        self.assertEqual(data.relation_to_valid_tails_train[artist_relation], [entities["train_teacher"]])
        self.assertEqual(data.relation_to_valid_tails_eval_val[artist_relation], [entities["val_teacher"]])
        self.assertEqual(data.relation_to_valid_tails_eval_test[artist_relation], [entities["test_teacher"]])
        artwork_relation = relations["isCreatedByArtist"]
        self.assertEqual(set(data.relation_to_valid_tails_eval_val[artwork_relation]),
                         {entities[f"{mode}_artist"] for mode in ("train", "val", "test")})

    def test_v2_downsample_matches_author_epoch_seed(self):
        frame = pd.DataFrame([(f"art{i}", "related", f"tail{i}", "train") for i in range(100)] +
                             [("keep", "attribute", "value", "train")],
                             columns=["head", "relation", "tail", "mode"])
        profile = {"downsample_relation": "related", "downsample_fraction": .1}
        for epoch_index in (0, 1, 5):
            expected = pd.concat([frame[frame.relation != "related"],
                                  frame[frame.relation == "related"].sample(frac=.1, random_state=42 + epoch_index)],
                                 ignore_index=True)
            pd.testing.assert_frame_equal(epoch_training_frame(frame, profile, 42, epoch_index), expected)
        self.assertIs(epoch_training_frame(frame, {}, 42, 0), frame)

    def test_negative_dispatch_matches_author_rng_and_candidates(self):
        h, r, t = torch.tensor([0, 1]), torch.tensor([0, 0]), torch.tensor([2, 3])
        model, filt, pool = SimpleNamespace(num_entities=8), {(0, 0): {2}, (1, 0): {3}}, {0: [2, 3, 4, 5, 6]}
        for dataset, sampler in [("wn9_img", helpers.negative_sampling_uniform),
                                 ("wikiart_mkg_v1", helpers.negative_sampling_per_relation),
                                 ("wikiart_mkg_v2", helpers.negative_sampling_per_relation_fast)]:
            first, second = torch.Generator().manual_seed(42), torch.Generator().manual_seed(42)
            actual = sample_negatives(helpers, dataset, h, r, t, model, 1, filt, {}, first, pool)
            expected = (sampler(h, r, t, 8, 1, filt, {}, second, use_bernoulli=False)
                        if dataset == "wn9_img" else sampler(h, r, t, 1, filt, pool, second))
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
            self.assertTrue(torch.equal(first.get_state(), second.get_state()))

    def test_v1_negative_shortage_is_not_silently_repaired(self):
        frame = pd.DataFrame([("h", "r", "t", "train")], columns=["head", "relation", "tail", "mode"])
        with self.assertRaisesRegex(ValueError, "no sampling fallback"):
            check_v1_negative_capacity(frame, {"h": 0, "t": 1}, {"r": 0}, {(0, 0): {1}}, {0: [1]}, 1)

    def test_sparse_fusion_inductive_mask_and_projection_are_author_equivalent(self):
        torch.manual_seed(42)
        model = VLGeometryV2(num_entities=4, num_relations=1, embedding_dim=768, matrix_dim=28,
                             entity_mapping="linear", visual_features=torch.ones(2, 768) * .2,
                             textual_features=torch.ones(1, 768) * .4,
                             visual_entity_to_index={0: 0, 2: 1}, textual_entity_to_index={1: 0},
                             use_visual=True, use_textual=True,
                             inductive=True, modality_asymmetry=True, device="cpu")
        frame = pd.DataFrame([("seen_art", "r", "attribute", "train")],
                             columns=["head", "relation", "tail", "mode"])
        self.assertEqual(mark_training_entities(model, frame, {"seen_art": 0, "attribute": 1}), 2)
        ids = torch.arange(4)
        self.assertEqual(model.seen_in_train.tolist(), [True, True, False, False])
        fused = VLKGEBase.get_entity_representations(model, ids)
        torch.testing.assert_close(fused[0], (model.entity_embeddings.weight[0] + .2) / 2)
        torch.testing.assert_close(fused[1], (model.entity_embeddings.weight[1] + .4) / 2)
        torch.testing.assert_close(fused[2], torch.ones(768) * .1)
        torch.testing.assert_close(fused[3], torch.zeros(768))
        coordinates = model.entity_coordinates(ids)
        torch.testing.assert_close(coordinates, model.projection(fused) * .1)
        coordinates.sum().backward()
        self.assertEqual(float(model.entity_embeddings.weight.grad[2:].abs().max()), 0)


class WikiArtEvaluationTests(unittest.TestCase):
    def setUp(self):
        ids = torch.arange(8)
        self.model = TableModel(((ids[:, None, None] * 2 + ids[None, None, :] +
                                  torch.arange(2)[None, :, None]) % 4).float())
        self.triples = torch.tensor([[0, 0, 4], [1, 0, 6], [2, 1, 7], [3, 0, 5], [1, 1, 3]])
        self.filter_map = {(0, 0): {2, 4}, (1, 0): {3, 6}, (2, 1): {0, 7}, (3, 0): {4, 5}}
        self.pools = {0: [5, 3, 4, 2, 3], 1: [0, 2, 3]}

    def author_metrics(self, triples, pools):
        frame = pd.DataFrame([(int(h), int(r), int(t), "val") for h, r, t in triples],
                             columns=["head", "relation", "tail", "mode"])
        loader = DataLoader(KGDataset(frame, {i: i for i in range(8)}, {i: i for i in range(2)}), batch_size=2)
        return helpers.evaluate_kge(self.model, loader, self.filter_map, bidirectional=False,
                                    relation_to_valid_tails=pools, device="cpu")

    def test_chunked_relation_pools_match_real_author_evaluator(self):
        expected_mrr, expected_hits, expected_relations = self.author_metrics(self.triples, self.pools)
        ranks = [int(round(1 / self.author_metrics(row[None], self.pools)[0])) for row in self.triples]
        for query_batch in (1, 2, 10):
            for candidate_chunk in (1, 3, 20):
                with self.subTest(query_batch=query_batch, candidate_chunk=candidate_chunk):
                    result = evaluate_full(self.model, self.triples, self.filter_map, "cpu",
                                           query_batch, candidate_chunk, bidirectional=False,
                                           relation_to_valid_tails=self.pools)
                    self.assertEqual(result["mrr"], expected_mrr)
                    self.assertEqual(result["ranks_tail"], ranks)
                    self.assertEqual(result["ranks_head"], [])
                    self.assertEqual(result["directional_queries"], len(self.triples))
                    self.assertEqual(result["hits"], {str(k): v for k, v in expected_hits.items()})
                    for relation, value in expected_relations.items():
                        self.assertEqual(result["per_relation"][str(relation)]["mrr"], value["MRR"])

    def test_missing_pool_restores_target_and_rng_mode_cap(self):
        self.model.train()
        rng = torch.get_rng_state().clone()
        result = evaluate_full(self.model, self.triples, self.filter_map, "cpu", max_queries=2,
                               bidirectional=False, relation_to_valid_tails={})
        self.assertEqual(result["ranks_tail"], [1, 1])
        self.assertFalse(result["complete"])
        self.assertTrue(self.model.training)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
