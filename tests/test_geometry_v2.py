"""CPU checks for geometry v2: logarithm accuracy, scoring, gradients, evaluator compatibility."""

import copy
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "upstream/vl-kge"),
                str(ROOT / "vendor/sl-manifold-core/src"), str(ROOT.parent / "sl-manifold-core/src")]

import torch

from sl_manifold.core import algebra_exp, coordinates_to_algebra, orthonormal_sl_basis
from geometry.evaluation import evaluate_full
from geometry.models_v2 import (VLGeometryV2, denman_beavers_sqrt, log_accuracy_check, principal_log,
                                principal_log_with_flags, radial_clip, sl_distance, spectrum_diagnostics)


def random_group(count, n, norm, generator, dtype=torch.float64):
    basis = orthonormal_sl_basis(n, dtype=dtype)
    coords = torch.randn(count, n * n - 1, generator=generator, dtype=dtype)
    coords = coords / coords.norm(dim=-1, keepdim=True) * norm
    return algebra_exp(coordinates_to_algebra(coords, n, basis=basis))


class LogarithmTests(unittest.TestCase):
    def test_sqrt_and_log_accuracy_across_norms(self):
        generator = torch.Generator().manual_seed(0)
        for n in (4, 8):
            for norm in (0.5, 1.5, 2.5):
                with self.subTest(n=n, norm=norm):
                    a = random_group(64, n, norm, generator)
                    b = random_group(64, n, norm, generator)
                    relative = torch.linalg.solve(a, b)
                    root = denman_beavers_sqrt(relative, 6)
                    torch.testing.assert_close(root @ root, relative, atol=1e-8, rtol=1e-8)
                    check = log_accuracy_check(relative, sqrt_steps=1, db_iterations=6, terms=12)
                    self.assertLess(check["log_max_rel_error_vs_eig"], 1e-5)
                    self.assertEqual(check["log_fallback_fraction"], 0.0)
                    stats = spectrum_diagnostics(relative)
                    self.assertEqual(stats["nonfinite_fraction"], 0.0)
                    self.assertGreater(stats["principal_domain_min_margin_rad"], 0.0)

    def test_fallback_flags_matrices_without_real_principal_log(self):
        # A rotation by pi has eigenvalues on the negative real axis: no real principal log.
        theta = torch.tensor(math.pi, dtype=torch.float64)
        rotation = torch.tensor([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
                                dtype=torch.float64)
        block = torch.block_diag(rotation, torch.eye(2, dtype=torch.float64))
        good = torch.eye(4, dtype=torch.float64) * 1.0
        good[0, 1] = 0.3
        good.requires_grad_(True)
        batch = torch.stack([block, good])
        logarithm, flagged = principal_log_with_flags(batch)
        self.assertTrue(bool(flagged[0]))
        self.assertFalse(bool(flagged[1]))
        self.assertTrue(torch.isfinite(logarithm).all())
        logarithm.square().sum().backward()
        self.assertTrue(torch.isfinite(good.grad).all())
        # The flagged matrix is scored as far away, never as identical.
        self.assertGreater(float(torch.linalg.matrix_norm(logarithm[0])), 1.0)

    def test_one_sqrt_step_beats_plain_gregory_far_from_identity(self):
        generator = torch.Generator().manual_seed(1)
        a, b = random_group(128, 8, 3.0, generator), random_group(128, 8, 3.0, generator)
        relative = torch.linalg.solve(a, b)
        plain = log_accuracy_check(relative, sqrt_steps=0, terms=12)["log_max_rel_error_vs_eig"]
        scaled = log_accuracy_check(relative, sqrt_steps=1, terms=12)["log_max_rel_error_vs_eig"]
        self.assertLess(scaled, 1e-6)
        self.assertGreater(plain, scaled)

    def test_distance_properties(self):
        generator = torch.Generator().manual_seed(2)
        a, b = random_group(32, 8, 1.5, generator), random_group(32, 8, 1.5, generator)
        torch.testing.assert_close(sl_distance(a, a), torch.zeros(32, dtype=torch.float64), atol=1e-9, rtol=0)
        torch.testing.assert_close(sl_distance(a, b), sl_distance(b, a), atol=1e-8, rtol=1e-8)
        g = random_group(1, 8, 1.0, generator)
        torch.testing.assert_close(sl_distance(g @ a, g @ b), sl_distance(a, b), atol=1e-8, rtol=1e-8)
        self.assertTrue((sl_distance(a, b) > 0).all())

    def test_log_is_differentiable_in_float32(self):
        generator = torch.Generator().manual_seed(3)
        coords = torch.randn(6, 63, generator=generator) * 0.3
        coords.requires_grad_(True)
        basis = orthonormal_sl_basis(8)
        groups = algebra_exp(coordinates_to_algebra(coords, 8, basis=basis))
        distance = sl_distance(groups[:3], groups[3:])
        distance.sum().backward()
        self.assertTrue(torch.isfinite(coords.grad).all())
        self.assertGreater(coords.grad.abs().max(), 0)

    def test_radial_clip(self):
        x = torch.tensor([[3.0, 4.0], [0.3, 0.4], [0.0, 0.0]])
        y = radial_clip(x, 1.0)
        torch.testing.assert_close(y[0], torch.tensor([0.6, 0.8]))
        torch.testing.assert_close(y[1], x[1])
        torch.testing.assert_close(y[2], x[2])


class ModelTests(unittest.TestCase):
    def make_model(self, geometry, matrix_dim=8, chunk=3, checkpoint=True, **extra):
        torch.manual_seed(42)
        features = torch.randn(7, 768) * 0.05
        return VLGeometryV2(geometry=geometry, matrix_dim=matrix_dim, num_entities=7, num_relations=3,
                            embedding_dim=768, visual_features=features, textual_features=features.clone(),
                            visual_entity_to_index={i: i for i in range(7)},
                            textual_entity_to_index={i: i for i in range(7)},
                            use_visual=True, use_textual=True, score_chunk=chunk,
                            checkpoint_blocks=checkpoint, **extra)

    def ids(self):
        return torch.tensor([0, 1, 2, 1, 0]), torch.tensor([0, 1, 2, 0, 1]), torch.tensor([2, 3, 4, 2, 1])

    def test_forward_backward_all_geometries_and_modes(self):
        for geometry in VLGeometryV2.GEOMETRIES:
            for mode in VLGeometryV2.RELATION_MODES:
                for n in (4, 8):
                    with self.subTest(geometry=geometry, mode=mode, n=n):
                        model = self.make_model(geometry, matrix_dim=n, relation_mode=mode)
                        output = model(*self.ids())
                        self.assertEqual(output.shape, (5,))
                        self.assertTrue(torch.isfinite(output).all())
                        output.square().mean().backward()
                        for name, parameter in model.named_parameters():
                            if parameter.requires_grad:
                                self.assertIsNotNone(parameter.grad, name)
                                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                        self.assertIsNone(model.visual_embeddings.weight.grad)

    def test_chunk_and_checkpoint_equivalence(self):
        for geometry in VLGeometryV2.GEOMETRIES:
            with self.subTest(geometry=geometry):
                left = self.make_model(geometry, chunk=2)
                right = self.make_model(geometry, chunk=100, checkpoint=False)
                right.load_state_dict(copy.deepcopy(left.state_dict()))
                a, b = left(*self.ids()), right(*self.ids())
                torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
                a.sum().backward()
                b.sum().backward()
                for (name, p), (_, q) in zip(left.named_parameters(), right.named_parameters()):
                    if p.requires_grad:
                        torch.testing.assert_close(p.grad, q.grad, atol=5e-5, rtol=5e-5, msg=name)

    def test_sl_determinant_relation_init_and_score_form(self):
        model = self.make_model("sl", relation_init_norm=0.5, initial_logit_scale=3.0, initial_offset=3.0)
        _, groups = model.get_geometry_representations(torch.arange(7))
        torch.testing.assert_close(torch.linalg.det(groups), torch.ones(7), atol=2e-5, rtol=2e-5)
        relation_norm = model.relation_coordinates(torch.arange(3))[0].norm(dim=-1)
        self.assertGreater(float(relation_norm.detach().mean()), 0.2)
        self.assertLess(float(relation_norm.detach().mean()), 1.0)
        # Score is offset - scale * D with D = 0 when the transformed head equals the tail.
        h, r, t = self.ids()
        head, tail = model.get_geometry_representations(h)[1], model.get_geometry_representations(t)[1]
        relation = model.get_relation_representations(r)
        d = model.discrepancy(model.transform_head(head, relation), tail)
        torch.testing.assert_close(model(h, r, t), 3.0 - 3.0 * d)
        torch.testing.assert_close(model.discrepancy(head, head), torch.zeros(5), atol=1e-6, rtol=0)

    def test_euclidean_control_matches_translation_formula(self):
        model = self.make_model("euclidean", checkpoint=False)
        h, r, t = self.ids()
        hc = model.entity_coordinates(h)
        tc = model.entity_coordinates(t)
        rc = model.relation_coordinates(r)[0]
        expected = model.score_offset - model.logit_scale * (hc + rc - tc).norm(dim=-1)
        torch.testing.assert_close(model(h, r, t), expected)

    def test_eval_cache_and_evaluator(self):
        model = self.make_model("sl", chunk=4)
        triples = torch.tensor([[0, 0, 2], [1, 1, 3], [2, 2, 4]])
        filter_map = {(0, 0): {2}, (1, 1): {3}, (2, 2): {4}, (2, 0): {0}, (3, 1): {1}, (4, 2): {2}}
        model.eval()
        with torch.no_grad():
            first = evaluate_full(model, triples, filter_map, "cpu", query_batch=2, candidate_chunk=3)
            second = evaluate_full(model, triples, filter_map, "cpu", query_batch=3, candidate_chunk=7)
        self.assertEqual(first["ranks_head"], second["ranks_head"])
        self.assertEqual(first["ranks_tail"], second["ranks_tail"])
        self.assertTrue(first["complete"])
        # Scores in eval/no_grad (cached) must equal scores computed with grad enabled.
        h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]
        with torch.no_grad():
            cached = model(h, r, t)
        model.train()
        live = model(h, r, t)
        torch.testing.assert_close(cached, live.detach(), atol=1e-5, rtol=1e-5)

    def test_diagnostics_are_finite_and_rng_free(self):
        for geometry in VLGeometryV2.GEOMETRIES:
            with self.subTest(geometry=geometry):
                model = self.make_model(geometry)
                before = torch.random.get_rng_state().clone()
                diagnostic = model.diagnostics(sample_size=5)
                self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
                self.assertTrue(diagnostic["parameters_finite"])
                if geometry == "sl":
                    self.assertEqual(diagnostic["relative_score_matrices"]["nonfinite_fraction"], 0.0)
                    self.assertLess(diagnostic["relative_score_matrices"]["log_max_rel_error_vs_eig"], 1e-4)
                    self.assertGreater(diagnostic["relative_score_matrices"]["principal_domain_min_margin_rad"], 0.0)

    def test_one_training_step_with_author_loss(self):
        sys.path.insert(0, str(ROOT / "upstream/vl-kge"))
        from vlkge import helpers
        model = self.make_model("sl", chunk=8)
        optimizer = torch.optim.Adagrad(model.parameters(), lr=0.03)
        h, r, t = self.ids()
        before = model(h, r, t).detach().clone()
        for _ in range(3):
            optimizer.zero_grad()
            pos = model(h, r, t)
            neg = model(h.repeat_interleave(4), r.repeat_interleave(4),
                        torch.tensor([3, 4, 5, 6] * 5)).view(5, 4)
            loss = helpers.compute_logistic_loss(pos, neg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
        after = model(h, r, t).detach()
        self.assertTrue(torch.isfinite(after).all())
        self.assertGreater(float((after - before).abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
