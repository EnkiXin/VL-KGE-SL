"""Small CPU numerical/gradient checks, with no data or network dependency."""

import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "upstream/vl-kge"),
                str(ROOT.parent / "sl-manifold-core/src")]

import torch

from geometry.models import VLGeometry, mobius_add, poincare_exp0, poincare_squared_distance


class GeometryModelTests(unittest.TestCase):
    def make_model(self, geometry, chunk=3, checkpoint=True):
        torch.manual_seed(42)
        features = torch.randn(7, 768) * 0.05
        return VLGeometry(geometry=geometry, num_entities=7, num_relations=3,
                          embedding_dim=768, visual_features=features,
                          textual_features=features.clone(),
                          visual_entity_to_index={i: i for i in range(7)},
                          textual_entity_to_index={i: i for i in range(7)},
                          use_visual=True, use_textual=True,
                          score_chunk=chunk, checkpoint_blocks=checkpoint)

    def ids(self):
        return torch.tensor([0, 1, 2, 1, 0]), torch.tensor([0, 1, 2, 0, 1]), torch.tensor([2, 3, 4, 2, 1])

    def test_forward_backward_all_geometries(self):
        for geometry in VLGeometry.GEOMETRIES:
            with self.subTest(geometry=geometry):
                model = self.make_model(geometry)
                output = model(*self.ids())
                self.assertEqual(output.shape, (5,))
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertIsNone(model.visual_embeddings.weight.grad)
                self.assertEqual(model.get_entity_representations(torch.tensor([0])).shape, (1, 768))

    def test_chunk_and_checkpoint_gradient_equivalence(self):
        for geometry in VLGeometry.GEOMETRIES:
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

    def test_sl_determinant_and_diagnostics_rng(self):
        model = self.make_model("sl8")
        _, groups = model.get_geometry_representations(torch.arange(7))
        torch.testing.assert_close(torch.linalg.det(groups), torch.ones(7), atol=2e-5, rtol=2e-5)
        before = torch.random.get_rng_state().clone()
        diagnostic = model.diagnostics()
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(diagnostic["relative_score_matrices"]["branch_risk_fraction"], 0)

    def test_hyperbolic_origin_and_matched_relation_order(self):
        zero = torch.zeros(4, 63, requires_grad=True)
        point = poincare_exp0(zero)
        poincare_squared_distance(point, torch.zeros_like(point)).sum().backward()
        self.assertTrue(torch.isfinite(zero.grad).all())
        torch.testing.assert_close(zero.grad, torch.zeros_like(zero))
        model = self.make_model("murp", checkpoint=False)
        h, r, t = self.ids()
        hc, _ = model.get_geometry_representations(h)
        _, tp = model.get_geometry_representations(t)
        rp = model.get_relation_representations(r)
        expected = model.score_offset - model.logit_scale * poincare_squared_distance(
            poincare_exp0(hc * model.relation_diagonal(r)), mobius_add(tp, rp))
        torch.testing.assert_close(model(h, r, t), expected, atol=2e-5, rtol=2e-5)

    def test_cache_invalidation(self):
        model = self.make_model("sl8").eval()
        with torch.no_grad():
            expected = model(*self.ids())
            cache = model._evaluation_cache
            torch.testing.assert_close(model(*self.ids()), expected)
            self.assertIs(cache, model._evaluation_cache)
            model.projection.weight.add_(0.03)
            model(*self.ids())
            self.assertIsNot(cache, model._evaluation_cache)
            model.load_state_dict(copy.deepcopy(model.state_dict()))
            self.assertIsNone(model._evaluation_cache)
            model(*self.ids())
            model.to("cpu")
            self.assertIsNone(model._evaluation_cache)
            model(*self.ids())
            model.train()
            self.assertIsNone(model._evaluation_cache)
        model.eval()
        model(*self.ids()).sum().backward()
        self.assertIsNotNone(model.projection.weight.grad)
        self.assertIsNone(model._evaluation_cache)

    def test_shared_initialization_and_zero_chart(self):
        reference = self.make_model("euclidean")
        for geometry in VLGeometry.GEOMETRIES:
            with self.subTest(geometry=geometry):
                model = self.make_model(geometry)
                torch.testing.assert_close(model.entity_embeddings.weight,
                                           reference.entity_embeddings.weight, rtol=0, atol=0)
                torch.testing.assert_close(model.projection.weight,
                                           reference.projection.weight, rtol=0, atol=0)
                torch.testing.assert_close(model.relation_embeddings.weight,
                                           reference.relation_embeddings.weight, rtol=0, atol=0)
                with torch.no_grad():
                    model.projection.weight.zero_()
                    model.relation_embeddings.weight.zero_()
                model(*self.ids()).sum().backward()
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_log_parameter_scale_initialization_and_gradient(self):
        for initial in (0.1, 1.0, 25.0, 100.0):
            with self.subTest(initial=initial):
                model = VLGeometry(geometry="euclidean", num_entities=7,
                                   num_relations=3, embedding_dim=768,
                                   initial_logit_scale=initial)
                torch.testing.assert_close(model.logit_scale,
                                           torch.tensor(initial), rtol=2e-6, atol=1e-7)
                model.logit_scale.backward()
                self.assertTrue(torch.isfinite(model.raw_logit_scale.grad))
                torch.testing.assert_close(model.raw_logit_scale.grad,
                                           torch.tensor(initial), rtol=2e-6, atol=1e-7)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
