"""No-linear SL(28): fixed coordinate injection, gradients and state handling."""
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "upstream/vl-kge"),
                str(ROOT / "vendor/sl-manifold-core/src")]

import torch
from torch import nn
from geometry.models_v2 import VLGeometryV2
from sl_manifold.core import coordinates_to_algebra
from vlkge.models.vlkge import VLKGEBase


def make_model(geometry="sl", **options):
    torch.manual_seed(42)
    features = torch.randn(4, 768) * 0.05
    config = dict(matrix_dim=28, entity_mapping="fixed_pad", score_chunk=2,
                  checkpoint_blocks=True)
    config.update(options)
    return VLGeometryV2(geometry=geometry, num_entities=4, num_relations=2,
                       embedding_dim=768, visual_features=features,
                       textual_features=features.clone(),
                       visual_entity_to_index={i: i for i in range(4)},
                       textual_entity_to_index={i: i for i in range(4)},
                       use_visual=True, use_textual=True, **config)


class FixedPadTests(unittest.TestCase):
    def test_exact_padding_without_learned_linear_or_extra_entity_parameters(self):
        model = make_model()
        self.assertIsNone(model.projection)
        self.assertFalse(any(isinstance(layer, nn.Linear) for layer in model.modules()))
        self.assertFalse(any("projection" in name for name, _ in model.named_parameters()))
        ids = torch.arange(4)
        fused = VLKGEBase.get_entity_representations(model, ids)
        coordinates = model.entity_coordinates(ids)
        self.assertEqual(coordinates.shape, (4, 783))
        torch.testing.assert_close(coordinates[:, :768], fused * 0.1, rtol=0, atol=0)
        self.assertEqual(int(torch.count_nonzero(coordinates[:, 768:])), 0)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),
                         4 * 768 + 2 * 783 + 2)
        algebra = coordinates_to_algebra(coordinates, 28, basis=model.sl_basis)
        torch.testing.assert_close(algebra.diagonal(dim1=-2, dim2=-1).sum(-1),
                                   torch.zeros(4), atol=1e-6, rtol=0)
        torch.testing.assert_close(algebra.norm(dim=(-2, -1)), coordinates.norm(dim=-1))

    def test_rejects_dimension_reduction_and_unknown_mapping(self):
        with self.assertRaisesRegex(ValueError, "cannot reduce"):
            make_model(matrix_dim=8)
        with self.assertRaisesRegex(ValueError, "unknown entity_mapping"):
            make_model(entity_mapping="truncate")

    def test_sl28_forward_backward_and_frozen_features(self):
        model = make_model()
        h, r, t = torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([2, 3])
        groups = model.get_geometry_representations(h)[1]
        torch.testing.assert_close(torch.linalg.det(groups), torch.ones(2), atol=2e-5, rtol=2e-5)
        score = model(h, r, t)
        self.assertTrue(torch.isfinite(score).all())
        score.sum().backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(float(model.entity_embeddings.weight.grad.abs().max()), 0)
        self.assertGreater(float(model.relation_embeddings.weight.grad.abs().max()), 0)
        self.assertIsNone(model.visual_embeddings.weight.grad)
        self.assertIsNone(model.textual_embeddings.weight.grad)

    def test_cache_checkpoint_device_empty_batch_and_diagnostics(self):
        # The author's frozen-feature gather allocates float32 output buffers;
        # keep the actual supported deployment dtype instead of changing it.
        model = make_model().to("cpu")
        clone = make_model(checkpoint_blocks=False).to("cpu")
        clone.load_state_dict(copy.deepcopy(model.state_dict()))
        h, r, t = torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([2, 3])
        live = model(h, r, t)
        torch.testing.assert_close(live, clone(h, r, t))
        model.eval()
        with torch.no_grad():
            cached = model(h, r, t)
            torch.testing.assert_close(cached, live)
        empty = torch.empty(0, dtype=torch.long)
        self.assertEqual(model(empty, empty, empty).dtype, torch.float32)
        diagnostic = model.diagnostics(sample_size=2)
        self.assertEqual(diagnostic["entity_mapping"], "fixed_pad")
        self.assertFalse(diagnostic["learned_entity_projection"])
        self.assertEqual(diagnostic["coordinate_dim"], 783)
        self.assertTrue(diagnostic["parameters_finite"])

    def test_euclidean_support_and_legacy_default_remain_available(self):
        fixed = make_model(geometry="euclidean")
        self.assertIsNone(fixed.projection)
        legacy = make_model(entity_mapping="linear", matrix_dim=8)
        self.assertIsInstance(legacy.projection, nn.Linear)
        self.assertEqual(tuple(legacy.projection.weight.shape), (63, 768))


if __name__ == "__main__":
    unittest.main()
