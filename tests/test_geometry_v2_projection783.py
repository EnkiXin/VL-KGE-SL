"""Selected experiment: author 768-D fusion -> Linear(768,783) -> SL(28)."""
import unittest

from tests.test_geometry_v2_fixed_pad import make_model
import torch
from torch import nn
from geometry.models_v2 import radial_clip
from vlkge.models.vlkge import VLKGEBase


class Projection783Tests(unittest.TestCase):
    def test_exact_selected_architecture_and_parameter_count(self):
        model = make_model(entity_mapping="linear")
        self.assertEqual(model.matrix_dim, 28)
        self.assertEqual(model.embedding_dim, 768)
        self.assertEqual(model.coordinate_dim, 783)
        self.assertEqual(model.entity_mapping, "linear")
        self.assertEqual(tuple(model.entity_embeddings.weight.shape), (4, 768))
        self.assertEqual(tuple(model.relation_embeddings.weight.shape), (2, 783))
        linears = {name: layer for name, layer in model.named_modules() if isinstance(layer, nn.Linear)}
        self.assertEqual(set(linears), {"projection"})
        self.assertEqual(tuple(model.projection.weight.shape), (783, 768))
        self.assertIsNone(model.projection.bias)
        self.assertIsInstance(model.visual_linear, nn.Identity)
        self.assertIsInstance(model.textual_linear, nn.Identity)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),
                         4 * 768 + 2 * 783 + 768 * 783 + 2)
        ids = torch.arange(4)
        fused = VLKGEBase.get_entity_representations(model, ids)
        self.assertEqual(fused.shape[-1], 768)
        expected = radial_clip(model.projection(fused) * 0.1, 2.0)
        torch.testing.assert_close(model.entity_coordinates(ids), expected, atol=0, rtol=0)

    def test_sl28_gradients_frozen_features_and_geometry_health(self):
        model = make_model(entity_mapping="linear")
        h, r, t = torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([2, 3])
        scores = model(h, r, t)
        self.assertTrue(torch.isfinite(scores).all())
        scores.sum().backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(float(model.projection.weight.grad.abs().max()), 0)
        self.assertIsNone(model.visual_embeddings.weight.grad)
        self.assertIsNone(model.textual_embeddings.weight.grad)
        diagnostic = model.diagnostics(sample_size=2)
        self.assertEqual(diagnostic["entity_mapping"], "linear")
        self.assertEqual(diagnostic["fused_dimension"], 768)
        self.assertTrue(diagnostic["learned_entity_projection"])
        self.assertLess(diagnostic["relative_score_matrices"]["log_max_rel_error_vs_eig"], 1e-4)


if __name__ == "__main__":
    unittest.main()
