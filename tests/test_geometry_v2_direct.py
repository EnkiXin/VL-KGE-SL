"""SL(28) with an initial 783-D representation and no post-fusion mapping."""
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "upstream/vl-kge"),
                str(ROOT / "vendor/sl-manifold-core/src")]

import torch
from torch import nn
from geometry.models_v2 import VLGeometryV2, radial_clip
from vlkge.models.vlkge import VLKGEBase


def make_direct(**options):
    torch.manual_seed(42)
    features = torch.randn(4, 768) * 0.05
    config = dict(matrix_dim=28, embedding_dim=783, entity_mapping="direct",
                  score_chunk=2, checkpoint_blocks=True)
    config.update(options)
    return VLGeometryV2(num_entities=4, num_relations=2, visual_features=features,
                       textual_features=features.clone(), use_visual=True, use_textual=True,
                       visual_entity_to_index={i: i for i in range(4)},
                       textual_entity_to_index={i: i for i in range(4)}, **config)


class Direct783Tests(unittest.TestCase):
    def test_initial_entity_and_relation_dimensions_and_author_modality_alignment(self):
        model = make_direct()
        self.assertEqual(tuple(model.entity_embeddings.weight.shape), (4, 783))
        self.assertEqual(tuple(model.relation_embeddings.weight.shape), (2, 783))
        self.assertIsNone(model.projection)
        self.assertEqual(tuple(model.visual_embeddings.weight.shape), (4, 768))
        self.assertEqual(tuple(model.textual_embeddings.weight.shape), (4, 768))
        linears = {name: layer for name, layer in model.named_modules() if isinstance(layer, nn.Linear)}
        self.assertEqual(set(linears), {"visual_linear", "textual_linear"})
        for layer in linears.values():
            self.assertEqual((layer.in_features, layer.out_features), (768, 783))
            self.assertIsNone(layer.bias)
        expected = 4 * 783 + 2 * 783 + 2 * (768 * 783) + 2
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), expected)

    def test_fused_783_coordinates_are_used_directly_without_padding_or_projection(self):
        model = make_direct()
        ids = torch.arange(4)
        fused = VLKGEBase.get_entity_representations(model, ids)
        coordinates = model.entity_coordinates(ids)
        self.assertEqual(fused.shape, (4, 783))
        torch.testing.assert_close(coordinates, radial_clip(fused * 0.1, 2.0), rtol=0, atol=0)
        self.assertGreater(int(torch.count_nonzero(coordinates[:, 768:])), 0)
        groups = model.get_geometry_representations(ids)[1]
        torch.testing.assert_close(torch.linalg.det(groups), torch.ones(4), atol=2e-5, rtol=2e-5)

    def test_forward_backward_checkpoint_and_cache(self):
        model = make_direct()
        other = make_direct(checkpoint_blocks=False)
        other.load_state_dict(copy.deepcopy(model.state_dict()))
        h, r, t = torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([2, 3])
        live = model(h, r, t)
        torch.testing.assert_close(live, other(h, r, t))
        live.sum().backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertIsNone(model.visual_embeddings.weight.grad)
        self.assertIsNone(model.textual_embeddings.weight.grad)
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(h, r, t), live)
        diagnostic = model.diagnostics(sample_size=2)
        self.assertEqual(diagnostic["entity_mapping"], "direct")
        self.assertEqual(diagnostic["fused_dimension"], 783)
        self.assertFalse(diagnostic["learned_entity_projection"])

    def test_rejects_fusion_geometry_dimension_mismatch(self):
        with self.assertRaisesRegex(ValueError, "embedding_dim == coordinate_dim"):
            make_direct(embedding_dim=768)
        with self.assertRaisesRegex(ValueError, "embedding_dim == coordinate_dim"):
            make_direct(matrix_dim=8)


if __name__ == "__main__":
    unittest.main()
