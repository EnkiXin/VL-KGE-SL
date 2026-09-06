"""CPU checks of author parity and the separately gated geometric residual."""

import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
CORE = next(path for path in (ROOT / "vendor/sl-manifold-core/src", ROOT.parent / "sl-manifold-core/src")
            if (path / "sl_manifold/core.py").is_file())
sys.path[:0] = [str(ROOT), str(ROOT / "upstream/vl-kge"),
                str(CORE)]

import torch
import torch.nn.functional as F

from geometry.distmult_sl import DistMultSL
from vlkge.models.distmult import DistMult


class DistMultSLTests(unittest.TestCase):
    def kwargs(self):
        features = torch.arange(7 * 768, dtype=torch.float32).reshape(7, 768).sin() * 0.05
        return dict(num_entities=7, num_relations=3, embedding_dim=768,
                    visual_features=features, textual_features=features.clone(),
                    visual_entity_to_index={i: i for i in range(7)},
                    textual_entity_to_index={i: i for i in range(7)},
                    use_visual=True, use_textual=True)

    def make(self, geometry="sl8", **extra):
        torch.manual_seed(42)
        return DistMultSL(geometry=geometry, **self.kwargs(), **extra)

    def triples(self):
        return torch.tensor([[0, 0, 2], [1, 1, 3], [2, 2, 4], [1, 0, 2],
                             [0, 1, 1], [6, 0, 1], [5, 1, 2], [3, 2, 1]])

    def score(self, model):
        triples = self.triples()
        return model(triples[:, 0], triples[:, 1], triples[:, 2])

    def loss(self, score):
        return F.softplus(-score[:4]).mean() + F.softplus(score[4:]).mean()

    def test_same_seed_base_initialization_and_rng_exact(self):
        torch.manual_seed(42)
        baseline = DistMult(**self.kwargs())
        expected_rng = torch.random.get_rng_state().clone()
        for geometry in ("sl8", "euclidean"):
            with self.subTest(geometry=geometry):
                model = self.make(geometry)
                self.assertTrue(torch.equal(expected_rng, torch.random.get_rng_state()))
                for name, parameter in baseline.named_parameters():
                    self.assertTrue(torch.equal(parameter, dict(model.named_parameters())[name]), name)
                self.assertEqual(model.get_entity_representations(torch.tensor([0])).shape, (1, 768))
                self.assertEqual(model.get_relation_representations(torch.tensor([0])).shape, (1, 768))

    def test_disabled_exact_score_loss_gradient_adagrad_step(self):
        torch.manual_seed(42)
        baseline = DistMult(**self.kwargs())
        model = self.make(enabled=False)
        # A broken disabled branch must have no effect on the exact fallback.
        with torch.no_grad():
            model.entity_sl_projection.weight.fill_(float("nan"))
        original_score, residual_score = self.score(baseline), self.score(model)
        self.assertTrue(torch.equal(original_score, residual_score))
        original_loss, residual_loss = self.loss(original_score), self.loss(residual_score)
        self.assertTrue(torch.equal(original_loss, residual_loss))
        original_loss.backward()
        residual_loss.backward()
        model_params = dict(model.named_parameters())
        for name, parameter in baseline.named_parameters():
            other = model_params[name]
            if parameter.requires_grad:
                self.assertTrue(torch.equal(parameter.grad, other.grad), name)
        self.assertIsNone(model.entity_sl_projection.weight.grad)
        torch.optim.Adagrad(baseline.parameters(), lr=0.1).step()
        torch.optim.Adagrad(model.parameters(), lr=0.1).step()
        for name, parameter in baseline.named_parameters():
            self.assertTrue(torch.equal(parameter, model_params[name]), name)

    def test_calibration_and_zero_gate_then_projection_gradients(self):
        for geometry in ("sl8", "euclidean"):
            with self.subTest(geometry=geometry):
                model = self.make(geometry)
                with self.assertRaises(RuntimeError):
                    self.score(model)
                before = torch.random.get_rng_state().clone()
                summary = model.calibrate(self.triples())
                self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
                self.assertTrue(summary["calibrated"])
                self.assertGreater(summary["scale"], 0)
                with self.assertRaises(RuntimeError):
                    model.calibrate(self.triples())
                self.assertFalse(model.calibration_center.requires_grad)
                self.assertFalse(model.calibration_scale.requires_grad)
                original = DistMult.forward(model, self.triples()[:, 0],
                                            self.triples()[:, 1], self.triples()[:, 2])
                self.assertTrue(torch.equal(self.score(model), original))
                optimizer = torch.optim.Adagrad(model.parameters(), lr=0.1)
                self.loss(self.score(model)).backward()
                self.assertTrue(torch.isfinite(model.raw_gate.grad))
                self.assertGreater(float(model.raw_gate.grad.abs()), 0)
                self.assertEqual(float(model.entity_sl_projection.weight.grad.abs().max()), 0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                self.loss(self.score(model)).backward()
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(float(model.entity_sl_projection.weight.grad.norm()), 0)
                self.assertGreater(float(model.relation_sl_projection.weight.grad.norm()), 0)

    def test_chunk_checkpoint_equivalence(self):
        for geometry in ("sl8", "euclidean"):
            with self.subTest(geometry=geometry):
                left = self.make(geometry, score_chunk=2)
                right = self.make(geometry, score_chunk=100, checkpoint_blocks=False)
                left.calibrate(self.triples())
                with torch.no_grad():
                    left.raw_gate.fill_(0.1)
                right.load_state_dict(copy.deepcopy(left.state_dict()))
                a, b = self.score(left), self.score(right)
                torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-5)
                self.loss(a).backward()
                self.loss(b).backward()
                for (name, p), (_, q) in zip(left.named_parameters(), right.named_parameters()):
                    if p.requires_grad:
                        torch.testing.assert_close(p.grad, q.grad, rtol=5e-4, atol=5e-5, msg=name)

    def test_cache_and_sample_diagnostics(self):
        model = self.make()
        model.calibrate(self.triples())
        model.eval()
        with torch.no_grad():
            self.score(model)
            cached = model._evaluation_cache
            self.score(model)
            self.assertIs(cached, model._evaluation_cache)
            model.entity_sl_projection.weight.add_(0.01)
            self.score(model)
            self.assertIsNot(cached, model._evaluation_cache)
            model.load_state_dict(copy.deepcopy(model.state_dict()))
            self.assertIsNone(model._evaluation_cache)
            self.score(model)
            model.to("cpu")
            self.assertIsNone(model._evaluation_cache)
            self.score(model)
            model.train()
            self.assertIsNone(model._evaluation_cache)
            rng = torch.random.get_rng_state().clone()
            summary = model.diagnostics()
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            self.assertEqual(summary["relative_score_matrices"]["branch_risk_fraction"], 0)
        model.eval()
        self.score(model).sum().backward()
        self.assertIsNone(model._evaluation_cache)

    def test_config_and_nonfinite_failures(self):
        for key, value in (("coordinate_scale", float("nan")),
                           ("chart_radius", float("inf")), ("score_chunk", 1.5),
                           ("residual_weight", -1)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.make(**{key: value})
        model = self.make()
        model.calibrate(self.triples())
        for value in (float("nan"), float("inf")):
            with torch.no_grad():
                model.raw_gate.fill_(value)
            with self.assertRaises(FloatingPointError):
                self.score(model)
        with torch.no_grad():
            model.raw_gate.zero_()
            model.calibration_center.fill_(float("inf"))
        with self.assertRaises(FloatingPointError):
            self.score(model)

    def test_calibration_floor_frozen_and_checkpoint_roundtrip(self):
        model = self.make()
        result = model.calibrate(self.triples())
        self.assertEqual(result["scale_floor_multiplier"], 4.0)
        self.assertEqual(result["scale_floor"], 1.0)
        self.assertGreaterEqual(result["scale"], result["empirical_std"])
        self.assertGreaterEqual(result["scale"], result["scale_floor"])
        original_scale = model.calibration_scale.clone()
        with torch.no_grad():
            model.entity_sl_projection.weight.mul_(2)
        before = torch.random.get_rng_state().clone()
        probe = model.compatibility_diagnostics()
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(probe["source"], "train_only_calibration_probe")
        self.assertTrue(torch.equal(model.calibration_scale, original_scale))
        restored = self.make()
        restored.load_state_dict(copy.deepcopy(model.state_dict()))
        self.assertEqual(restored.compatibility_diagnostics(), probe)
        self.assertTrue(torch.equal(restored.calibration_probe_triples,
                                    model.calibration_probe_triples))
        self.assertEqual(tuple(restored.calibration_probe_triples.shape), (256, 3))

    def test_zero_variance_calibration_rejected(self):
        model = self.make()
        with self.assertRaises(ValueError):
            model.calibrate(self.triples()[0:1])
        self.assertFalse(bool(model.calibrated))

    def test_real_author_logistic_steps_retain_variable_compatibility(self):
        from vlkge.helpers import compute_logistic_loss
        for geometry in ("sl8", "euclidean"):
            with self.subTest(geometry=geometry):
                model = self.make(geometry)
                model.calibrate(self.triples())
                optimizer = torch.optim.Adagrad(model.parameters(), lr=0.1)
                triples = self.triples()
                for step in range(4):
                    positive = model(triples[:4, 0], triples[:4, 1], triples[:4, 2])
                    negatives = model(triples[4:, 0], triples[4:, 1], triples[4:, 2])
                    loss = compute_logistic_loss(positive, negatives[:, None].expand(-1, 100))
                    optimizer.zero_grad()
                    loss.backward()
                    self.assertTrue(torch.isfinite(model.raw_gate.grad))
                    self.assertGreater(float(model.raw_gate.grad.abs()), 0)
                    optimizer.step()
                probe = model.compatibility_diagnostics()
                self.assertGreater(probe["q_std"], 1e-6)
                self.assertLess(probe["saturation_abs_q_gt_0_99"], 0.5)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
