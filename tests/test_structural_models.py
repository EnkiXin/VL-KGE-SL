"""CPU mathematical checks for matched structural geometry models."""

import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "upstream/vl-kge"), str(ROOT / "vendor/sl-manifold-core/src"), str(ROOT.parent / "sl-manifold-core/src")]

import numpy as np
import torch
from scipy.linalg import logm

from structural_kge.models import StructuralKGE, mobius_add, poincare_exp0, poincare_half_distance
from structural_kge.matrix_log import matrix_log_quadrature, relative_log_quadrature, log_pair_diagnostics


class StructuralModelTests(unittest.TestCase):
    def make(self, geometry, **kwargs):
        torch.manual_seed(42)
        return StructuralKGE(11, 3, geometry=geometry, **kwargs)

    def ids(self):
        return torch.tensor([0, 1, 2, 1]), torch.tensor([0, 1, 2, 0]), torch.tensor([2, 3, 4, 5])

    def test_equal_parameters_and_initialization(self):
        models = [self.make(geometry) for geometry in StructuralKGE.GEOMETRIES]
        counts = [sum(p.numel() for p in model.parameters()) for model in models]
        self.assertEqual(len(set(counts)), 1)
        for model in models[1:]:
            for p, q in zip(models[0].parameters(), model.parameters()):
                self.assertTrue(torch.equal(p, q))
        self.assertEqual(models[0].coordinate_radius,1.5)
        self.assertEqual(counts[0],(11+3)*63+2)
        for model in models:
            self.assertFalse(hasattr(model,'entity_bias'))
            for table in (model.entity_embeddings,model.relation_embeddings):
                torch.testing.assert_close(model._bound(table.weight).norm(dim=-1),torch.full((len(table.weight),),.5))

    def test_forward_backward_zero_and_regularization(self):
        for geometry in StructuralKGE.GEOMETRIES:
            with self.subTest(geometry=geometry):
                model = self.make(geometry)
                score = model(*self.ids())
                self.assertEqual(score.shape, (4,))
                (score.mean() + 0.01 * model.regularization(*self.ids())).backward()
                for p in model.parameters():
                    self.assertIsNotNone(p.grad)
                    self.assertTrue(torch.isfinite(p.grad).all())
                with torch.no_grad():
                    model.entity_embeddings.weight.zero_()
                    model.relation_embeddings.weight.zero_()
                model.zero_grad()
                model(*self.ids()).sum().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_chunk_checkpoint_cache_and_diagnostics(self):
        for geometry in StructuralKGE.GEOMETRIES:
            with self.subTest(geometry=geometry):
                a = self.make(geometry, score_chunk=2)
                b = self.make(geometry, score_chunk=100, checkpoint_blocks=False)
                b.load_state_dict(copy.deepcopy(a.state_dict()))
                x, y = a(*self.ids()), b(*self.ids())
                torch.testing.assert_close(x, y, rtol=3e-5, atol=3e-5)
                x.sum().backward(); y.sum().backward()
                for p, q in zip(a.parameters(), b.parameters()):
                    torch.testing.assert_close(p.grad, q.grad, rtol=5e-5, atol=5e-5)
                a.eval()
                with torch.no_grad():
                    cached = a.prepare_eval_cache()
                    torch.testing.assert_close(a(*self.ids()), x.detach())
                    self.assertIs(cached, a._eval_cache)
                    a.entity_embeddings.weight.add_(0.001)
                    a(*self.ids())
                    self.assertIsNot(cached, a._eval_cache)
                    a.load_state_dict(copy.deepcopy(a.state_dict()))
                    self.assertIsNone(a._eval_cache)
                    rng = torch.random.get_rng_state().clone()
                    diagnostic = a.diagnostics(*self.ids(), scipy_reference=geometry == "sl8")
                    self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
                    if geometry == "sl8":
                        self.assertTrue(diagnostic["principal_log"]["passed"])

    def test_local_geometry_parity_and_hyperbolic_action_order(self):
        models = [self.make(geometry, init_scale=1e-4) for geometry in StructuralKGE.GEOMETRIES]
        scores = [model(*self.ids()).detach() for model in models]
        for score in scores[1:]:
            torch.testing.assert_close(score, scores[0], rtol=0.005, atol=1e-10)
        model = self.make("hyperbolic")
        h, r, t = self.ids()
        hp, rp, tp = (poincare_exp0(model._bound(table(ids))) for table, ids in
                      ((model.entity_embeddings, h), (model.relation_embeddings, r), (model.entity_embeddings, t)))
        expected = -poincare_half_distance(mobius_add(rp, hp), tp)
        torch.testing.assert_close(model(h, r, t), expected)

    def test_requested_linear_gregory_exactness_and_metadata(self):
        from sl_manifold.core import symmetric_distance
        from unittest.mock import patch
        for radius in (1.5,2.):
            model=self.make('sl8',coordinate_radius=radius)
            h,r,t=self.ids();hm=model._map(model.entity_embeddings(h));rm=model._map(model.relation_embeddings(r));tm=model._map(model.entity_embeddings(t))
            expected=model.score_offset-model.score_scale*symmetric_distance(rm@hm,tm,terms=12,jitter=1e-7,trace_project=True)
            with patch('structural_kge.models.checked_symmetric_log_distance',side_effect=AssertionError('GL hotpath')):
                torch.testing.assert_close(model(h,r,t),expected,rtol=0,atol=0)
            diagnostic=model.diagnostics(h,r,t)
            self.assertEqual(diagnostic['log_backend'],'gregory12')
            self.assertEqual(diagnostic['model_contract']['distance_power'],1)
            self.assertFalse(diagnostic['model_contract']['entity_bias'])
            self.assertTrue(diagnostic['principal_log']['scipy_reference_used'])
            self.assertTrue(diagnostic['sampled_health_passed'])

    def test_quadrature_scipy_reference_nonnormal_beyond_cayley_norm(self):
        matrix = torch.eye(8, dtype=torch.double)
        matrix[0, 1] = 4.0  # Valid principal log; Cayley spectral norm is 2 > 1.
        for order in (16, 32):
            logarithm = matrix_log_quadrature(matrix, order=order)
            reference = torch.from_numpy(logm(matrix.numpy())).double()
            torch.testing.assert_close(logarithm, reference, rtol=1e-11, atol=1e-11)
        diagnostic = log_pair_diagnostics(torch.eye(8, dtype=torch.double), matrix, scipy_reference=True)
        self.assertTrue(diagnostic["passed"])

    def test_quadrature_random_sl_reference_and_gradcheck(self):
        torch.manual_seed(8)
        algebra = torch.randn(2, 3, 3, dtype=torch.double) * 0.2
        algebra -= torch.diag_embed(algebra.diagonal(dim1=-2, dim2=-1).mean(-1).expand(3, 2).T)
        matrix = torch.matrix_exp(algebra).requires_grad_()
        for order in (16, 32):
            calculated = matrix_log_quadrature(matrix, order=order)
            reference = torch.tensor(np.stack([logm(m) for m in matrix.detach().numpy()]))
            torch.testing.assert_close(calculated, reference, rtol=1e-10, atol=1e-10)
        self.assertTrue(torch.autograd.gradcheck(lambda x: matrix_log_quadrature(x, order=16),
                                               (matrix[:1],), eps=1e-6, atol=1e-5, rtol=1e-4))

    def test_negative_axis_is_rejected_without_cayley_rule(self):
        matrix = torch.diag(torch.tensor([-1.0, -1.0, 1.0, 1.0], dtype=torch.double))
        diagnostic = log_pair_diagnostics(torch.eye(4, dtype=torch.double), matrix, raise_on_failure=False)
        self.assertFalse(diagnostic["passed"])
        self.assertEqual(diagnostic["principal_branch_cut_eigenvalue_count"], 2)
        with self.assertRaises(FloatingPointError):
            log_pair_diagnostics(torch.eye(4, dtype=torch.double), matrix)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
