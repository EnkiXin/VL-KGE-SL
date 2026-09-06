"""CPU checks for WN9 linear-distance matched geometry heads."""
import copy
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "upstream/vl-kge"), str(ROOT / "vendor/sl-manifold-core/src"), str(ROOT.parent / "sl-manifold-core/src")]
import torch
from geometry.wn9_geometry_v2 import WN9GeometryV2
from geometry.matrix_log import checked_symmetric_log_distance, log_pair_diagnostics
from geometry.gregory_diagnostics import gregory_pair_diagnostics
from sl_manifold.core import symmetric_distance


class WN9V2Tests(unittest.TestCase):
    def make(self, geometry, radius=1.5, **kwargs):
        torch.manual_seed(42)
        features = torch.randn(9, 768) * .5
        return WN9GeometryV2(geometry=geometry, chart_radius=radius, num_entities=9,
            num_relations=3, embedding_dim=768, visual_features=features,
            textual_features=features.clone(), visual_entity_to_index={i:i for i in range(9)},
            textual_entity_to_index={i:i for i in range(9)}, use_visual=True, use_textual=True, **kwargs)

    def ids(self):
        return torch.tensor([0,1,2,1]), torch.tensor([0,1,2,0]), torch.tensor([2,3,4,5])

    def test_equal_capacity_train_only_initialization(self):
        models = [self.make(g) for g in WN9GeometryV2.GEOMETRIES]
        self.assertEqual(len({sum(p.numel() for p in m.parameters()) for m in models}), 1)
        for m in models:
            with self.assertRaises(RuntimeError): m(*self.ids())
            rng = torch.random.get_rng_state().clone()
            result = m.initialize_geometry(torch.arange(6), target_norm=.5)
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            self.assertAlmostEqual(result['train_entity_bounded_norm_median'], .5, places=5)
            self.assertAlmostEqual(result['relation_bounded_norm_mean'], .5, places=5)
            self.assertEqual(result['training_entity_count'], 6)
        for m in models[1:]:
            for p,q in zip(models[0].parameters(),m.parameters()): self.assertTrue(torch.equal(p,q))

    def test_linear_forward_backward_and_health(self):
        for radius in (1.5, 2.):
            for geometry in WN9GeometryV2.GEOMETRIES:
                with self.subTest(radius=radius, geometry=geometry):
                    m = self.make(geometry, radius, score_chunk=2)
                    m.initialize_geometry(torch.arange(6))
                    scores = m(*self.ids()); scores.mean().backward()
                    self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad))
                    d = m.diagnostics(*self.ids(), scipy_reference=geometry=='sl8')
                    self.assertTrue(d['sampled_health_passed'])
                    self.assertEqual(d['model_contract']['distance_power'],1)
                    if geometry=='euclidean':
                        h,r,t=self.ids(); _,hv=m.get_geometry_representations(h); _,tv=m.get_geometry_representations(t)
                        expected=m.score_offset-m.logit_scale*(hv+m.get_relation_representations(r)-tv).norm(dim=-1)
                        torch.testing.assert_close(scores,expected)

    def test_zero_gradient_cache_checkpoint(self):
        for g in WN9GeometryV2.GEOMETRIES:
            m=self.make(g);m.initialize_geometry(torch.arange(6));n=self.make(g)
            n.load_state_dict(copy.deepcopy(m.state_dict()));torch.testing.assert_close(m(*self.ids()),n(*self.ids()))
            with torch.no_grad(): m.projection.weight.zero_();m.relation_embeddings.weight.zero_()
            m(*self.ids()).sum().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad))
            m.eval()
            with torch.no_grad(): m(*self.ids());self.assertIsNotNone(m._evaluation_cache)
            m.train();self.assertIsNone(m._evaluation_cache)

    def test_real_block_guard_not_cayley_bound(self):
        identity=torch.eye(8);valid=identity.clone();valid[0,1]=4.
        self.assertTrue(torch.isfinite(checked_symmetric_log_distance(identity,valid)))
        diagnostics = log_pair_diagnostics(identity, valid)
        self.assertTrue(diagnostics['passed'])
        self.assertAlmostEqual(diagnostics['max_cayley_spectral_norm'], 2., places=5)
        self.assertEqual(diagnostics['cayley_spectral_norm_ge_one_count'], 1)
        self.assertFalse(diagnostics['cayley_warning_is_fatal'])
        bad=torch.diag(torch.tensor([-1.,-1.,1.,1.,1.,1.,1.,1.]))
        with self.assertRaises(FloatingPointError):checked_symmetric_log_distance(identity,bad)

    def test_singular_cayley_diagnostic_is_json_safe_nonfatal_information(self):
        import json
        identity = torch.eye(8)
        bad = torch.diag(torch.tensor([-1., -1., 1., 1., 1., 1., 1., 1.]))
        diagnostics = log_pair_diagnostics(identity, bad, raise_on_failure=False)
        self.assertEqual(diagnostics['cayley_diagnostic_status'], 'unavailable')
        self.assertIsNone(diagnostics['max_cayley_spectral_norm'])
        self.assertEqual(diagnostics['cayley_unavailable_count'], 1)
        self.assertFalse(diagnostics['cayley_warning_is_fatal'])
        self.assertFalse(diagnostics['passed'])  # The actual principal branch is invalid.
        json.dumps(diagnostics, allow_nan=False)

    def test_default_gregory_exact_old_linear_kernel_and_no_hotpath_gl(self):
        from unittest.mock import patch
        m = self.make('sl8'); m.initialize_geometry(torch.arange(6))
        self.assertEqual(m.log_backend, 'gregory12')
        h,r,t = self.ids(); _,hm=m.get_geometry_representations(h); _,tm=m.get_geometry_representations(t)
        rm=m.get_relation_representations(r)
        expected=m.score_offset-m.logit_scale*symmetric_distance(rm@hm,tm,terms=12,jitter=1e-7,trace_project=True)
        with patch('geometry.wn9_geometry_v2.checked_symmetric_log_distance', side_effect=AssertionError('GL hot path called')):
            torch.testing.assert_close(m(h,r,t),expected,rtol=0,atol=0)
        d=m.diagnostics(h,r,t)
        self.assertEqual(d['principal_log']['backend'],'gregory12')
        self.assertTrue(d['principal_log']['scipy_reference_used'])
        self.assertEqual(len(d['principal_log']['per_pair_accuracy']),len(h))
        with self.assertRaises(ValueError): m.diagnostics()

    def test_gregory_nonfatal_cayley_and_accuracy_are_distinct(self):
        import json
        identity=torch.eye(8);valid=identity.clone();valid[0,1]=4.
        d=gregory_pair_diagnostics(identity,valid)
        self.assertGreater(d['max_cayley_spectral_norm'],1)
        self.assertTrue(d['passed'])
        self.assertTrue(d['reference_accuracy_passed'])
        self.assertFalse(d['cayley_warning_is_fatal'])
        large=torch.diag(torch.tensor([100.,.01,1.,1.,1.,1.,1.,1.]))
        d=gregory_pair_diagnostics(identity,large)
        self.assertTrue(d['passed'])
        self.assertFalse(d['reference_accuracy_passed'])
        self.assertFalse(d['reference_accuracy_failure_is_fatal'])
        json.dumps(d,allow_nan=False)

    def test_explicit_gauss_legendre_still_available(self):
        m=self.make('sl8',log_backend='gauss_legendre')
        m.initialize_geometry(torch.arange(6))
        m(*self.ids()).mean().backward()
        d=m.diagnostics(*self.ids(),scipy_reference=True)
        self.assertTrue(d['sampled_health_passed'])
        self.assertEqual(d['principal_log']['backend'],'gauss_legendre')


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
