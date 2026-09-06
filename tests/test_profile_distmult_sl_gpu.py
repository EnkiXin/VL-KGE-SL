"""No CUDA needed: profile safety/parser/provenance unit tests."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("profile_script", ROOT / "scripts/profile_distmult_sl_gpu.py")
PROFILE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROFILE)


class ProfileTests(unittest.TestCase):
    def argv(self):
        return ["--root", str(ROOT), "--output", "/unused", "--method", "sl8"]

    def test_bounded_defaults(self):
        args = PROFILE.parse_args(self.argv())
        self.assertEqual(args.validation_queries, 32)
        self.assertEqual(args.score_chunk, 1024)

    def test_invalid_work_bounds(self):
        for name, value in (("--score-chunk", "0"), ("--validation-queries", "0"),
                            ("--validation-queries", "3"), ("--validation-queries", "130")):
            with self.subTest(name=name, value=value), self.assertRaises(SystemExit):
                PROFILE.parse_args(self.argv() + [name, value])
        with self.assertRaises(SystemExit):
            PROFILE.parse_args(["--root", str(ROOT), "--output", "/unused", "--method", "baseline", "--compare-chunks"])

    def test_atomic_json_and_finite_contract(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            PROFILE.save(path, {"status": "complete", "value": 1.0})
            self.assertEqual(json.loads(path.read_text())["value"], 1.0)
            with self.assertRaises(ValueError):
                PROFILE.save(path, {"nonfinite": float("nan")})

    def test_zero_gate_comparison_checks_actual_geometry_without_mutation(self):
        import sys
        sys.path[:0] = [str(ROOT), str(ROOT / "upstream/vl-kge"),
                       str(ROOT / "vendor/sl-manifold-core/src"), str(ROOT.parent / "sl-manifold-core/src")]
        import torch
        from geometry.distmult_sl import DistMultSL
        torch.set_num_threads(1)
        torch.manual_seed(42)
        model = DistMultSL(num_entities=7, num_relations=3, embedding_dim=768)
        triples = torch.tensor([[0, 0, 2], [1, 1, 3], [2, 2, 4], [3, 0, 5]])
        model.calibrate(triples)
        parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
        rng = torch.random.get_rng_state().clone()
        with patch.object(torch.cuda, "synchronize", return_value=None):
            result = PROFILE.compare_score_chunks(model, triples[:2], torch)
        self.assertTrue(result["initial_gate_zero_score_parity_alone_is_vacuous"])
        self.assertLess(result["raw_discrepancy_max_absolute_difference"], 1e-5)
        self.assertTrue(result["virtual_nonzero_gate_all_candidate_orders_equal"])
        self.assertEqual(result["virtual_gate_for_nontrivial_comparison"], 0.5)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(parameter, parameters[name]), name)


if __name__ == "__main__":
    unittest.main()
