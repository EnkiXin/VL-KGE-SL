"""No GPU or network: HPO preparation is deterministic and test-blind."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("hpo_plan", ROOT / "scripts/plan_distmult_sl_hpo.py")
HPO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HPO)


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.spec = HPO.checked_spec(ROOT / "experiments/distmult-sl-hpo-v1.json")

    def test_deterministic_unique_shared_pool_and_factor_coverage(self):
        pool = HPO.candidate_pool(self.spec)
        self.assertEqual(pool, HPO.candidate_pool(self.spec))
        self.assertEqual(len(pool), 48)
        self.assertEqual(len({HPO.canonical(c["shared"]) for c in pool}), 48)
        for key, values in self.spec["shared_space"].items():
            self.assertEqual({c["shared"][key] for c in pool}, set(values))
        for key, values in self.spec["extension_space"].items():
            self.assertEqual({c["extension"][key] for c in pool}, set(values))
        self.assertTrue(all(len(c["extension_factor_change"]) <= 1 for c in pool))

    def test_matched_groups_interleaved_no_test(self):
        for stage, count in [("profile", 3), ("anchors", 18), ("coarse", 144)]:
            jobs = HPO.planned_jobs(self.spec, stage)
            self.assertEqual(len(jobs), count)
            for index in range(0, len(jobs), 3):
                a, b, c = jobs[index:index + 3]
                self.assertEqual([a["method"], b["method"], c["method"]], list(HPO.METHODS))
                self.assertEqual(a["author_overrides"], b["author_overrides"])
                self.assertEqual(a["author_overrides"], c["author_overrides"])
                self.assertEqual(b["extension_config"], c["extension_config"])
                self.assertEqual(a["extension_config"], {})
                self.assertTrue(all(j["validation_only"] for j in [a, b, c]))

    def test_preparation_does_not_authorize_or_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "prepared"
            manifest = HPO.write_plan(self.spec, out, "anchors")
            self.assertFalse(manifest["auto_launch"])
            self.assertIsNone(manifest["gpu_hours_authorized"])
            self.assertTrue(all(j["status"] == "not_started" for j in manifest["jobs"]))
            for job in manifest["jobs"]:
                self.assertTrue((out / job["extension_config_path"]).is_file())
            with self.assertRaises(FileExistsError):
                HPO.write_plan(self.spec, out, "anchors")

    def records(self):
        return [{"candidate_id": f"c{i:03d}", "status": "completed_validation_only",
                 "method": "sl8", "stage": "coarse", "seed": 42, "spec_sha256": "a" * 64,
                 "test_evaluated": False, "best_validation_mrr": 0.1 + i * 0.1,
                 "author_overrides": {"lr": 0.1, "batch_size": 512},
                 "validation_improvement_last_10": 0.2 if i == 1 else 0}
                for i in range(6)]

    def test_promotion_protects_anchor_slow_and_best(self):
        selected = HPO.promote_validation_records(self.records(), 3)
        self.assertEqual([r["candidate_id"] for r in selected], ["c000", "c001", "c005"])

    def test_promotion_rejects_test_failed_duplicate_and_nonfinite(self):
        for change in [{"final_test": {"mrr": .99}}, {"test_evaluated": True},
                       {"status": "failed"}, {"best_validation_mrr": float("nan")},
                       {"candidate_id": "c000"}]:
            records = self.records()
            records[1].update(change)
            with self.assertRaises(ValueError):
                HPO.promote_validation_records(records, 3)

    def test_small_pool_cannot_claim_full_search_coverage(self):
        with self.assertRaises(ValueError):
            HPO.candidate_pool(self.spec, 6)

    def test_promotion_rejects_mixed_methods_stages_seeds_specs(self):
        for change in [{"method": "baseline"}, {"stage": "refine"},
                       {"seed": 2026}, {"spec_sha256": "b" * 64}]:
            records = self.records()
            records[1].update(change)
            with self.assertRaisesRegex(ValueError, "mix"):
                HPO.promote_validation_records(records, 3)

    def test_slow_reservation_does_not_collapse_into_anchor(self):
        records = self.records()
        records[0]["validation_improvement_last_10"] = .8
        selected = HPO.promote_validation_records(records, 3)
        self.assertEqual([r["candidate_id"] for r in selected], ["c000", "c001", "c005"])

    def test_final_test_stage_cannot_be_materialized_implicitly(self):
        with self.assertRaises(ValueError):
            HPO.planned_jobs(self.spec, "locked_final")


if __name__ == "__main__":
    unittest.main()
