from copy import deepcopy
import tempfile
from pathlib import Path
import unittest

from src.analysis.v2 import development_baseline_report
from src.experiments.v2 import (
    build_v2_development_baseline_plan,
    build_v2_heldout_baseline_plan,
    build_v2_heldout_compression_plan,
    build_v2_heldout_main_plan,
)
from src.tasks.export_v2 import export_frozen_v2_dataset, verify_frozen_v2_dataset
from src.tasks.v2 import generate_v2_candidates, split_v2_candidates


def _record(task, language, success=True):
    return {
        "pair_id": task.pair_id,
        "task_checksum": task.checksum(),
        "language": language,
        "condition": "B0",
        "failure_type": None if success else "argument_error",
        "success": success,
        "analysis_valid": True,
        "accounting_warning_count": 0,
        "model_cost_usd": {"billed": 0.01},
        "compressor_cost_usd": {"billed": 0},
    }


class V2Week2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.development, cls.held_out = split_v2_candidates(generate_v2_candidates())

    def test_plans_cover_each_language_once(self):
        development = build_v2_development_baseline_plan(self.development)
        heldout_b0 = build_v2_heldout_baseline_plan(self.held_out)
        heldout_th = build_v2_heldout_compression_plan(self.held_out)
        self.assertEqual(len(development), 50)
        self.assertEqual(len(heldout_b0), 150)
        self.assertEqual(len(heldout_th), 150)
        self.assertEqual({entry.condition.value for entry in development}, {"B0"})
        self.assertEqual({entry.condition.value for entry in heldout_th}, {"T-H"})
        main = build_v2_heldout_main_plan(self.held_out)
        self.assertEqual(len(main), 750)
        self.assertEqual(
            {entry.method for entry in main},
            {"b0", "uniform", "head_tail", "query_aware_filter", "oracle_contract"},
        )

    def test_development_gate_blocks_perfect_ceiling(self):
        records = [
            _record(task, language)
            for task in self.development
            for language in ("ko", "en")
        ]
        report = development_baseline_report(records, self.development)
        self.assertEqual(report["status"], "blocked")
        self.assertIn("baseline_ceiling", report["reasons"])

    def test_development_gate_passes_one_non_success(self):
        records = [
            _record(task, language)
            for task in self.development
            for language in ("ko", "en")
        ]
        records[0] = {**records[0], "success": False, "failure_type": "argument_error"}
        report = development_baseline_report(records, self.development)
        self.assertEqual(report["status"], "pass")
        with tempfile.TemporaryDirectory() as directory:
            manifest = export_frozen_v2_dataset(
                directory, self.held_out, development_report=report
            )
            verified = verify_frozen_v2_dataset(directory, self.held_out)
            self.assertEqual(manifest["semantic_set_checksum"], verified["semantic_set_checksum"])

    def test_freeze_rejects_blocked_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "did not pass"):
                export_frozen_v2_dataset(
                    directory,
                    self.held_out,
                    development_report={"status": "blocked"},
                )


if __name__ == "__main__":
    unittest.main()
