import unittest

from src.analysis.v2_main import _retention_stratum_summary, information_retention_report, summarize_v2_main
from src.experiments.v2 import V2_METHODS
from src.tasks.v2 import generate_v2_candidates, split_v2_candidates


class V2MainAnalysisTests(unittest.TestCase):
    def test_complete_paired_summary(self):
        _, held_out = split_v2_candidates(generate_v2_candidates(long_pairs=4, event_pairs=4))
        records = []
        ordinal = 0
        for task in held_out:
            for language in ("ko", "en"):
                for method in V2_METHODS:
                    ordinal += 1
                    success = not (method == "uniform" and ordinal % 3 == 0)
                    records.append(
                        {
                            "plan_ordinal": ordinal,
                            "pair_id": task.pair_id,
                            "task_checksum": task.checksum(),
                            "task_family": task.task_family,
                            "language": language,
                            "experiment_method": method,
                            "failure_type": None if success else "argument_error",
                            "success": success,
                            "analysis_valid": True,
                            "accounting_warning_count": 0,
                            "model_cost_usd": {"billed": 0.01 if method == "b0" else 0.008},
                            "compressor_cost_usd": {"billed": 0},
                            "tokens": {"provider_input": 100, "local_t_input": 50},
                            "realized_ratio": 1.0 if method == "b0" else 0.8,
                        }
                    )
        report = summarize_v2_main(records, held_out, bootstrap_replicates=100)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["terminal_runs"], len(held_out) * 2 * 5)
        self.assertIn("uniform", report["paired_comparisons_vs_b0"])
        self.assertEqual(
            set(report["multiplicity"]["family"]),
            {"uniform", "head_tail", "query_aware_filter", "oracle_contract"},
        )

    def test_information_retention_partitions_all_runs(self):
        class Counter:
            def count_text(self, text):
                return len(str(text).split())

            def count_json(self, value):
                if isinstance(value, str):
                    return max(1, len(value.split()))
                if isinstance(value, dict):
                    return len(value) + sum(self.count_json(item) for item in value.values())
                if isinstance(value, (list, tuple)):
                    return sum(self.count_json(item) for item in value)
                return 1

        _, held_out = split_v2_candidates(generate_v2_candidates(long_pairs=4, event_pairs=4))
        records = []
        for task in held_out:
            for language in ("ko", "en"):
                for method in ("b0", "uniform", "head_tail"):
                    records.append(
                        {
                            "pair_id": task.pair_id,
                            "language": language,
                            "experiment_method": method,
                            "failure_type": None,
                            "success": True,
                        }
                    )
        report = information_retention_report(records, held_out, Counter())
        for method in ("uniform", "head_tail"):
            retained = report[method]["strata"]["all:retained"]["runs"]
            removed = report[method]["strata"]["all:removed"]["runs"]
            self.assertEqual(retained + removed, len(held_out) * 2)
            self.assertIn("retained_distractor_strata", report[method])
            self.assertTrue(report[method]["retained_distractor_strata"])

        uniform_retained = report["uniform"]["strata"]["all:retained"]
        self.assertEqual(uniform_retained["failures"], 0)
        self.assertIsNotNone(
            uniform_retained["one_sided_95_upper_failure_rate_if_zero_failures"]
        )
    def test_zero_failure_upper_bound_uses_pairs_not_runs(self):      # ← 여기서부터 추가
        rows = [
            {"pair_id": f"p{i}", "language": lang, "success": True}
            for i in range(10) for lang in ("ko", "en")
        ]
        result = _retention_stratum_summary(rows, rows)
        self.assertAlmostEqual(
            result["one_sided_95_upper_failure_rate_if_zero_failures"],
            1 - 0.05 ** (1 / 10),
        )


if __name__ == "__main__":
    unittest.main()
