from copy import deepcopy
import unittest

from src.analysis.v2_followup import summarize_v2_followup
from src.tasks.v2 import generate_v2_candidates


class V2FollowupAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = generate_v2_candidates()[0]

    def record(self, *, language, method, ratio, ordinal, success=True, failure=None):
        return {
            "plan_ordinal": ordinal,
            "pair_id": self.task.pair_id,
            "language": language,
            "experiment_method": method,
            "target_ratio": ratio,
            "failure_type": failure,
            "analysis_valid": failure is None,
            "success": success,
            "task_checksum": self.task.checksum(),
            "realized_ratio": 1.0 if method.startswith("b0") else ratio,
            "model_cost_usd": {"list_price": 0.02, "billed": 0.02},
            "compressor_cost_usd": {"list_price": 0.0, "billed": 0.0},
            "tokens": {
                "local_i_input": 100,
                "local_t_input": 200,
                "provider_input": 300,
                "provider_total_output": 20,
            },
        }

    def fixture(self):
        historical = [
            self.record(language=language, method="b0", ratio=0.8, ordinal=index)
            for index, language in enumerate(("ko", "en"), 1)
        ]
        followup = [
            self.record(language="ko", method="b0_repeat", ratio=0.8, ordinal=1),
            self.record(language="en", method="b0_repeat", ratio=0.8, ordinal=2),
            self.record(
                language="ko", method="uniform", ratio=0.6, ordinal=3,
                success=False, failure="provider_error",
            ),
            self.record(
                language="ko", method="uniform", ratio=0.6, ordinal=3,
                success=False, failure="provider_error",
            ),
            self.record(language="en", method="uniform", ratio=0.6, ordinal=4),
        ]
        return historical, followup

    def test_final_provider_error_is_one_planned_failure(self):
        historical, followup = self.fixture()
        report = summarize_v2_followup(
            historical,
            followup,
            (self.task,),
            method_ratios=(("uniform", 0.6),),
            bootstrap_replicates=10,
        )
        self.assertEqual(report["completeness"]["terminal_followup_rows"], 4)
        self.assertEqual(report["cells"]["uniform@0.6"]["runs"], 2)
        self.assertEqual(report["cells"]["uniform@0.6"]["successes"], 1)
        self.assertEqual(report["cells"]["uniform@0.6"]["analysis_invalid_rows"], 1)

    def test_missing_planned_row_fails_before_analysis(self):
        historical, followup = self.fixture()
        incomplete = [row for row in followup if row["plan_ordinal"] != 4]
        with self.assertRaisesRegex(ValueError, "completeness validation failed"):
            summarize_v2_followup(
                historical,
                incomplete,
                (self.task,),
                method_ratios=(("uniform", 0.6),),
                bootstrap_replicates=10,
            )


if __name__ == "__main__":
    unittest.main()
