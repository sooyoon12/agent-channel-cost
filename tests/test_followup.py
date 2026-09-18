from copy import deepcopy
import unittest

from src.analysis.followup import (
    analyze_contract_guard,
    analyze_ratio_sweep,
    clopper_pearson_upper,
    evaluation_method_comparison,
    holm_adjust,
)


def make_record(task_id, pair_id, condition, cost, success=True, ratio=0.4):
    return {
        "task_id": task_id,
        "pair_id": pair_id,
        "language": task_id.rsplit("-", 1)[1],
        "sensitivity_group": "neutral",
        "condition": condition,
        "repetition": 1,
        "target_ratio": ratio,
        "success": success,
        "analysis_valid": True,
        "turn_count": 2,
        "tool_call_count": 1,
        "realized_ratio": 1.0 if condition == "B0" else ratio,
        "model_cost_usd": {"list_price": cost, "billed": cost},
        "compressor_cost_usd": {"list_price": 0, "billed": 0},
        "shadow": {"path_a_list_cost": 10, "valid": True},
        "tokens": {
            "local_i_input": 40,
            "local_t_input": 60,
            "protocol_residual_input": 4,
            "provider_input": 104 if condition == "B0" else 70,
            "assistant_natural_output": 8,
            "assistant_tool_call_output": 2,
            "provider_reasoning_output": 0,
            "provider_total_output": 10,
        },
        "events": [],
    }


def baseline_records():
    values = []
    for pair in ("p1", "p2"):
        for language in ("ko", "en"):
            task_id = f"{pair}-{language}"
            for repetition in (1, 2, 3):
                value = make_record(task_id, pair, "B0", 10, ratio=1.0)
                value["repetition"] = repetition
                values.append(value)
    return values


class FollowupAnalysisTests(unittest.TestCase):
    def test_exact_upper_bound_for_zero_failures_at_eighty(self):
        self.assertLess(clopper_pearson_upper(0, 80), 0.05)
        self.assertGreater(clopper_pearson_upper(1, 80), 0.05)

    def test_holm_adjustment_preserves_original_order(self):
        self.assertEqual(holm_adjust([0.04, 0.01]), [0.04, 0.02])

    def test_evaluation_comparison_rejects_cheap_failure(self):
        records = baseline_records()
        for pair in ("p1", "p2"):
            for language in ("ko", "en"):
                task_id = f"{pair}-{language}"
                records.append(make_record(task_id, pair, "I-H", 2, success=False))
                records.append(make_record(task_id, pair, "O", 9, success=True))
        rows = {row["condition"]: row for row in evaluation_method_comparison(records)}
        self.assertTrue(rows["I-H"]["token_only_recommends"])
        self.assertFalse(rows["I-H"]["success_gated_eligible"])
        self.assertTrue(rows["O"]["success_gated_recommends"])

    def test_ratio_sweep_uses_invalid_run_as_safety_failure(self):
        sweep = []
        for condition in ("I-H", "T-H"):
            for ratio in (0.6, 0.8):
                for pair in ("p1", "p2"):
                    for language in ("ko", "en"):
                        sweep.append(
                            make_record(
                                f"{pair}-{language}", pair, condition, 8, ratio=ratio
                            )
                        )
        invalid = deepcopy(sweep[0])
        invalid["analysis_valid"] = False
        sweep[0] = invalid
        result = analyze_ratio_sweep(
            baseline_records(),
            sweep,
            success_margin=0.8,
            bootstrap_replicates=20,
        )
        cell = next(
            value
            for value in result["cells"]
            if value["condition"] == "I-H" and value["target_ratio"] == 0.6
        )
        self.assertEqual(cell["planned_run_count"], 4)
        self.assertEqual(cell["analysis_valid_count"], 3)
        self.assertEqual(cell["success_rate_planned"], 0.75)

    def test_contract_guard_can_pass_when_uniform_fails(self):
        baseline = []
        uniform = []
        guard = []
        for index in range(40):
            pair = f"p{index:02d}"
            for language in ("ko", "en"):
                task_id = f"{pair}-{language}"
                for repetition in (1, 2, 3):
                    value = make_record(task_id, pair, "B0", 10, ratio=1.0)
                    value["repetition"] = repetition
                    baseline.append(value)
                uniform.append(
                    make_record(task_id, pair, "T-H", 8, success=False, ratio=0.8)
                )
                guarded = make_record(
                    task_id, pair, "T-H", 9, success=True, ratio=0.8
                )
                guarded["events"] = [
                    {"event_type": "compression", "algorithm": "contract_guard"}
                ]
                guard.append(guarded)
        result = analyze_contract_guard(
            baseline, uniform, guard, bootstrap_replicates=20
        )
        self.assertTrue(result["contract_guard_recommended"])
        self.assertEqual(result["paired_success_test"]["first_only_success"], 80)


if __name__ == "__main__":
    unittest.main()
