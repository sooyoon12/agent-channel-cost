from copy import deepcopy
from decimal import Decimal
import tempfile
import unittest

from src.analysis.metrics import (
    aggregate_cost_transfer,
    bootstrap_cdr,
    build_cost_transfer_observations,
    displacement_decomposition,
    mcnemar_exact,
    summarize_conditions,
)
from src.analysis.reporting import write_analysis_outputs


def record(task_id, pair_id, condition, repetition, model_cost, shadow_cost=None, success=True):
    return {
        "task_id": task_id,
        "pair_id": pair_id,
        "language": task_id.rsplit("-", 1)[1],
        "condition": condition,
        "repetition": repetition,
        "success": success,
        "turn_count": 2,
        "tool_call_count": 1,
        "realized_ratio": 1 if condition == "B0" else 0.4,
        "model_cost_usd": {"list_price": model_cost, "billed": model_cost},
        "compressor_cost_usd": {"list_price": 0, "billed": 0},
        "shadow": {"path_a_list_cost": shadow_cost or model_cost, "valid": True},
        "tokens": {
            "local_i_input": 40,
            "local_t_input": 60,
            "protocol_residual_input": 4,
            "provider_input": 104,
            "assistant_natural_output": 8,
            "assistant_tool_call_output": 2,
            "provider_reasoning_output": 0,
            "provider_total_output": 10,
        },
        "events": [],
    }


def fixture_records():
    values = []
    for pair in ("p1", "p2"):
        for language in ("ko", "en"):
            task_id = f"{pair}-{language}"
            for repetition in (1, 2, 3):
                values.append(record(task_id, pair, "B0", repetition, 10))
            values.append(record(task_id, pair, "I-H", 1, 7, 11))
            values.append(record(task_id, pair, "T-H", 1, 8, 10))
            values.append(record(task_id, pair, "O", 1, 9, 9))
    return values


class AnalysisTests(unittest.TestCase):
    def test_cdr_uses_ratio_of_sums_and_baseline_mean(self):
        observations = build_cost_transfer_observations(fixture_records(), "I-H")
        aggregate = aggregate_cost_transfer(observations)
        self.assertEqual(aggregate.gross_saving, Decimal("16"))
        self.assertEqual(aggregate.behavioral_displacement, Decimal("4"))
        self.assertEqual(aggregate.cdr, Decimal("0.25"))
        self.assertEqual(aggregate.net_saving, Decimal("12"))

    def test_bootstrap_resamples_pair_clusters(self):
        observations = build_cost_transfer_observations(fixture_records(), "I-H")
        interval = bootstrap_cdr(observations, replicates=100, seed=1)
        self.assertEqual(interval.estimate, 0.25)
        self.assertEqual(interval.valid_replicates, 100)
        self.assertEqual((interval.lower, interval.upper), (0.25, 0.25))

    def test_mcnemar_uses_majority_baseline(self):
        values = fixture_records()
        changed = deepcopy(values)
        next(value for value in changed if value["task_id"] == "p1-ko" and value["condition"] == "I-H")["success"] = False
        result = mcnemar_exact(changed, "I-H")
        self.assertEqual(result.baseline_only_success, 1)
        self.assertEqual(result.condition_only_success, 0)
        self.assertEqual(result.exact_two_sided_p, 1.0)

    def test_summary_and_files_cover_o_without_cdr(self):
        values = fixture_records()
        summary = summarize_conditions(values)
        self.assertEqual(set(summary), {"B0", "I-H", "T-H", "O"})
        with tempfile.TemporaryDirectory() as directory:
            analysis = write_analysis_outputs(values, directory, bootstrap_replicates=20)
            self.assertNotIn("O", analysis["cost_transfer"])
            from pathlib import Path

            self.assertTrue((Path(directory) / "summary.json").exists())
            self.assertTrue((Path(directory) / "condition_summary.csv").exists())

    def test_missing_baseline_repetition_is_rejected(self):
        values = fixture_records()
        values.pop(0)
        with self.assertRaises(ValueError):
            build_cost_transfer_observations(values, "I-H")

    def test_displacement_decomposition_reconciles_unallocated_cost(self):
        result = displacement_decomposition(
            fixture_records(),
            "I-H",
            uncached_input_price_per_million=2,
            output_price_per_million=8,
        )
        self.assertEqual(result["observation_count"], 4)
        self.assertAlmostEqual(result["behavioral_displacement_usd"], 4.0)
        self.assertAlmostEqual(
            sum(result["model_cost_delta_usd"].values()), 4.0
        )


if __name__ == "__main__":
    unittest.main()
