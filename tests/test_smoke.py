from copy import deepcopy
import unittest

from src.experiments.smoke import evaluate_smoke_run


def smoke_record(condition):
    ratio = 0.4 if condition == "T-H" else 1.0
    return {
        "condition": condition,
        "success": True,
        "realized_ratio": ratio,
        "model_cost_usd": {"billed": 0.01},
        "compressor_cost_usd": {"billed": 0},
        "tokens": {
            "local_i_input": 40,
            "local_t_input": 60,
            "protocol_residual_input": 4,
            "provider_input": 104,
        },
        "shadow": {"valid": True},
        "events": [{"event_type": "model_output", "provider_input_tokens": 104}],
    }


class SmokeTests(unittest.TestCase):
    def test_all_gates_pass_for_valid_fixture(self):
        records = [smoke_record("B0") for _ in range(3)] + [smoke_record("T-H")]
        report = evaluate_smoke_run(
            records, realized_ratio_absolute_tolerance=0.05, max_runs=4
        )
        self.assertTrue(report.passed)
        self.assertEqual(report.estimated_main_cost_usd, 4.8)

    def test_broken_accounting_and_baseline_fail(self):
        records = [smoke_record("B0") for _ in range(3)] + [smoke_record("T-H")]
        records = deepcopy(records)
        records[0]["tokens"]["provider_input"] = 999
        records[1]["success"] = False
        records[2]["success"] = False
        report = evaluate_smoke_run(
            records, realized_ratio_absolute_tolerance=0.05, max_runs=4
        )
        failed = {gate.name for gate in report.gates if not gate.passed}
        self.assertIn("channel_identity", failed)
        self.assertIn("baseline_success", failed)

    def test_provider_retry_rows_are_excluded_from_research_run_count(self):
        records = [smoke_record("B0") for _ in range(3)] + [smoke_record("T-H")]
        retry = deepcopy(smoke_record("B0"))
        retry["analysis_valid"] = False
        retry["success"] = False
        retry["tokens"]["provider_input"] = 999
        report = evaluate_smoke_run(
            [retry, *records], realized_ratio_absolute_tolerance=0.05, max_runs=4
        )
        self.assertEqual(report.run_count, 4)
        self.assertTrue(report.passed)

    def test_incomplete_valid_smoke_does_not_pass(self):
        records = [smoke_record("B0") for _ in range(3)] + [smoke_record("T-H")]
        records[0]["analysis_valid"] = False
        report = evaluate_smoke_run(
            records, realized_ratio_absolute_tolerance=0.05, max_runs=4
        )
        gate = next(value for value in report.gates if value.name == "run_count")
        self.assertFalse(gate.passed)


if __name__ == "__main__":
    unittest.main()
