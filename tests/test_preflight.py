import unittest

from src.experiments.preflight import require_live_ready, validate_live_configuration


def valid_configs():
    return {
        "model": {
            "provider": "openai",
            "model_id": "gpt-5.6-terra",
            "requested_model_label": "gpt-5.6-terra",
            "temperature": None,
            "reasoning_effort": "low",
            "tokenizer": {
                "exact_encoding": "o200k_base",
                "proxy_encoding": None,
                "version": "0.14.0",
            },
        },
        "pricing": {
            "unit_tokens": 1000000,
            "standard": {
                "uncached_input_per_unit": 1,
                "cached_input_per_unit": 0.1,
                "cache_write_input_per_unit": 1.25,
                "output_per_unit": 4,
            },
            "long_context": {
                "threshold_input_tokens": 272000,
                "uncached_input_per_unit": 2,
                "cached_input_per_unit": 0.2,
                "cache_write_input_per_unit": 2.5,
                "output_per_unit": 6,
            },
            "price_version": "2026-01-01",
            "source_url": "https://example.invalid/official-placeholder",
        },
        "experiment": {
            "residual_tolerance": {"absolute_tokens": 8, "fraction_of_provider_input": 0.01},
            "realized_ratio_absolute_tolerance": 0.05,
            "max_agent_turns": 6,
            "max_identical_tool_calls": 2,
            "max_request_input_tokens": 240000,
            "baseline_success_aggregation": "majority_vote",
            "bootstrap_resampling_unit": "pair_id",
            "bootstrap_replicates": 10000,
            "execution_order_seed": 20260914,
            "provider_error_retries": 1,
        },
        "compression": {"target_ratio": 0.4, "algorithm": "uniform_extract"},
    }


class PreflightTests(unittest.TestCase):
    def test_complete_configuration_is_ready(self):
        report = validate_live_configuration(**valid_configs())
        self.assertTrue(report.ready)
        self.assertEqual(report.issues, ())

    def test_missing_external_facts_fail_closed(self):
        configs = valid_configs()
        configs["model"]["model_id"] = None
        configs["pricing"]["standard"]["output_per_unit"] = None
        report = validate_live_configuration(**configs)
        self.assertFalse(report.ready)
        self.assertGreaterEqual(len(report.issues), 2)
        with self.assertRaises(RuntimeError):
            require_live_ready(**configs)


if __name__ == "__main__":
    unittest.main()
