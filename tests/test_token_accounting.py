import unittest

from src.accounting.invariants import check_input_residual
from src.accounting.provider_usage import parse_provider_usage


class TokenAccountingTests(unittest.TestCase):
    def test_responses_usage_preserves_null_reasoning(self):
        usage = parse_provider_usage(
            {
                "input_tokens": 105,
                "input_tokens_details": {"cached_tokens": 20},
                "output_tokens": 11,
                "total_tokens": 116,
            }
        )
        self.assertEqual(usage.uncached_input_tokens, 85)
        self.assertIsNone(usage.reasoning_output_tokens)

    def test_chat_usage_is_supported(self):
        usage = parse_provider_usage(
            {
                "prompt_tokens": 50,
                "prompt_tokens_details": {"cached_tokens": 10},
                "completion_tokens": 8,
                "completion_tokens_details": {"reasoning_tokens": 3},
            }
        )
        self.assertEqual((usage.input_tokens, usage.output_tokens), (50, 8))
        self.assertEqual(usage.reasoning_output_tokens, 3)

    def test_residual_identity_and_bounds(self):
        valid = check_input_residual(60, 35, 100, absolute_tolerance=8, fractional_tolerance=0.01)
        self.assertEqual(valid.residual_tokens, 5)
        self.assertTrue(valid.valid)
        too_large = check_input_residual(50, 20, 100, absolute_tolerance=8, fractional_tolerance=0.01)
        self.assertFalse(too_large.valid)
        negative = check_input_residual(80, 30, 100, absolute_tolerance=8, fractional_tolerance=0.01)
        self.assertFalse(negative.valid)
        self.assertIn("negative", negative.reason)

    def test_missing_is_not_zero(self):
        usage = parse_provider_usage({"input_tokens": 10, "output_tokens": 2})
        self.assertIsNone(usage.cached_input_tokens)
        self.assertIsNone(usage.reasoning_output_tokens)


if __name__ == "__main__":
    unittest.main()

