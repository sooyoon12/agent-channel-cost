from decimal import Decimal
import unittest

from src.accounting.pricing import PriceTier, PricingSchedule
from src.accounting.shadow import (
    ReplayShadowRequest,
    ShadowContent,
    aggregate_cdr,
    behavioral_displacement,
    compressed_replay_list_cost,
    gross_saving,
    validate_retokenized_shadow_paths,
)


class WordCounter:
    """Deterministic test tokenizer; production uses the recorded adapter."""

    def count_json(self, value):
        if isinstance(value, dict):
            return sum(self.count_json(v) for v in value.values())
        return len(str(value).split())


class ShadowPathTests(unittest.TestCase):
    def setUp(self):
        self.schedule = PricingSchedule(
            standard=PriceTier(Decimal("2"), Decimal("0.5"), Decimal("8"))
        )

    def test_reconstructed_path_matches_independent_delta_path(self):
        shared = ShadowContent(
            compressed={"role": "system", "content": "use tools"},
            original={"role": "system", "content": "use tools"},
        )
        compressed = ShadowContent(
            compressed={"role": "tool", "content": "alpha result"},
            original={"role": "tool", "content": "alpha beta gamma detailed result"},
        )
        requests = [
            ReplayShadowRequest((shared, compressed), 3, 7, 2),
            ReplayShadowRequest((shared, compressed), 3, 5, 1),
        ]
        result = validate_retokenized_shadow_paths(
            requests, self.schedule, WordCounter(), tolerance=Decimal("1e-12")
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.difference, Decimal(0))
        compressed_cost = compressed_replay_list_cost(requests, self.schedule, WordCounter())
        self.assertGreater(gross_saving(result.path_a_list_cost, compressed_cost), Decimal(0))

    def test_long_context_tier_crossing_is_detected_as_mismatch(self):
        schedule = PricingSchedule(
            standard=PriceTier(Decimal("1"), Decimal("0.1"), Decimal("4")),
            long_context=PriceTier(Decimal("2"), Decimal("0.2"), Decimal("6")),
            long_context_threshold=100,
        )
        counter = WordCounter()
        request = ReplayShadowRequest(
            (
                ShadowContent(
                    compressed={"content": "one " * 79},
                    original={"content": "one " * 119},
                ),
            ),
            0,
            10,
            0,
        )
        result = validate_retokenized_shadow_paths([request], schedule, counter, tolerance=Decimal("0"))
        self.assertFalse(result.valid)

    def test_aggregate_cdr_uses_ratio_of_sums(self):
        displacements = [Decimal("2"), Decimal("1")]
        savings = [Decimal("4"), Decimal("2")]
        self.assertEqual(aggregate_cdr(savings, displacements), Decimal("0.5"))
        self.assertIsNone(aggregate_cdr([Decimal("0")], [Decimal("1")]))

    def test_compressor_cost_is_added_once(self):
        value = behavioral_displacement(Decimal("10"), Decimal("8"), Decimal("0.5"))
        self.assertEqual(value, Decimal("2.5"))


if __name__ == "__main__":
    unittest.main()
