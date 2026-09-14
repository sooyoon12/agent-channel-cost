from decimal import Decimal
import unittest

from src.accounting.pricing import PriceTier, PricingSchedule, calculate_model_cost
from src.accounting.provider_usage import ProviderUsage


class PricingTests(unittest.TestCase):
    def setUp(self):
        self.schedule = PricingSchedule(
            standard=PriceTier(Decimal("2"), Decimal("0.5"), Decimal("8"))
        )

    def test_billed_and_list_price_are_separate(self):
        usage = ProviderUsage(1_000_000, 250_000, 100_000, 20_000, 1_100_000)
        cost = calculate_model_cost(usage, self.schedule)
        self.assertEqual(cost.billed.uncached_input, Decimal("1.5"))
        self.assertEqual(cost.billed.cached_input, Decimal("0.125"))
        self.assertEqual(cost.billed.total, Decimal("2.425"))
        self.assertEqual(cost.list_price.total, Decimal("2.8"))

    def test_reasoning_is_not_double_charged(self):
        tier = PriceTier(Decimal("1"), Decimal("0.1"), Decimal("4"), Decimal("6"))
        usage = ProviderUsage(0, 0, 100, 25, 100)
        cost = calculate_model_cost(usage, PricingSchedule(tier))
        expected = (Decimal(75) * Decimal(4) + Decimal(25) * Decimal(6)) / Decimal(1_000_000)
        self.assertEqual(cost.billed.total, expected)

    def test_missing_provider_details_cannot_be_priced_as_zero(self):
        usage = ProviderUsage(10, None, 2, None, 12)
        with self.assertRaises(ValueError):
            calculate_model_cost(usage, self.schedule)

    def test_missing_reasoning_subcount_still_prices_total_output(self):
        usage = ProviderUsage(10, 0, 2, None, 12)
        cost = calculate_model_cost(usage, self.schedule)
        expected = (Decimal(10) * Decimal(2) + Decimal(2) * Decimal(8)) / Decimal(1_000_000)
        self.assertEqual(cost.billed.total, expected)


if __name__ == "__main__":
    unittest.main()
