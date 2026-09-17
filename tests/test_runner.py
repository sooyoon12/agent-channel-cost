from collections import Counter
from decimal import Decimal
import unittest

from src.agent.runtime import ExperimentCondition
from src.experiments.runner import (
    BudgetGuard,
    build_main_plan,
    build_smoke_plan,
    canonical_plan_attempts,
    plan_checksum,
    provider_error_is_retryable,
)
from src.tasks.generator import generate_task_set


class RunnerTests(unittest.TestCase):
    def test_main_plan_has_exact_frozen_counts(self):
        plan = build_main_plan(generate_task_set())
        self.assertEqual(len(plan), 480)
        counts = Counter(entry.condition for entry in plan)
        self.assertEqual(counts[ExperimentCondition.BASELINE], 240)
        self.assertEqual(counts[ExperimentCondition.INSTRUCTION_HISTORY], 80)
        self.assertEqual(counts[ExperimentCondition.TOOL_OBSERVATION], 80)
        self.assertEqual(counts[ExperimentCondition.CONCISE_OUTPUT], 80)
        self.assertEqual([entry.ordinal for entry in plan], list(range(1, 481)))

    def test_plans_are_deterministic_and_smoke_is_capped(self):
        tasks = generate_task_set()
        first = build_main_plan(tasks)
        second = build_main_plan(tasks)
        self.assertEqual(plan_checksum(first), plan_checksum(second))
        smoke = build_smoke_plan(tasks)
        self.assertEqual(len(smoke), 20)
        self.assertEqual(len({entry.pair_id for entry in smoke}), 4)
        self.assertEqual({entry.language for entry in smoke}, {"ko", "en"})
        smoke_counts = Counter(entry.condition for entry in smoke)
        self.assertEqual(smoke_counts[ExperimentCondition.BASELINE], 8)
        self.assertEqual(smoke_counts[ExperimentCondition.INSTRUCTION_HISTORY], 4)
        self.assertEqual(smoke_counts[ExperimentCondition.TOOL_OBSERVATION], 4)
        self.assertEqual(smoke_counts[ExperimentCondition.CONCISE_OUTPUT], 4)
        with self.assertRaises(ValueError):
            build_smoke_plan(tasks, max_runs=21)

    def test_budget_warning_and_hard_stop(self):
        guard = BudgetGuard()
        normal = guard.check(Decimal("10"), Decimal("20"))
        self.assertTrue(normal.allowed)
        self.assertFalse(normal.warning)
        warning = guard.check(Decimal("69"), Decimal("2"))
        self.assertTrue(warning.allowed)
        self.assertTrue(warning.warning)
        stopped = guard.check(Decimal("99"), Decimal("2"))
        self.assertFalse(stopped.allowed)
        self.assertIsNotNone(stopped.reason)

    def test_only_explicitly_retryable_provider_errors_are_retried(self):
        retryable = {"events": [{"event_type": "provider_error", "retryable": True}]}
        permanent = {"events": [{"event_type": "provider_error", "retryable": False}]}
        legacy = {"events": [{"event_type": "provider_error"}]}
        self.assertTrue(provider_error_is_retryable(retryable))
        self.assertFalse(provider_error_is_retryable(permanent))
        self.assertFalse(provider_error_is_retryable(legacy))

    def test_canonical_attempts_retry_only_provider_errors(self):
        records = [
            {"plan_ordinal": 1, "failure_type": "provider_error"},
            {"plan_ordinal": 1, "failure_type": None},
            {"plan_ordinal": 2, "failure_type": "accounting_invalid"},
            {"plan_ordinal": 2, "failure_type": None},
        ]
        selected = canonical_plan_attempts(records)
        self.assertEqual(selected, tuple(records[:3]))


if __name__ == "__main__":
    unittest.main()
