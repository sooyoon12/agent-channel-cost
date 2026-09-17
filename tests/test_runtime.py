from decimal import Decimal
import json
from pathlib import Path
import re
import tempfile
import unittest

from src.accounting.pricing import PriceTier, PricingSchedule
from src.accounting.provider_usage import ProviderUsage
from src.agent.runtime import AgentRuntime, ExperimentCondition, RuntimeSettings
from src.compression.fixed_ratio import FixedRatioCompressor
from src.experiments.logger import JsonlRunLogger, validate_run_record
from src.experiments.provider import ProviderResponse, ScriptedProvider
from src.tasks.generator import generate_task_set


class UnitCounter:
    def count_text(self, text):
        return len(re.findall(r"\S+", text))

    def count_json(self, value):
        if isinstance(value, str):
            return self.count_text(value)
        if isinstance(value, dict):
            return len(value) + sum(self.count_json(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(self.count_json(item) for item in value)
        return 1


class RuntimeTests(unittest.TestCase):
    def make_runtime(self, task):
        tokenizer = UnitCounter()
        provider = ScriptedProvider(task.expected_tool_calls, tokenizer, protocol_residual_tokens=4)
        pricing = PricingSchedule(
            PriceTier(Decimal("2"), Decimal("0.5"), Decimal("8"))
        )
        return AgentRuntime(
            provider=provider,
            tokenizer=tokenizer,
            pricing=pricing,
            compressor=FixedRatioCompressor(),
            settings=RuntimeSettings(model="offline-fixed-model"),
            tokenizer_name="unit-counter",
            tokenizer_version="1",
        )

    def test_offline_runtime_completes_all_conditions(self):
        task = next(value for value in generate_task_set() if value.pair_id == "long-011")
        for condition in ExperimentCondition:
            result = self.make_runtime(task).run(
                task, language="ko", condition=condition, repetition=1, run_id=f"run-{condition.value}"
            )
            validate_run_record(result.record)
            self.assertTrue(result.record["success"], condition)
            self.assertTrue(result.record["shadow"]["valid"])
            self.assertEqual(
                result.record["tokens"]["local_i_input"]
                + result.record["tokens"]["local_t_input"]
                + result.record["tokens"]["protocol_residual_input"],
                result.record["tokens"]["provider_input"],
            )

    def test_only_selected_channel_is_compressed(self):
        task = next(value for value in generate_task_set() if value.pair_id == "long-011")
        baseline = self.make_runtime(task).run(task, language="en", condition=ExperimentCondition.BASELINE, repetition=1)
        i_run = self.make_runtime(task).run(task, language="en", condition=ExperimentCondition.INSTRUCTION_HISTORY, repetition=1)
        t_run = self.make_runtime(task).run(task, language="en", condition=ExperimentCondition.TOOL_OBSERVATION, repetition=1)
        self.assertLess(i_run.record["tokens"]["local_i_input"], baseline.record["tokens"]["local_i_input"])
        self.assertEqual(i_run.record["tokens"]["local_t_input"], baseline.record["tokens"]["local_t_input"])
        self.assertLess(t_run.record["tokens"]["local_t_input"], baseline.record["tokens"]["local_t_input"])

    def test_jsonl_logger_appends_valid_records(self):
        task = generate_task_set()[0]
        result = self.make_runtime(task).run(task, language="ko", condition=ExperimentCondition.BASELINE, repetition=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw" / "runs.jsonl"
            logger = JsonlRunLogger(path)
            logger.append(result.record)
            logger.append(result.record)
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["run_id"], result.record["run_id"])

    def test_string_encoded_provider_arguments_are_parsed(self):
        task = generate_task_set()[0]
        tokenizer = UnitCounter()
        provider = ScriptedProvider(
            task.expected_tool_calls, tokenizer, arguments_as_json=True
        )
        runtime = AgentRuntime(
            provider=provider,
            tokenizer=tokenizer,
            pricing=PricingSchedule(PriceTier(Decimal("2"), Decimal("0.5"), Decimal("8"))),
            compressor=FixedRatioCompressor(),
            settings=RuntimeSettings(model="offline-fixed-model"),
            tokenizer_name="unit-counter",
            tokenizer_version="1",
        )
        result = runtime.run(
            task, language="ko", condition=ExperimentCondition.BASELINE, repetition=1
        )
        self.assertTrue(result.record["success"])

    def test_tool_call_on_last_allowed_turn_is_max_turn_failure(self):
        task = generate_task_set()[0]
        tokenizer = UnitCounter()
        calls = tuple(task.expected_tool_calls) * 3
        runtime = AgentRuntime(
            provider=ScriptedProvider(calls, tokenizer),
            tokenizer=tokenizer,
            pricing=PricingSchedule(PriceTier(Decimal("2"), Decimal("0.5"), Decimal("8"))),
            compressor=FixedRatioCompressor(),
            settings=RuntimeSettings(
                model="offline-fixed-model", max_turns=2, max_identical_calls=10
            ),
            tokenizer_name="unit-counter",
            tokenizer_version="1",
        )
        result = runtime.run(
            task, language="ko", condition=ExperimentCondition.BASELINE, repetition=1
        )
        self.assertFalse(result.record["success"])
        self.assertEqual(result.record["failure_type"], "max_turn/budget")

    def test_accounting_error_is_logged_as_an_invalid_record(self):
        class ImpossibleUsageProvider:
            name = "scripted-offline"

            def complete(self, request):
                return ProviderResponse(
                    response_id="bad-accounting",
                    output_items=(
                        {"type": "message", "role": "assistant", "content": "done"},
                    ),
                    usage=ProviderUsage(
                        input_tokens=0,
                        cached_input_tokens=0,
                        cache_write_input_tokens=0,
                        output_tokens=1,
                        reasoning_output_tokens=0,
                        total_tokens=1,
                    ),
                    latency_ms=1.0,
                )

        task = generate_task_set()[0]
        tokenizer = UnitCounter()
        runtime = AgentRuntime(
            provider=ImpossibleUsageProvider(),
            tokenizer=tokenizer,
            pricing=PricingSchedule(PriceTier(Decimal("2"), Decimal("0.5"), Decimal("8"))),
            compressor=FixedRatioCompressor(),
            settings=RuntimeSettings(model="offline-fixed-model"),
            tokenizer_name="unit-counter",
            tokenizer_version="1",
        )
        result = runtime.run(
            task, language="ko", condition=ExperimentCondition.BASELINE, repetition=1
        )
        self.assertEqual(result.record["failure_type"], "accounting_invalid")
        self.assertFalse(result.record["shadow"]["valid"])
        self.assertEqual(result.record["events"][-1]["event_type"], "accounting_error")


if __name__ == "__main__":
    unittest.main()
