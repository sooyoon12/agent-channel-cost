from collections import Counter
from dataclasses import replace
from copy import deepcopy
from decimal import Decimal
import unittest

from src.accounting.pricing import PriceTier, PricingSchedule
from src.agent.runtime import AgentRuntime, ExperimentCondition, RuntimeSettings
from src.compression.contract_aware import OracleContractCompressor
from src.compression.head_tail import HeadTailCompressor
from src.compression.query_aware import QueryAwareCompressor
from src.environment.tools import DeterministicToolEnvironment
from src.experiments.provider import ScriptedProvider
from src.tasks.generator import task_set_checksum
from src.tasks.v2 import audit_v2_candidates, generate_v2_candidates, split_v2_candidates


class WordCounter:
    def count_text(self, text):
        return len(text.split())

    def count_json(self, value):
        if isinstance(value, str):
            return self.count_text(value)
        if isinstance(value, dict):
            return len(value) + sum(self.count_json(item) for item in value.values())
        if isinstance(value, list):
            return sum(self.count_json(item) for item in value)
        return 1


class V2CandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = generate_v2_candidates()

    def test_candidate_shape_and_determinism(self):
        self.assertEqual(len(self.tasks), 100)
        self.assertEqual(
            Counter(task.task_family for task in self.tasks),
            {"long_observation": 80, "event_followup": 20},
        )
        self.assertEqual(
            task_set_checksum(self.tasks), task_set_checksum(generate_v2_candidates())
        )
        self.assertEqual(
            audit_v2_candidates(self.tasks), {"pairs": 100, "th_active_pairs": 100}
        )

    def test_development_and_held_out_candidates_do_not_overlap(self):
        development, held_out = split_v2_candidates(self.tasks)
        self.assertEqual(
            Counter(task.task_family for task in held_out),
            {"long_observation": 60, "event_followup": 15},
        )
        self.assertEqual(len({task.pair_id for task in development} & {task.pair_id for task in held_out}), 0)
        self.assertEqual(audit_v2_candidates(held_out)["th_active_pairs"], 75)

    def test_long_records_have_no_answer_label_in_notes_or_code_prefix(self):
        for task in self.tasks:
            if task.task_family != "long_observation":
                continue
            records = task.hidden_state["records"]
            self.assertGreater(len({record["notes"] for record in records}), 1)
            self.assertEqual(len({record["evidence_code"] for record in records}), len(records))
            self.assertTrue(all(code.startswith("EV-") for code in (record["evidence_code"] for record in records)))
            self.assertTrue(all("target" not in record["notes"].lower() for record in records))
            self.assertTrue(all("distractor" not in record["notes"].lower() for record in records))

    def test_audit_rejects_answer_bearing_note(self):
        task = next(task for task in self.tasks if task.task_family == "long_observation")
        state = deepcopy(task.hidden_state)
        target_id = task.expected_final_state["submission"]["record_id"]
        next(record for record in state["records"] if record["record_id"] == target_id)[
            "notes"
        ] = "target evidence"
        with self.assertRaisesRegex(ValueError, "answer-bearing notes"):
            audit_v2_candidates((replace(task, hidden_state=state),))

    def test_event_followup_needs_observed_id_and_contract_filter_keeps_it(self):
        task = next(task for task in self.tasks if task.task_family == "event_followup")
        environment = DeterministicToolEnvironment(task.hidden_state)
        first = task.expected_tool_calls[0]
        result = environment.execute(first["name"], first["arguments"]).result
        self.assertEqual(len(result["events"]), 8)
        selected = [event for event in result["events"] if event["room"] == "HALL-A"]
        self.assertEqual(len(selected), 1)
        event_id = selected[0]["event_id"]
        self.assertNotIn(event_id, task.ko_surface)
        self.assertNotIn(event_id, task.en_surface)
        compressed = OracleContractCompressor().compress(
            result, target_ratio=0.8, tokenizer=WordCounter()
        ).compressed
        selected_after = [event for event in compressed["events"] if event["room"] == "HALL-A"]
        self.assertEqual([event["event_id"] for event in selected_after], [event_id])
        self.assertTrue(all("notes" not in event for event in compressed["events"]))

    def test_event_target_positions_vary_without_id_format_hint(self):
        positions = set()
        for task in self.tasks:
            if task.task_family != "event_followup":
                continue
            environment = DeterministicToolEnvironment(task.hidden_state)
            first = task.expected_tool_calls[0]
            events = environment.execute(first["name"], first["arguments"]).result["events"]
            target_id = task.expected_tool_calls[1]["arguments"]["event_id"]
            positions.add(next(index for index, event in enumerate(events) if event["event_id"] == target_id))
            self.assertEqual({len(event["event_id"]) for event in events}, {len(target_id)})
        self.assertGreater(len(positions), 2)

    def test_sensitive_long_target_is_not_the_threshold_shortcut(self):
        for task in self.tasks:
            if task.task_family != "long_observation" or "minimum_amount" not in task.semantic_form:
                continue
            target_id = task.expected_final_state["submission"]["record_id"]
            target = next(
                record for record in task.hidden_state["records"] if record["record_id"] == target_id
            )
            self.assertNotEqual(target["amount"], task.semantic_form["minimum_amount"])
            self.assertNotEqual(target["date"], task.semantic_form["on_or_after"])

    def test_comparison_compressors_have_distinct_declared_roles(self):
        task = next(task for task in self.tasks if task.task_family == "long_observation")
        environment = DeterministicToolEnvironment(task.hidden_state)
        first = task.expected_tool_calls[0]
        result = environment.execute(first["name"], first["arguments"]).result
        counter = WordCounter()
        head_tail = HeadTailCompressor().compress(result, target_ratio=0.8, tokenizer=counter)
        query = QueryAwareCompressor(task.semantic_form).compress(
            result, target_ratio=0.8, tokenizer=counter
        )
        oracle = OracleContractCompressor().compress(
            result, target_ratio=0.8, tokenizer=counter
        )
        self.assertEqual(head_tail.algorithm, "head_tail")
        self.assertEqual(query.algorithm, "query_aware_filter")
        self.assertEqual(oracle.algorithm, "oracle_contract")
        self.assertEqual(len(query.compressed["records"]), 1)
        self.assertTrue(all("notes" not in record for record in oracle.compressed["records"]))

    def test_offline_runtime_orchestrates_both_active_families(self):
        # The scripted provider validates wiring only; it is not a success result.
        for family in ("long_observation", "event_followup"):
            task = next(task for task in self.tasks if task.task_family == family)
            counter = WordCounter()
            runtime = AgentRuntime(
                provider=ScriptedProvider(task.expected_tool_calls, counter),
                tokenizer=counter,
                pricing=PricingSchedule(PriceTier(Decimal("2"), Decimal("0.5"), Decimal("8"))),
                compressor=OracleContractCompressor(),
                settings=RuntimeSettings(model="offline-fixed-model", target_ratio=0.8),
                tokenizer_name="word-counter",
                tokenizer_version="1",
            )
            result = runtime.run(
                task, language="ko", condition=ExperimentCondition.TOOL_OBSERVATION, repetition=1
            )
            self.assertTrue(result.record["success"], family)
            self.assertTrue(result.record["shadow"]["valid"], family)
            self.assertLess(result.record["realized_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
