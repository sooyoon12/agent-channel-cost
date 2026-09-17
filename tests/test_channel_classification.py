import json
from pathlib import Path
import unittest

from src.accounting.channels import (
    Channel,
    classify_input_item,
    classify_output_item,
    count_input_channels,
)
from src.accounting.invariants import assert_channel_partition


FIXTURE = Path(__file__).parents[1] / "data" / "fixtures" / "accounting_trajectory.json"


class LengthCounter:
    def count_text(self, text):
        return len(text)

    def count_json(self, value):
        return len(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


class ChannelClassificationTests(unittest.TestCase):
    def test_fixture_partitions_input_and_output(self):
        data = json.loads(FIXTURE.read_text(encoding="utf-8"))
        inputs = [classify_input_item(item) for item in data["input_items"]]
        outputs = [classify_output_item(item) for item in data["current_output_items"]]
        assert_channel_partition(len(inputs), len(data["input_items"]))
        self.assertEqual([x.channel for x in inputs], [Channel.INSTRUCTION_HISTORY] * 3 + [Channel.TOOL_OBSERVATION])
        self.assertTrue(all(x.channel is Channel.NEW_OUTPUT for x in outputs))

    def test_tool_role_and_provider_output_suffix_are_t(self):
        self.assertEqual(classify_input_item({"role": "tool"}).channel, Channel.TOOL_OBSERVATION)
        self.assertEqual(
            classify_input_item({"type": "future_tool_call_output"}).channel,
            Channel.TOOL_OBSERVATION,
        )

    def test_replayed_reasoning_and_tool_call_are_i(self):
        self.assertEqual(classify_input_item({"type": "reasoning"}).channel, Channel.INSTRUCTION_HISTORY)
        self.assertEqual(classify_input_item({"type": "function_call"}).channel, Channel.INSTRUCTION_HISTORY)
        self.assertEqual(classify_input_item({"type": "tool_schema"}).channel, Channel.INSTRUCTION_HISTORY)

    def test_unknown_shapes_fail_closed(self):
        with self.assertRaises(ValueError):
            classify_input_item({"type": "new_provider_magic"})
        with self.assertRaises(ValueError):
            classify_output_item({"type": "function_call_output"})

    def test_channel_token_totals_are_disjoint(self):
        data = json.loads(FIXTURE.read_text(encoding="utf-8"))
        counts = count_input_channels(data["input_items"], LengthCounter())
        self.assertGreater(counts.i_tokens, 0)
        self.assertGreater(counts.t_tokens, 0)
        self.assertEqual(counts.item_count, len(data["input_items"]))

    def test_protocol_wrappers_are_excluded_from_content_tokens(self):
        counter = LengthCounter()
        message = {
            "type": "message",
            "role": "user",
            "id": "hidden",
            "encrypted_content": "opaque-ciphertext",
            "content": "abc",
        }
        counts = count_input_channels([message], counter)
        self.assertEqual(counts.i_tokens, len("content") + len("abc"))

    def test_tool_output_matches_the_transmitted_json_string(self):
        counter = LengthCounter()
        item = {
            "type": "function_call_output",
            "call_id": "ignored",
            "output": {"alpha": 1, "beta": True},
        }
        counts = count_input_channels([item], counter)
        serialized = json.dumps(
            item["output"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        self.assertEqual(counts.t_tokens, len(serialized))


if __name__ == "__main__":
    unittest.main()
