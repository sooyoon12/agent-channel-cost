import json
import re
import unittest

from src.compression.fixed_ratio import FixedRatioCompressor
from src.compression.identity import IdentityCompressor


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


class CompressionTests(unittest.TestCase):
    def setUp(self):
        self.counter = UnitCounter()

    def test_identity_preserves_content_and_reports_one(self):
        value = {"content": "하나 둘 셋"}
        result = IdentityCompressor().compress(value, target_ratio=0.4, tokenizer=self.counter)
        self.assertEqual(result.original, result.compressed)
        self.assertEqual(result.realized_ratio, 1.0)
        self.assertEqual(result.original_handle, result.compressed_handle)

    def test_fixed_ratio_is_deterministic_and_keeps_keys(self):
        value = {
            "dataset_id": "DS-1",
            "records": [
                {"record_id": f"R-{index}", "notes": "alpha beta gamma delta epsilon"}
                for index in range(30)
            ],
        }
        first = FixedRatioCompressor().compress(value, target_ratio=0.4, tokenizer=self.counter)
        second = FixedRatioCompressor().compress(value, target_ratio=0.4, tokenizer=self.counter)
        self.assertEqual(first.compressed, second.compressed)
        self.assertEqual(first.compressed_handle, second.compressed_handle)
        self.assertEqual(set(first.compressed), set(value))
        self.assertLessEqual(first.realized_ratio, 0.4)
        json.dumps(first.compressed)

    def test_uniform_sampling_includes_boundaries_when_multiple_items_fit(self):
        value = {"records": list(range(20))}
        result = FixedRatioCompressor().compress(value, target_ratio=0.5, tokenizer=self.counter)
        records = result.compressed["records"]
        self.assertEqual(records[0], 0)
        self.assertEqual(records[-1], 19)

    def test_invalid_ratio_is_rejected(self):
        with self.assertRaises(ValueError):
            FixedRatioCompressor().compress("text", target_ratio=0, tokenizer=self.counter)


if __name__ == "__main__":
    unittest.main()

