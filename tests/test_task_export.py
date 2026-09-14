import json
from pathlib import Path
import tempfile
import unittest

from src.tasks.export import export_frozen_dataset, verify_frozen_dataset


class TaskExportTests(unittest.TestCase):
    def test_export_has_40_pairs_and_80_instances(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = export_frozen_dataset(directory)
            self.assertEqual(manifest["semantic_pair_count"], 40)
            self.assertEqual(manifest["task_instance_count"], 80)
            self.assertEqual(len((Path(directory) / "generated_tasks_ko.jsonl").read_text().splitlines()), 40)
            self.assertEqual(len((Path(directory) / "generated_tasks_en.jsonl").read_text().splitlines()), 40)
            self.assertEqual(verify_frozen_dataset(directory), manifest)

    def test_export_refuses_to_overwrite_changed_frozen_data(self):
        with tempfile.TemporaryDirectory() as directory:
            export_frozen_dataset(directory)
            path = Path(directory) / "semantic_tasks.jsonl"
            path.write_text(path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                export_frozen_dataset(directory)
            with self.assertRaises(RuntimeError):
                verify_frozen_dataset(directory)

    def test_language_instances_keep_shared_hidden_state(self):
        with tempfile.TemporaryDirectory() as directory:
            export_frozen_dataset(directory)
            ko = json.loads((Path(directory) / "generated_tasks_ko.jsonl").read_text().splitlines()[0])
            en = json.loads((Path(directory) / "generated_tasks_en.jsonl").read_text().splitlines()[0])
            self.assertEqual(ko["pair_id"], en["pair_id"])
            self.assertEqual(ko["hidden_state"], en["hidden_state"])
            self.assertEqual(ko["expected_tool_calls"], en["expected_tool_calls"])
            self.assertNotEqual(ko["surface"], en["surface"])


if __name__ == "__main__":
    unittest.main()

