from collections import Counter
import unittest

from src.environment.tools import DeterministicToolEnvironment
from src.tasks.generator import generate_preview, generate_task_set, task_set_checksum


class TaskPairTests(unittest.TestCase):
    def test_full_generator_has_frozen_shape(self):
        tasks = generate_task_set()
        self.assertEqual(len(tasks), 40)
        counts = Counter((task.task_family, task.sensitivity_group) for task in tasks)
        self.assertEqual(set(counts.values()), {10})
        self.assertEqual(len({task.pair_id for task in tasks}), 40)

    def test_generation_and_checksums_are_deterministic(self):
        first = generate_task_set()
        second = generate_task_set()
        self.assertEqual(first, second)
        self.assertEqual(task_set_checksum(first), task_set_checksum(second))
        self.assertTrue(all(task.checksum() == twin.checksum() for task, twin in zip(first, second)))

    def test_surfaces_share_one_semantic_structure(self):
        for task in generate_task_set():
            self.assertTrue(task.ko_surface)
            self.assertTrue(task.en_surface)
            self.assertNotEqual(task.ko_surface, task.en_surface)
            for call in task.expected_tool_calls:
                self.assertIn(call["name"], {"search_events", "set_attendance", "delete_event", "read_records", "submit_evidence"})

    def test_long_observation_has_exactly_one_matching_record(self):
        for task in generate_task_set():
            if task.task_family != "long_observation":
                continue
            semantic = task.semantic_form
            matches = []
            for record in task.hidden_state["records"]:
                matches_base = (
                    record["region"] == semantic["region"] and record["status"] == semantic["status"]
                )
                matches_boundary = (
                    "minimum_amount" not in semantic
                    or (
                        record["amount"] >= semantic["minimum_amount"]
                        and record["date"] >= semantic["on_or_after"]
                    )
                )
                if matches_base and matches_boundary:
                    matches.append(record)
            self.assertEqual(len(matches), 1, task.pair_id)
            self.assertEqual(matches[0]["record_id"], task.expected_tool_calls[1]["arguments"]["record_id"])

    def test_every_expected_path_executes_successfully(self):
        for task in generate_task_set():
            env = DeterministicToolEnvironment(task.hidden_state)
            for call in task.expected_tool_calls:
                result = env.execute(call["name"], call["arguments"])
                self.assertIsNone(result.error_code, task.pair_id)

    def test_preview_contains_three_pairs_per_family(self):
        preview = generate_preview()
        counts = Counter(task.task_family for task in preview)
        self.assertEqual(counts, {"precise_argument": 3, "long_observation": 3})


if __name__ == "__main__":
    unittest.main()
