from pathlib import Path
import tempfile
import unittest

from src.experiments.config import load_mapping_yaml


ROOT = Path(__file__).parents[1]


class ConfigTests(unittest.TestCase):
    def test_repository_configs_load_without_external_yaml_package(self):
        experiment = load_mapping_yaml(ROOT / "configs" / "experiment.yaml")
        model = load_mapping_yaml(ROOT / "configs" / "model.yaml")
        pricing = load_mapping_yaml(ROOT / "configs" / "pricing.yaml")
        self.assertEqual(experiment["target_ratio"], 0.4)
        self.assertEqual(experiment["max_agent_turns"], 6)
        self.assertIsNone(model["model_id"])
        self.assertIsNone(pricing["standard"]["output_per_unit"])

    def test_duplicate_keys_and_lists_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            duplicate = Path(directory) / "duplicate.yaml"
            duplicate.write_text("key: 1\nkey: 2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_mapping_yaml(duplicate)
            sequence = Path(directory) / "sequence.yaml"
            sequence.write_text("items:\n  - one\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_mapping_yaml(sequence)


if __name__ == "__main__":
    unittest.main()

