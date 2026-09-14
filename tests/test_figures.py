import tempfile
from pathlib import Path
import unittest

from src.analysis.figures import write_required_figures


class FigureTests(unittest.TestCase):
    def test_required_svgs_are_written_without_plotting_dependency(self):
        analysis = {
            "conditions": {
                "B0": {"mean_provider_input_tokens": 100, "mean_local_i_input_tokens": 40, "mean_local_t_input_tokens": 60, "mean_provider_output_tokens": 20, "mean_model_list_cost_usd": 1},
                "I-H": {"mean_provider_input_tokens": 70, "mean_local_i_input_tokens": 10, "mean_local_t_input_tokens": 60, "mean_provider_output_tokens": 20, "mean_model_list_cost_usd": 0.8},
                "T-H": {"mean_provider_input_tokens": 60, "mean_local_i_input_tokens": 40, "mean_local_t_input_tokens": 20, "mean_provider_output_tokens": 20, "mean_model_list_cost_usd": 0.7},
                "O": {"mean_provider_input_tokens": 100, "mean_local_i_input_tokens": 40, "mean_local_t_input_tokens": 60, "mean_provider_output_tokens": 8, "mean_model_list_cost_usd": 0.9},
            },
            "cost_transfer": {
                "I-H": {"aggregate": {"gross_saving": 1.0, "behavioral_displacement": 0.2}},
                "T-H": {"aggregate": {"gross_saving": 2.0, "behavioral_displacement": 0.5}},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            first, second = write_required_figures(analysis, directory)
            self.assertTrue(first.exists())
            self.assertTrue(second.exists())
            self.assertIn("<svg", first.read_text(encoding="utf-8"))
            self.assertIn("I-H G", second.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
