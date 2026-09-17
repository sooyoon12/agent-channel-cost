from pathlib import Path
import tempfile
import unittest

from src.experiments.dotenv import environment_or_dotenv, read_dotenv


class DotenvTests(unittest.TestCase):
    def test_unquoted_and_quoted_values_are_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                "# secrets\nOPENAI_API_KEY=sk-local\nQUOTED=\"hello world\"\n",
                encoding="utf-8",
            )
            self.assertEqual(
                read_dotenv(path),
                {"OPENAI_API_KEY": "sk-local", "QUOTED": "hello world"},
            )

    def test_process_environment_takes_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("OPENAI_API_KEY=sk-file\n", encoding="utf-8")
            value = environment_or_dotenv(
                "OPENAI_API_KEY",
                dotenv_path=path,
                environment={"OPENAI_API_KEY": "sk-process"},
            )
            self.assertEqual(value, "sk-process")

    def test_missing_or_empty_value_is_none(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("OPENAI_API_KEY=\n", encoding="utf-8")
            self.assertIsNone(
                environment_or_dotenv("OPENAI_API_KEY", dotenv_path=path, environment={})
            )

    def test_malformed_entry_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("OPENAI_API_KEY\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_dotenv(path)


if __name__ == "__main__":
    unittest.main()
