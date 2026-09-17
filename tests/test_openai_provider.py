from io import BytesIO
import unittest
from urllib.error import HTTPError
from unittest.mock import patch

from src.experiments.openai_provider import (
    OpenAIResponsesProvider,
    ProviderTransportError,
    UrllibJsonTransport,
    build_responses_payload,
)
from src.experiments.provider import ProviderRequest


class FakeTransport:
    def __init__(self):
        self.path = None
        self.payload = None

    def post(self, path, payload):
        self.path = path
        self.payload = payload
        return {
            "id": "resp-test",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "lookup",
                    "arguments": "{\"id\":\"A\"}",
                }
            ],
            "usage": {
                "input_tokens": 100,
                "input_tokens_details": {"cached_tokens": 20, "cache_write_tokens": 5},
                "output_tokens": 10,
                "output_tokens_details": {"reasoning_tokens": 3},
                "total_tokens": 110,
            },
        }


def request(style="normal", temperature=None):
    return ProviderRequest(
        items=(
            {"role": "user", "content": "hello"},
            {
                "type": "tool_schema",
                "name": "lookup",
                "description": "lookup",
                "parameters": {"type": "object", "properties": {}},
            },
            {"type": "function_call_output", "call_id": "old", "output": {"ok": True}},
        ),
        model="fixed-model",
        temperature=temperature,
        reasoning_effort="low",
        max_output_tokens=100,
        output_style=style,
    )


class OpenAIProviderTests(unittest.TestCase):
    def test_payload_separates_tools_and_serializes_tool_output(self):
        payload = build_responses_payload(request("concise"))
        self.assertEqual(payload["model"], "fixed-model")
        self.assertEqual(len(payload["tools"]), 1)
        self.assertEqual(len(payload["input"]), 2)
        self.assertIsInstance(payload["input"][1]["output"], str)
        self.assertEqual(payload["text"], {"verbosity": "low"})
        self.assertNotIn("temperature", payload)

    def test_temperature_is_sent_only_when_explicitly_supported(self):
        payload = build_responses_payload(request(temperature=0))
        self.assertEqual(payload["temperature"], 0)

    def test_adapter_normalizes_usage_without_network(self):
        transport = FakeTransport()
        response = OpenAIResponsesProvider(transport).complete(request())
        self.assertEqual(transport.path, "responses")
        self.assertEqual(response.response_id, "resp-test")
        self.assertEqual(response.usage.cached_input_tokens, 20)
        self.assertEqual(response.usage.cache_write_input_tokens, 5)
        self.assertEqual(response.usage.reasoning_output_tokens, 3)

    def test_http_400_is_not_retryable(self):
        error = HTTPError("https://example.test", 400, "bad request", {}, BytesIO(b"bad"))
        transport = UrllibJsonTransport("https://example.test", "secret")
        with patch("src.experiments.openai_provider.urlopen", side_effect=error):
            with self.assertRaises(ProviderTransportError) as raised:
                transport.post("responses", {"model": "fixed-model"})
        self.assertFalse(raised.exception.retryable)

    def test_http_429_is_retryable(self):
        error = HTTPError("https://example.test", 429, "rate limited", {}, BytesIO(b"wait"))
        transport = UrllibJsonTransport("https://example.test", "secret")
        with patch("src.experiments.openai_provider.urlopen", side_effect=error):
            with self.assertRaises(ProviderTransportError) as raised:
                transport.post("responses", {"model": "fixed-model"})
        self.assertTrue(raised.exception.retryable)

    def test_read_timeout_is_retryable(self):
        transport = UrllibJsonTransport("https://example.test", "secret")
        with patch(
            "src.experiments.openai_provider.urlopen",
            side_effect=TimeoutError("read timed out"),
        ):
            with self.assertRaises(ProviderTransportError) as raised:
                transport.post("responses", {"model": "fixed-model"})
        self.assertTrue(raised.exception.retryable)


if __name__ == "__main__":
    unittest.main()
