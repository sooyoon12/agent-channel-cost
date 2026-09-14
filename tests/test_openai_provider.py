import unittest

from src.experiments.openai_provider import OpenAIResponsesProvider, build_responses_payload
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
                "input_tokens_details": {"cached_tokens": 20},
                "output_tokens": 10,
                "output_tokens_details": {"reasoning_tokens": 3},
                "total_tokens": 110,
            },
        }


def request(style="normal"):
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
        temperature=0,
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

    def test_adapter_normalizes_usage_without_network(self):
        transport = FakeTransport()
        response = OpenAIResponsesProvider(transport).complete(request())
        self.assertEqual(transport.path, "responses")
        self.assertEqual(response.response_id, "resp-test")
        self.assertEqual(response.usage.cached_input_tokens, 20)
        self.assertEqual(response.usage.reasoning_output_tokens, 3)


if __name__ == "__main__":
    unittest.main()

