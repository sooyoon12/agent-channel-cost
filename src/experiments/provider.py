"""Provider boundary plus a network-free scripted implementation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import time
from typing import Any, Mapping, Protocol, Sequence

from src.accounting.provider_usage import ProviderUsage


@dataclass(frozen=True)
class ProviderRequest:
    items: tuple[Mapping[str, Any], ...]
    model: str
    temperature: float
    reasoning_effort: str
    max_output_tokens: int
    output_style: str


@dataclass(frozen=True)
class ProviderResponse:
    response_id: str
    output_items: tuple[Mapping[str, Any], ...]
    usage: ProviderUsage
    latency_ms: float


class Provider(Protocol):
    name: str

    def complete(self, request: ProviderRequest) -> ProviderResponse: ...


class JsonTokenCounter(Protocol):
    def count_json(self, value: Any) -> int: ...


class ScriptedProvider:
    """Emit a gold tool trajectory followed by a final message.

    This is an integration fixture, not a behavioral model. It never opens a
    network connection and provides deterministic synthetic usage.
    """

    name = "scripted-offline"

    def __init__(
        self,
        expected_tool_calls: Sequence[Mapping[str, Any]],
        tokenizer: JsonTokenCounter,
        *,
        protocol_residual_tokens: int = 4,
        arguments_as_json: bool = False,
    ) -> None:
        self._calls = tuple(deepcopy(expected_tool_calls))
        self._tokenizer = tokenizer
        self._position = 0
        self._residual = protocol_residual_tokens
        self._arguments_as_json = arguments_as_json

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        started = time.perf_counter_ns()
        if self._position < len(self._calls):
            call = self._calls[self._position]
            output = (
                {
                    "type": "function_call",
                    "call_id": f"offline-call-{self._position + 1}",
                    "name": call["name"],
                    "arguments": (
                        json.dumps(call["arguments"], ensure_ascii=False, sort_keys=True)
                        if self._arguments_as_json
                        else deepcopy(call["arguments"])
                    ),
                },
            )
        else:
            output = ({"type": "message", "role": "assistant", "content": "완료했습니다."},)
        self._position += 1
        local_input = sum(self._tokenizer.count_json(item) for item in request.items)
        local_output = sum(self._tokenizer.count_json(item) for item in output)
        usage = ProviderUsage(
            input_tokens=local_input + self._residual,
            cached_input_tokens=0,
            output_tokens=local_output,
            reasoning_output_tokens=0,
            total_tokens=local_input + self._residual + local_output,
        )
        elapsed = (time.perf_counter_ns() - started) / 1_000_000
        return ProviderResponse(
            response_id=f"offline-response-{self._position}",
            output_items=output,
            usage=usage,
            latency_ms=elapsed,
        )
