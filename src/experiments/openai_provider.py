"""Responses-compatible provider adapter with an injectable JSON transport."""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.accounting.provider_usage import parse_provider_usage

from .provider import ProviderRequest, ProviderResponse


class ProviderTransportError(RuntimeError):
    pass


class JsonTransport(Protocol):
    def post(self, path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...


@dataclass
class UrllibJsonTransport:
    base_url: str
    api_key: str
    timeout_seconds: float = 120.0

    def post(self, path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        request = Request(
            self.base_url.rstrip("/") + "/" + path.lstrip("/"),
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            # Do not include headers or request data; either could contain secrets.
            response_text = exc.read().decode("utf-8", errors="replace")[:1000]
            raise ProviderTransportError(f"provider HTTP {exc.code}: {response_text}") from exc
        except URLError as exc:
            raise ProviderTransportError(f"provider connection failed: {exc.reason}") from exc


def _normalize_input_item(item: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(item)
    if value.get("type") == "function_call" and isinstance(value.get("arguments"), Mapping):
        value["arguments"] = json.dumps(
            value["arguments"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    if value.get("type", "").endswith("_call_output") and not isinstance(value.get("output"), str):
        value["output"] = json.dumps(
            value.get("output"), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    return value


def build_responses_payload(request: ProviderRequest) -> dict[str, Any]:
    tools = []
    input_items = []
    for item in request.items:
        if item.get("type") == "tool_schema":
            tools.append(
                {
                    "type": "function",
                    "name": item["name"],
                    "description": item["description"],
                    "parameters": item["parameters"],
                    "strict": True,
                }
            )
        else:
            input_items.append(_normalize_input_item(item))
    payload: dict[str, Any] = {
        "model": request.model,
        "input": input_items,
        "tools": tools,
        "temperature": request.temperature,
        "reasoning": {"effort": request.reasoning_effort},
        "max_output_tokens": request.max_output_tokens,
    }
    if request.output_style == "concise":
        payload["text"] = {"verbosity": "low"}
    return payload


class OpenAIResponsesProvider:
    name = "openai"

    def __init__(self, transport: JsonTransport) -> None:
        self.transport = transport

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        payload = build_responses_payload(request)
        started = time.perf_counter_ns()
        raw = self.transport.post("responses", payload)
        elapsed = (time.perf_counter_ns() - started) / 1_000_000
        if "usage" not in raw:
            raise ProviderTransportError("provider response is missing usage")
        output = raw.get("output")
        if not isinstance(output, list):
            raise ProviderTransportError("provider response output is not a list")
        return ProviderResponse(
            response_id=str(raw.get("id", "missing-response-id")),
            output_items=tuple(output),
            usage=parse_provider_usage(raw["usage"]),
            latency_ms=elapsed,
        )

