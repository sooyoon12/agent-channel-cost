"""Normalize provider usage without turning missing values into zero."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ProviderUsage:
    input_tokens: int
    cached_input_tokens: int | None
    output_tokens: int
    reasoning_output_tokens: int | None
    total_tokens: int | None
    cache_write_input_tokens: int | None = None

    @property
    def uncached_input_tokens(self) -> int | None:
        if self.cached_input_tokens is None or self.cache_write_input_tokens is None:
            return None
        return self.input_tokens - self.cached_input_tokens - self.cache_write_input_tokens


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    result = int(value)
    if result < 0:
        raise ValueError("usage token counts cannot be negative")
    return result


def _required_int(value: Any, field: str) -> int:
    result = _optional_int(value)
    if result is None:
        raise ValueError(f"provider usage is missing required {field}")
    return result


def parse_provider_usage(raw: Mapping[str, Any]) -> ProviderUsage:
    """Parse Responses-style or Chat-Completions-style usage objects."""

    input_details = raw.get("input_tokens_details") or raw.get("prompt_tokens_details") or {}
    output_details = raw.get("output_tokens_details") or raw.get("completion_tokens_details") or {}
    input_value = raw.get("input_tokens", raw.get("prompt_tokens"))
    output_value = raw.get("output_tokens", raw.get("completion_tokens"))
    cached_value = input_details.get("cached_tokens") if isinstance(input_details, Mapping) else None
    cache_write_value = (
        input_details.get("cache_write_tokens") if isinstance(input_details, Mapping) else None
    )
    reasoning_value = (
        output_details.get("reasoning_tokens") if isinstance(output_details, Mapping) else None
    )
    usage = ProviderUsage(
        input_tokens=_required_int(input_value, "input token count"),
        cached_input_tokens=_optional_int(cached_value),
        output_tokens=_required_int(output_value, "output token count"),
        reasoning_output_tokens=_optional_int(reasoning_value),
        total_tokens=_optional_int(raw.get("total_tokens")),
        cache_write_input_tokens=_optional_int(cache_write_value),
    )
    if usage.cached_input_tokens is not None and usage.cached_input_tokens > usage.input_tokens:
        raise ValueError("cached input tokens exceed total input tokens")
    if (
        usage.cached_input_tokens is not None
        and usage.cache_write_input_tokens is not None
        and usage.cached_input_tokens + usage.cache_write_input_tokens > usage.input_tokens
    ):
        raise ValueError("cached-read plus cache-write input tokens exceed total input tokens")
    if usage.reasoning_output_tokens is not None and usage.reasoning_output_tokens > usage.output_tokens:
        raise ValueError("reasoning output tokens exceed total output tokens")
    return usage
