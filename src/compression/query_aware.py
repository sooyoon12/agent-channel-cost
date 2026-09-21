"""Structured query-aware filtering reference for v2 observations.

This compressor receives the parsed task constraints.  It does not receive the
gold record/event id, but it is still an upper-reference implementation because
the benchmark supplies a perfect semantic parse.
"""

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import time
from typing import Any, Mapping

from .base import CompressionResult, TokenCounter, content_handle, validate_ratio


class QueryAwareCompressor:
    name = "query_aware_filter"
    version = "1"

    def __init__(self, semantic_form: Mapping[str, Any]) -> None:
        self.semantic_form = deepcopy(dict(semantic_form))

    def _record_matches(self, record: Mapping[str, Any]) -> bool:
        semantic = self.semantic_form
        if record.get("region") != semantic.get("region"):
            return False
        if record.get("status") != semantic.get("status"):
            return False
        if "category" in semantic and record.get("category") != semantic["category"]:
            return False
        if "priority" in semantic and record.get("priority") != semantic["priority"]:
            return False
        if "minimum_amount" in semantic and record.get("amount", float("-inf")) < semantic["minimum_amount"]:
            return False
        if "on_or_after" in semantic and record.get("date", "") < semantic["on_or_after"]:
            return False
        if "maximum_risk" in semantic and record.get("risk_score", float("inf")) > semantic["maximum_risk"]:
            return False
        return True

    def _transform(self, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return deepcopy(value)
        result = deepcopy(dict(value))
        records = result.get("records")
        if isinstance(records, list):
            result["records"] = [record for record in records if self._record_matches(record)]
        events = result.get("events")
        room = self.semantic_form.get("room")
        if isinstance(events, list) and room is not None:
            result["events"] = [event for event in events if event.get("room") == room]
            if "count" in result:
                result["count"] = len(result["events"])
        return result

    def compress(
        self, value: Any, *, target_ratio: float, tokenizer: TokenCounter
    ) -> CompressionResult:
        validate_ratio(target_ratio)
        started = time.perf_counter_ns()
        original = deepcopy(value)
        original_tokens = tokenizer.count_json(original)
        compressed = self._transform(original)
        compressed_tokens = tokenizer.count_json(compressed)
        if compressed_tokens > original_tokens:
            compressed = deepcopy(original)
            compressed_tokens = original_tokens
        elapsed = (time.perf_counter_ns() - started) / 1_000_000
        return CompressionResult(
            algorithm=self.name,
            algorithm_version=self.version,
            original=original,
            compressed=compressed,
            original_handle=content_handle(original),
            compressed_handle=content_handle(compressed),
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            target_ratio=target_ratio,
            realized_ratio=compressed_tokens / original_tokens if original_tokens else 0.0,
            latency_ms=elapsed,
            compressor_input_tokens=0,
            compressor_output_tokens=0,
            billed_cost_usd=Decimal(0),
            list_price_cost_usd=Decimal(0),
        )
