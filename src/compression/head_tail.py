"""Deterministic head/tail truncation baseline for structured observations."""

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import math
import re
import time
from typing import Any

from .base import CompressionResult, TokenCounter, content_handle, validate_ratio


_TEXT_UNITS = re.compile(r"\S+\s*", re.UNICODE)


def _head_tail_indices(length: int, keep: int) -> tuple[int, ...]:
    if keep <= 0 or length <= 0:
        return ()
    if keep >= length:
        return tuple(range(length))
    head = (keep + 1) // 2
    tail = keep - head
    return tuple(range(head)) + tuple(range(length - tail, length))


def _transform(value: Any, ratio: float) -> Any:
    if isinstance(value, str):
        units = _TEXT_UNITS.findall(value)
        if len(units) <= 1:
            units = list(value)
        keep = min(len(units), max(0, math.floor(len(units) * ratio)))
        return "".join(units[index] for index in _head_tail_indices(len(units), keep)).rstrip()
    if isinstance(value, list):
        keep = min(len(value), max(0, math.floor(len(value) * ratio)))
        return [deepcopy(value[index]) for index in _head_tail_indices(len(value), keep)]
    if isinstance(value, tuple):
        keep = min(len(value), max(0, math.floor(len(value) * ratio)))
        return tuple(deepcopy(value[index]) for index in _head_tail_indices(len(value), keep))
    if isinstance(value, dict):
        return {key: _transform(item, ratio) for key, item in value.items()}
    return deepcopy(value)


class HeadTailCompressor:
    name = "head_tail"
    version = "1"

    def compress(
        self, value: Any, *, target_ratio: float, tokenizer: TokenCounter
    ) -> CompressionResult:
        validate_ratio(target_ratio)
        started = time.perf_counter_ns()
        original = deepcopy(value)
        original_tokens = tokenizer.count_json(original)
        target_tokens = max(1, math.floor(original_tokens * target_ratio)) if original_tokens else 0
        if target_ratio == 1 or original_tokens == 0:
            compressed = deepcopy(original)
            compressed_tokens = original_tokens
        else:
            low, high = 0.0, 1.0
            compressed = _transform(original, 0.0)
            compressed_tokens = tokenizer.count_json(compressed)
            for _ in range(24):
                middle = (low + high) / 2
                candidate = _transform(original, middle)
                candidate_tokens = tokenizer.count_json(candidate)
                if candidate_tokens <= target_tokens:
                    low = middle
                    compressed, compressed_tokens = candidate, candidate_tokens
                else:
                    high = middle
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
