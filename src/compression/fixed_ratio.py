"""Fixed-ratio, deterministic extractive compression.

The algorithm is intentionally simple and outcome-independent. It keeps JSON
keys intact, samples list entries at evenly spaced positions, and keeps evenly
spaced text units. A binary search tightens the internal keep ratio until the
configured tokenizer reports a result close to (and normally below) the target.
"""

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import math
import re
import time
from typing import Any

from .base import CompressionResult, TokenCounter, content_handle, validate_ratio


_TEXT_UNITS = re.compile(r"\S+\s*", re.UNICODE)


def _uniform_indices(length: int, keep: int) -> tuple[int, ...]:
    if keep <= 0 or length <= 0:
        return ()
    if keep >= length:
        return tuple(range(length))
    if keep == 1:
        return (0,)
    return tuple(round(position * (length - 1) / (keep - 1)) for position in range(keep))


def _compress_text(text: str, ratio: float) -> str:
    units = _TEXT_UNITS.findall(text)
    if len(units) <= 1:
        units = list(text)
    keep = min(len(units), max(0, math.floor(len(units) * ratio)))
    return "".join(units[index] for index in _uniform_indices(len(units), keep)).rstrip()


def _transform(value: Any, ratio: float) -> Any:
    if isinstance(value, str):
        return _compress_text(value, ratio)
    if isinstance(value, list):
        keep = min(len(value), max(0, math.floor(len(value) * ratio)))
        return [deepcopy(value[index]) for index in _uniform_indices(len(value), keep)]
    if isinstance(value, tuple):
        keep = min(len(value), max(0, math.floor(len(value) * ratio)))
        return tuple(deepcopy(value[index]) for index in _uniform_indices(len(value), keep))
    if isinstance(value, dict):
        # Keys are protocol/content structure and are never compressed.
        return {key: _transform(item, ratio) for key, item in value.items()}
    return deepcopy(value)


class FixedRatioCompressor:
    name = "uniform_extract"
    version = "1"

    def compress(self, value: Any, *, target_ratio: float, tokenizer: TokenCounter) -> CompressionResult:
        validate_ratio(target_ratio)
        started = time.perf_counter_ns()
        original_tokens = tokenizer.count_json(value)
        target_tokens = max(1, math.floor(original_tokens * target_ratio)) if original_tokens else 0
        if target_ratio == 1 or original_tokens == 0:
            compressed = deepcopy(value)
            compressed_tokens = original_tokens
        else:
            low, high = 0.0, 1.0
            best = _transform(value, 0.0)
            best_tokens = tokenizer.count_json(best)
            # Monotonicity holds for the supported JSON/text transforms. Keep
            # the largest representation that does not exceed the token target.
            for _ in range(24):
                middle = (low + high) / 2
                candidate = _transform(value, middle)
                candidate_tokens = tokenizer.count_json(candidate)
                if candidate_tokens <= target_tokens:
                    low = middle
                    best, best_tokens = candidate, candidate_tokens
                else:
                    high = middle
            compressed, compressed_tokens = best, best_tokens
        elapsed = (time.perf_counter_ns() - started) / 1_000_000
        realized = compressed_tokens / original_tokens if original_tokens else 0.0
        return CompressionResult(
            algorithm=self.name,
            algorithm_version=self.version,
            original=deepcopy(value),
            compressed=compressed,
            original_handle=content_handle(value),
            compressed_handle=content_handle(compressed),
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            target_ratio=target_ratio,
            realized_ratio=realized,
            latency_ms=elapsed,
            compressor_input_tokens=0,
            compressor_output_tokens=0,
            billed_cost_usd=Decimal(0),
            list_price_cost_usd=Decimal(0),
        )

