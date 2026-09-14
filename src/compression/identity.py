"""No-op compressor used by B0 and uncompressed channels."""

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import time
from typing import Any

from .base import CompressionResult, TokenCounter, content_handle, validate_ratio


class IdentityCompressor:
    name = "identity"
    version = "1"

    def compress(self, value: Any, *, target_ratio: float, tokenizer: TokenCounter) -> CompressionResult:
        validate_ratio(target_ratio)
        started = time.perf_counter_ns()
        copied = deepcopy(value)
        tokens = tokenizer.count_json(value)
        elapsed = (time.perf_counter_ns() - started) / 1_000_000
        handle = content_handle(value)
        return CompressionResult(
            algorithm=self.name,
            algorithm_version=self.version,
            original=deepcopy(value),
            compressed=copied,
            original_handle=handle,
            compressed_handle=handle,
            original_tokens=tokens,
            compressed_tokens=tokens,
            target_ratio=target_ratio,
            realized_ratio=1.0 if tokens else 0.0,
            latency_ms=elapsed,
            compressor_input_tokens=0,
            compressor_output_tokens=0,
            billed_cost_usd=Decimal(0),
            list_price_cost_usd=Decimal(0),
        )

