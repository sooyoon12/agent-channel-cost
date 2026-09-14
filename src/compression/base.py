"""Compression contracts and immutable provenance records."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
from typing import Any, Protocol


class TokenCounter(Protocol):
    def count_text(self, text: str) -> int: ...

    def count_json(self, value: Any) -> int: ...


@dataclass(frozen=True)
class CompressionResult:
    algorithm: str
    algorithm_version: str
    original: Any
    compressed: Any
    original_handle: str
    compressed_handle: str
    original_tokens: int
    compressed_tokens: int
    target_ratio: float
    realized_ratio: float
    latency_ms: float
    compressor_input_tokens: int
    compressor_output_tokens: int
    billed_cost_usd: Decimal
    list_price_cost_usd: Decimal

    @property
    def target_met(self) -> bool:
        if self.original_tokens == 0:
            return True
        return self.compressed_tokens <= max(1, int(self.original_tokens * self.target_ratio))


class Compressor(Protocol):
    name: str
    version: str

    def compress(self, value: Any, *, target_ratio: float, tokenizer: TokenCounter) -> CompressionResult: ...


def content_handle(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_ratio(target_ratio: float) -> None:
    if not 0 < target_ratio <= 1:
        raise ValueError("target_ratio must be in (0, 1]")

