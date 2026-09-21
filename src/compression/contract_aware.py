"""Contract-aware compression that never removes decision-bearing fields."""

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import time
from typing import Any, Iterable

from .base import CompressionResult, TokenCounter, content_handle, validate_ratio


def _drop_non_contract_fields(value: Any, keys: frozenset[str]) -> Any:
    if isinstance(value, dict):
        return {
            key: _drop_non_contract_fields(item, keys)
            for key, item in value.items()
            if key not in keys
        }
    if isinstance(value, list):
        return [_drop_non_contract_fields(item, keys) for item in value]
    if isinstance(value, tuple):
        return tuple(_drop_non_contract_fields(item, keys) for item in value)
    return deepcopy(value)


class ContractAwareCompressor:
    """Remove only fields declared non-contractual by the tool adapter.

    The target ratio is a budget request, not permission to damage the tool
    contract.  If removing all declared expendable fields cannot reach the
    target, the compressor stops at the contract-preserving floor and records
    the resulting ratio truthfully.
    """

    name = "contract_guard"
    version = "1"

    def __init__(self, *, non_contract_keys: Iterable[str] = ("notes",)) -> None:
        self.non_contract_keys = frozenset(non_contract_keys)
        if not self.non_contract_keys:
            raise ValueError("at least one non-contract key is required")

    def compress(
        self, value: Any, *, target_ratio: float, tokenizer: TokenCounter
    ) -> CompressionResult:
        validate_ratio(target_ratio)
        started = time.perf_counter_ns()
        original = deepcopy(value)
        original_tokens = tokenizer.count_json(original)
        candidate = _drop_non_contract_fields(original, self.non_contract_keys)
        candidate_tokens = tokenizer.count_json(candidate)
        # Never increase the payload when a configured key is absent.
        if candidate_tokens > original_tokens:
            candidate, candidate_tokens = original, original_tokens
        elapsed = (time.perf_counter_ns() - started) / 1_000_000
        realized = candidate_tokens / original_tokens if original_tokens else 0.0
        return CompressionResult(
            algorithm=self.name,
            algorithm_version=self.version,
            original=original,
            compressed=candidate,
            original_handle=content_handle(original),
            compressed_handle=content_handle(candidate),
            original_tokens=original_tokens,
            compressed_tokens=candidate_tokens,
            target_ratio=target_ratio,
            realized_ratio=realized,
            latency_ms=elapsed,
            compressor_input_tokens=0,
            compressor_output_tokens=0,
            billed_cost_usd=Decimal(0),
            list_price_cost_usd=Decimal(0),
        )


class OracleContractCompressor(ContractAwareCompressor):
    """V2 label for the schema-oracle upper bound, not a proposed method."""

    name = "oracle_contract"
    version = "2"
