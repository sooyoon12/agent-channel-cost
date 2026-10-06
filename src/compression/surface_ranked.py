"""Surface-grounded structured extraction without gold or parsed semantics.

The compressor ranks records by how many of their scalar values occur in the
raw user request.  It never receives a gold identifier, a generated semantic
form, or a declaration of safe-to-drop fields.  This makes it a deployable
middle baseline between position-only truncation and the oracle references.
"""

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import math
import re
import time
from typing import Any, Mapping

from .base import CompressionResult, TokenCounter, content_handle, validate_ratio


_WORDISH = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", re.UNICODE)


def _normalized_literals(text: str) -> frozenset[str]:
    """Return case-folded surface literals, including Korean-adjacent values."""

    folded = text.casefold()
    literals = {match.group(0) for match in _WORDISH.finditer(folded)}
    # Full-substring matching is also used for values such as Korean strings.
    literals.add(folded)
    return frozenset(literals)


def _surface_match(value: Any, *, prompt: str, literals: frozenset[str]) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, (dict, list, tuple)):
        return False
    rendered = str(value).strip().casefold()
    if len(rendered) < 2:
        return False
    return rendered in literals or rendered in prompt


def _record_score(record: Mapping[str, Any], *, prompt: str, literals: frozenset[str]) -> int:
    # Opaque identifiers and prose blobs are poor relevance signals: identifiers
    # are normally unknown before the tool call and prose creates accidental hits.
    ignored_fragments = ("id", "code", "note", "description", "memo", "attendee")
    return sum(
        _surface_match(value, prompt=prompt, literals=literals)
        for key, value in record.items()
        if not any(fragment in str(key).casefold() for fragment in ignored_fragments)
    )


def _candidate_list(value: Any) -> tuple[dict[str, Any], str, list[Mapping[str, Any]]] | None:
    if not isinstance(value, dict):
        return None
    choices: list[tuple[int, str, list[Mapping[str, Any]]]] = []
    for key, item in value.items():
        if isinstance(item, list) and item and all(isinstance(entry, Mapping) for entry in item):
            choices.append((len(item), str(key), item))
    if not choices:
        return None
    _, key, records = max(choices, key=lambda candidate: candidate[0])
    return value, key, records


class SurfaceRankedCompressor:
    """Keep the most surface-relevant structured records within a token budget."""

    name = "surface_ranked_extract"
    version = "1"

    def __init__(self, user_request: str) -> None:
        if not user_request.strip():
            raise ValueError("user_request must be nonempty")
        self.user_request = user_request
        self._prompt = user_request.casefold()
        self._literals = _normalized_literals(user_request)

    def compress(
        self, value: Any, *, target_ratio: float, tokenizer: TokenCounter
    ) -> CompressionResult:
        validate_ratio(target_ratio)
        started = time.perf_counter_ns()
        original = deepcopy(value)
        original_tokens = tokenizer.count_json(original)
        target_tokens = max(1, math.floor(original_tokens * target_ratio)) if original_tokens else 0
        compressed = deepcopy(original)
        selected = _candidate_list(compressed)
        if selected is not None and original_tokens > target_tokens:
            container, key, records = selected
            ranked = sorted(
                enumerate(records),
                key=lambda item: (
                    -_record_score(item[1], prompt=self._prompt, literals=self._literals),
                    item[0],
                ),
            )
            kept_indices = set(range(len(records)))
            # Remove the least relevant record first.  Stop at the largest
            # representation that does not exceed the requested token budget.
            removal_order = list(reversed([index for index, _ in ranked]))
            best = deepcopy(compressed)
            for index in removal_order:
                kept_indices.remove(index)
                candidate = deepcopy(compressed)
                candidate[key] = [
                    deepcopy(record)
                    for position, record in enumerate(records)
                    if position in kept_indices
                ]
                if "count" in candidate and isinstance(candidate["count"], int):
                    candidate["count"] = len(candidate[key])
                best = candidate
                if tokenizer.count_json(candidate) <= target_tokens:
                    break
            compressed = best
        compressed_tokens = tokenizer.count_json(compressed)
        if compressed_tokens > original_tokens:
            compressed, compressed_tokens = deepcopy(original), original_tokens
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
