"""Independent shadow list-price calculations over a fixed trajectory."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Iterable, Protocol

from .pricing import MILLION, PricingSchedule


class JsonTokenCounter(Protocol):
    def count_json(self, value: Any) -> int: ...


@dataclass(frozen=True)
class ShadowValidation:
    path_a_list_cost: Decimal
    path_b_list_cost: Decimal
    difference: Decimal
    tolerance: Decimal
    valid: bool


@dataclass(frozen=True)
class ShadowContent:
    """One replayed request item and its restorable original.

    Uncompressed items use equal values. Compressed I or T items preserve both
    values, corresponding to the original/compressed handles in the run log.
    """

    compressed: Any
    original: Any


@dataclass(frozen=True)
class ReplayShadowRequest:
    contents: tuple[ShadowContent, ...]
    protocol_residual_tokens: int
    output_tokens: int
    reasoning_output_tokens: int

    def __post_init__(self) -> None:
        if min(self.protocol_residual_tokens, self.output_tokens, self.reasoning_output_tokens) < 0:
            raise ValueError("shadow token counts cannot be negative")
        if self.reasoning_output_tokens > self.output_tokens:
            raise ValueError("reasoning output exceeds total output")


def _content_tokens(values: Iterable[Any], tokenizer: JsonTokenCounter) -> int:
    return sum(tokenizer.count_json(value) for value in values)


def _replay_request_cost(
    input_content_tokens: int,
    request: ReplayShadowRequest,
    schedule: PricingSchedule,
) -> Decimal:
    total_input = input_content_tokens + request.protocol_residual_tokens
    tier = schedule.tier_for(total_input)
    reasoning_rate = tier.reasoning_output_per_million or tier.output_per_million
    visible_output = request.output_tokens - request.reasoning_output_tokens
    return (
        Decimal(total_input) * tier.uncached_input_per_million
        + Decimal(visible_output) * tier.output_per_million
        + Decimal(request.reasoning_output_tokens) * reasoning_rate
    ) / MILLION


def shadow_path_a_retokenized(
    requests: Iterable[ReplayShadowRequest],
    schedule: PricingSchedule,
    tokenizer: JsonTokenCounter,
) -> Decimal:
    """Path A: reconstruct every original request item, then retokenize it."""

    total = Decimal(0)
    for request in requests:
        original_tokens = _content_tokens((part.original for part in request.contents), tokenizer)
        total += _replay_request_cost(original_tokens, request, schedule)
    return total


def shadow_path_b_delta(
    requests: Iterable[ReplayShadowRequest],
    schedule: PricingSchedule,
    tokenizer: JsonTokenCounter,
) -> Decimal:
    """Path B: price compressed requests, then add each replayed token delta."""

    total = Decimal(0)
    for request in requests:
        compressed_tokens = _content_tokens((part.compressed for part in request.contents), tokenizer)
        total += _replay_request_cost(compressed_tokens, request, schedule)
        tier = schedule.tier_for(compressed_tokens + request.protocol_residual_tokens)
        for part in request.contents:
            delta = tokenizer.count_json(part.original) - tokenizer.count_json(part.compressed)
            total += Decimal(delta) * tier.uncached_input_per_million / MILLION
    return total


def compressed_replay_list_cost(
    requests: Iterable[ReplayShadowRequest],
    schedule: PricingSchedule,
    tokenizer: JsonTokenCounter,
) -> Decimal:
    total = Decimal(0)
    for request in requests:
        compressed_tokens = _content_tokens((part.compressed for part in request.contents), tokenizer)
        total += _replay_request_cost(compressed_tokens, request, schedule)
    return total


def validate_retokenized_shadow_paths(
    requests: Iterable[ReplayShadowRequest],
    schedule: PricingSchedule,
    tokenizer: JsonTokenCounter,
    *,
    tolerance: Decimal,
) -> ShadowValidation:
    request_list = tuple(requests)
    path_a = shadow_path_a_retokenized(request_list, schedule, tokenizer)
    path_b = shadow_path_b_delta(request_list, schedule, tokenizer)
    difference = abs(path_a - path_b)
    return ShadowValidation(path_a, path_b, difference, tolerance, difference <= tolerance)


def gross_saving(shadow_list_cost: Decimal, compressed_list_cost_value: Decimal) -> Decimal:
    return shadow_list_cost - compressed_list_cost_value


def behavioral_displacement(
    shadow_list_cost: Decimal,
    baseline_list_cost: Decimal,
    compressor_list_cost: Decimal,
) -> Decimal:
    return shadow_list_cost - baseline_list_cost + compressor_list_cost


def aggregate_cdr(
    gross_savings: Iterable[Decimal],
    behavioral_displacements: Iterable[Decimal],
    *,
    minimum_denominator: Decimal = Decimal("0"),
) -> Decimal | None:
    gross = sum(gross_savings, Decimal(0))
    if gross <= minimum_denominator:
        return None
    return sum(behavioral_displacements, Decimal(0)) / gross
