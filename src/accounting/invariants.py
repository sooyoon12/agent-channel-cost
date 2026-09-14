"""Validation rules for per-request and per-run accounting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResidualCheck:
    local_i_tokens: int
    local_t_tokens: int
    provider_input_tokens: int
    residual_tokens: int
    allowed_tokens: int
    valid: bool
    reason: str | None


def check_input_residual(
    local_i_tokens: int,
    local_t_tokens: int,
    provider_input_tokens: int,
    *,
    absolute_tolerance: int,
    fractional_tolerance: float,
) -> ResidualCheck:
    values = (local_i_tokens, local_t_tokens, provider_input_tokens, absolute_tolerance)
    if any(value < 0 for value in values):
        raise ValueError("token counts and tolerances cannot be negative")
    if fractional_tolerance < 0:
        raise ValueError("fractional tolerance cannot be negative")
    residual = provider_input_tokens - local_i_tokens - local_t_tokens
    allowed = max(absolute_tolerance, int(provider_input_tokens * fractional_tolerance))
    if residual < 0:
        valid, reason = False, "negative protocol residual"
    elif residual > allowed:
        valid, reason = False, "protocol residual exceeds configured tolerance"
    else:
        valid, reason = True, None
    return ResidualCheck(
        local_i_tokens=local_i_tokens,
        local_t_tokens=local_t_tokens,
        provider_input_tokens=provider_input_tokens,
        residual_tokens=residual,
        allowed_tokens=allowed,
        valid=valid,
        reason=reason,
    )


def assert_channel_partition(classified_count: int, source_count: int) -> None:
    if classified_count != source_count:
        raise AssertionError(
            f"channel partition lost or duplicated items: {classified_count=} != {source_count=}"
        )

