"""Cache-aware billed cost and cache-free list-price cost."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .provider_usage import ProviderUsage


MILLION = Decimal(1_000_000)


@dataclass(frozen=True)
class PriceTier:
    uncached_input_per_million: Decimal
    cached_input_per_million: Decimal
    output_per_million: Decimal
    reasoning_output_per_million: Decimal | None = None
    cache_write_input_per_million: Decimal | None = None

    def __post_init__(self) -> None:
        for value in (
            self.uncached_input_per_million,
            self.cached_input_per_million,
            self.output_per_million,
        ):
            if value < 0:
                raise ValueError("prices cannot be negative")
        if self.reasoning_output_per_million is not None and self.reasoning_output_per_million < 0:
            raise ValueError("prices cannot be negative")
        if self.cache_write_input_per_million is not None and self.cache_write_input_per_million < 0:
            raise ValueError("prices cannot be negative")


@dataclass(frozen=True)
class PricingSchedule:
    standard: PriceTier
    long_context: PriceTier | None = None
    long_context_threshold: int | None = None
    version: str | None = None

    def tier_for(self, input_tokens: int) -> PriceTier:
        if (
            self.long_context is not None
            and self.long_context_threshold is not None
            and input_tokens > self.long_context_threshold
        ):
            return self.long_context
        return self.standard


@dataclass(frozen=True)
class CostBreakdown:
    uncached_input: Decimal
    cached_input: Decimal
    cache_write_input: Decimal
    visible_output: Decimal
    reasoning_output: Decimal

    @property
    def total(self) -> Decimal:
        return (
            self.uncached_input
            + self.cached_input
            + self.cache_write_input
            + self.visible_output
            + self.reasoning_output
        )


@dataclass(frozen=True)
class DualCost:
    billed: CostBreakdown
    list_price: CostBreakdown


def _cost(tokens: int, price_per_million: Decimal) -> Decimal:
    return Decimal(tokens) * price_per_million / MILLION


def calculate_model_cost(usage: ProviderUsage, schedule: PricingSchedule) -> DualCost:
    tier = schedule.tier_for(usage.input_tokens)
    cached = usage.cached_input_tokens
    if cached is None:
        raise ValueError("billed cost requires provider cached_input_tokens; missing is not zero")
    cache_write = usage.cache_write_input_tokens
    if cache_write is None:
        if tier.cache_write_input_per_million is not None:
            raise ValueError(
                "billed cost requires provider cache_write_input_tokens when cache writes are priced"
            )
        cache_write = 0
    elif cache_write and tier.cache_write_input_per_million is None:
        raise ValueError("cache-write tokens cannot be priced without a cache-write rate")
    if cached + cache_write > usage.input_tokens:
        raise ValueError("cached-read plus cache-write input tokens exceed total input tokens")
    reasoning = usage.reasoning_output_tokens
    if reasoning is None:
        if tier.reasoning_output_per_million is not None:
            raise ValueError(
                "a distinct reasoning rate cannot be applied when reasoning tokens are unavailable"
            )
        # Total output remains billable even when its reasoning sub-count is not
        # exposed. The token log still preserves reasoning as null.
        reasoning = 0
    if reasoning > usage.output_tokens:
        raise ValueError("reasoning output tokens exceed total output tokens")
    visible_output = usage.output_tokens - reasoning
    reasoning_rate = tier.reasoning_output_per_million or tier.output_per_million
    cache_write_rate = tier.cache_write_input_per_million or Decimal(0)

    billed = CostBreakdown(
        uncached_input=_cost(
            usage.input_tokens - cached - cache_write, tier.uncached_input_per_million
        ),
        cached_input=_cost(cached, tier.cached_input_per_million),
        cache_write_input=_cost(cache_write, cache_write_rate),
        visible_output=_cost(visible_output, tier.output_per_million),
        reasoning_output=_cost(reasoning, reasoning_rate),
    )
    list_price = CostBreakdown(
        uncached_input=_cost(usage.input_tokens, tier.uncached_input_per_million),
        cached_input=Decimal(0),
        cache_write_input=Decimal(0),
        visible_output=_cost(visible_output, tier.output_per_million),
        reasoning_output=_cost(reasoning, reasoning_rate),
    )
    return DualCost(billed=billed, list_price=list_price)


@dataclass(frozen=True)
class SeparatedRunCost:
    model: DualCost
    compressor: DualCost | None
