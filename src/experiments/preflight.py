"""Fail-closed validation before any paid provider request."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class PreflightReport:
    ready: bool
    issues: tuple[str, ...]


def validate_live_configuration(
    *,
    model: Mapping[str, Any],
    pricing: Mapping[str, Any],
    experiment: Mapping[str, Any],
    compression: Mapping[str, Any],
) -> PreflightReport:
    issues: list[str] = []
    if model.get("provider") != "openai":
        issues.append("model.provider differs from frozen provider openai")
    if model.get("model_id") != "gpt-5.6-terra":
        issues.append("model.model_id differs from the frozen value gpt-5.6-terra")
    if model.get("model_id") != model.get("requested_model_label"):
        issues.append("model.model_id differs from the frozen requested model label")
    if model.get("temperature") != 0:
        issues.append("model.temperature differs from the frozen value 0")
    if model.get("reasoning_effort") != "low":
        issues.append("model.reasoning_effort differs from the frozen value low")
    tokenizer = model.get("tokenizer", {})
    if tokenizer.get("exact_encoding") != "o200k_base" or tokenizer.get("proxy_encoding") is not None:
        issues.append("tokenizer must use the frozen exact o200k_base mapping without a proxy")
    if tokenizer.get("version") != "0.14.0":
        issues.append("tokenizer.version differs from the frozen value 0.14.0")
    standard = pricing.get("standard", {})
    if pricing.get("unit_tokens") != 1_000_000:
        issues.append("pricing.unit_tokens must be 1000000 for the current calculator")
    for field in (
        "uncached_input_per_unit",
        "cached_input_per_unit",
        "cache_write_input_per_unit",
        "output_per_unit",
    ):
        if standard.get(field) is None:
            issues.append(f"pricing.standard.{field} is missing")
    long_context = pricing.get("long_context", {})
    if long_context.get("threshold_input_tokens") != 272_000:
        issues.append("pricing.long_context.threshold_input_tokens must be 272000")
    for field in (
        "uncached_input_per_unit",
        "cached_input_per_unit",
        "cache_write_input_per_unit",
        "output_per_unit",
    ):
        if long_context.get(field) is None:
            issues.append(f"pricing.long_context.{field} is missing")
    if not pricing.get("price_version"):
        issues.append("pricing.price_version is missing")
    if not pricing.get("source_url"):
        issues.append("pricing.source_url is missing")
    residual = experiment.get("residual_tolerance", {})
    if residual.get("absolute_tokens") is None:
        issues.append("residual absolute tolerance is not frozen")
    if residual.get("fraction_of_provider_input") is None:
        issues.append("residual fractional tolerance is not frozen")
    if experiment.get("realized_ratio_absolute_tolerance") is None:
        issues.append("realized-ratio tolerance is not frozen")
    if experiment.get("max_agent_turns") != 6:
        issues.append("max_agent_turns differs from the frozen value 6")
    if experiment.get("max_request_input_tokens") != 240_000:
        issues.append("max_request_input_tokens differs from the frozen value 240000")
    if experiment.get("max_identical_tool_calls") != 2:
        issues.append("max_identical_tool_calls differs from the frozen value 2")
    if experiment.get("baseline_success_aggregation") != "majority_vote":
        issues.append("baseline_success_aggregation must be majority_vote")
    if experiment.get("bootstrap_resampling_unit") != "pair_id":
        issues.append("bootstrap_resampling_unit must be pair_id")
    if experiment.get("bootstrap_replicates") != 10_000:
        issues.append("bootstrap_replicates differs from the frozen value 10000")
    if experiment.get("execution_order_seed") != 20_260_914:
        issues.append("execution_order_seed differs from the frozen value 20260914")
    if experiment.get("provider_error_retries") != 1:
        issues.append("provider_error_retries differs from the frozen value 1")
    if compression.get("target_ratio") != 0.4:
        issues.append("compression target_ratio differs from the frozen value 0.4")
    if compression.get("algorithm") != "uniform_extract":
        issues.append("compression algorithm is not the implemented frozen algorithm")
    return PreflightReport(not issues, tuple(issues))


def require_live_ready(**configs: Mapping[str, Any]) -> None:
    report = validate_live_configuration(**configs)
    if not report.ready:
        raise RuntimeError("live run preflight failed: " + "; ".join(report.issues))
