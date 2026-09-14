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
    if not model.get("model_id"):
        issues.append("model.model_id is not fixed")
    tokenizer = model.get("tokenizer", {})
    if not tokenizer.get("exact_encoding") and not tokenizer.get("proxy_encoding"):
        issues.append("neither exact nor proxy tokenizer encoding is recorded")
    standard = pricing.get("standard", {})
    if pricing.get("unit_tokens") != 1_000_000:
        issues.append("pricing.unit_tokens must be 1000000 for the current calculator")
    for field in (
        "uncached_input_per_unit",
        "cached_input_per_unit",
        "output_per_unit",
    ):
        if standard.get(field) is None:
            issues.append(f"pricing.standard.{field} is missing")
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
    if compression.get("target_ratio") != 0.4:
        issues.append("compression target_ratio differs from the frozen value 0.4")
    if compression.get("algorithm") != "uniform_extract":
        issues.append("compression algorithm is not the implemented frozen algorithm")
    return PreflightReport(not issues, tuple(issues))


def require_live_ready(**configs: Mapping[str, Any]) -> None:
    report = validate_live_configuration(**configs)
    if not report.ready:
        raise RuntimeError("live run preflight failed: " + "; ".join(report.issues))
