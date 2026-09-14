"""Evaluate the frozen smoke-run gates without modifying raw records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class SmokeGate:
    name: str
    passed: bool
    observed: float | int | bool | None
    threshold: str


@dataclass(frozen=True)
class SmokeReport:
    passed: bool
    run_count: int
    estimated_main_cost_usd: float | None
    gates: tuple[SmokeGate, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_smoke_run(
    records: Iterable[Mapping[str, Any]],
    *,
    target_ratio: float = 0.4,
    realized_ratio_absolute_tolerance: float,
    max_runs: int = 20,
    main_run_count: int = 480,
) -> SmokeReport:
    values = tuple(record for record in records if record.get("analysis_valid", True))
    if not values:
        raise ValueError("smoke records are empty")
    identities = [
        record["tokens"]["local_i_input"]
        + record["tokens"]["local_t_input"]
        + record["tokens"]["protocol_residual_input"]
        == record["tokens"]["provider_input"]
        for record in values
    ]
    shadow_valid = [bool(record["shadow"]["valid"]) for record in values]
    baseline = [record for record in values if record["condition"] == "B0"]
    baseline_success = mean(bool(record["success"]) for record in baseline) if baseline else 0.0
    baseline_i = sum(record["tokens"]["local_i_input"] for record in baseline)
    baseline_t = sum(record["tokens"]["local_t_input"] for record in baseline)
    t_share = baseline_t / (baseline_i + baseline_t) if baseline_i + baseline_t else 0.0
    t_runs = [record for record in values if record["condition"] == "T-H"]
    compression_events = [
        event
        for record in t_runs
        for event in record.get("events", [])
        if event.get("event_type") == "compression" and event.get("channel") == "T"
    ]
    original_compressed_tokens = sum(
        float(event["original_tokens"]) for event in compression_events
    )
    if original_compressed_tokens:
        realized = (
            sum(float(event["compressed_tokens"]) for event in compression_events)
            / original_compressed_tokens
        )
    else:
        realized = mean(float(record["realized_ratio"]) for record in t_runs) if t_runs else None
    under_context = all(
        int(event["provider_input_tokens"]) < 240_000
        for record in values
        for event in record["events"]
        if event.get("event_type") == "model_output"
    )
    average_cost = mean(
        float(record["model_cost_usd"]["billed"])
        + float(record["compressor_cost_usd"]["billed"])
        for record in values
    )
    estimated_main = average_cost * main_run_count
    gates = (
        SmokeGate("run_count", len(values) <= max_runs, len(values), f"<= {max_runs}"),
        SmokeGate("channel_identity", all(identities), all(identities), "all true"),
        SmokeGate("shadow_paths", all(shadow_valid), all(shadow_valid), "all true"),
        SmokeGate("t_input_share", t_share >= 0.4, t_share, ">= 0.4"),
        SmokeGate("baseline_success", baseline_success >= 0.7, baseline_success, ">= 0.7"),
        SmokeGate(
            "t_h_realized_ratio",
            realized is not None and abs(realized - target_ratio) <= realized_ratio_absolute_tolerance,
            realized,
            f"abs(value - {target_ratio}) <= {realized_ratio_absolute_tolerance}",
        ),
        SmokeGate("request_context", under_context, under_context, "every request < 240000"),
        SmokeGate("budget_projection", estimated_main <= 100, estimated_main, "<= 100 USD"),
    )
    return SmokeReport(all(gate.passed for gate in gates), len(values), estimated_main, gates)
