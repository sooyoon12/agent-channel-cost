"""Prespecified comparisons for the post-hoc ratio-sweep study."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from decimal import Decimal
import math
import random
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from .metrics import (
    aggregate_cost_transfer,
    bootstrap_cdr,
    build_cost_transfer_observations,
    mcnemar_exact,
    summarize_conditions,
)


def evaluation_method_comparison(
    records: Iterable[Mapping[str, Any]], *, success_margin: float = 0.05
) -> list[dict[str, Any]]:
    """Contrast token-only ranking with a success-gated deployment decision."""

    summaries = summarize_conditions(records)
    baseline = summaries["B0"]
    rows = []
    for condition, summary in summaries.items():
        if condition == "B0":
            continue
        successes = summary["run_count"] * summary["success_rate"]
        total_cost = summary["run_count"] * summary["mean_model_list_cost_usd"]
        input_saving = 1 - (
            summary["mean_provider_input_tokens"]
            / baseline["mean_provider_input_tokens"]
        )
        list_cost_saving = 1 - (
            summary["mean_model_list_cost_usd"]
            / baseline["mean_model_list_cost_usd"]
        )
        success_drop = baseline["success_rate"] - summary["success_rate"]
        rows.append(
            {
                "condition": condition,
                "provider_input_token_saving": input_saving,
                "list_price_cost_saving": list_cost_saving,
                "success_rate": summary["success_rate"],
                "success_drop_vs_b0": success_drop,
                "list_cost_per_success_usd": (
                    total_cost / successes if successes else None
                ),
                "token_only_recommends": False,
                "success_gated_eligible": (
                    success_drop <= success_margin and list_cost_saving > 0
                ),
            }
        )
    token_winner = max(rows, key=lambda row: row["provider_input_token_saving"])
    token_winner["token_only_recommends"] = True
    eligible = [row for row in rows if row["success_gated_eligible"]]
    if eligible:
        max(eligible, key=lambda row: row["list_price_cost_saving"])[
            "success_gated_recommends"
        ] = True
    for row in rows:
        row.setdefault("success_gated_recommends", False)
    return sorted(rows, key=lambda row: row["condition"])


def _binomial_cdf(k: int, n: int, probability: float) -> float:
    return sum(
        math.comb(n, index)
        * probability**index
        * (1 - probability) ** (n - index)
        for index in range(k + 1)
    )


def clopper_pearson_upper(k: int, n: int, *, alpha: float = 0.05) -> float:
    """One-sided exact upper confidence bound for a binomial proportion."""

    if not 0 <= k <= n or n <= 0:
        raise ValueError("require 0 <= k <= n and n > 0")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    if k == n:
        return 1.0
    low, high = k / n, 1.0
    for _ in range(80):
        middle = (low + high) / 2
        if _binomial_cdf(k, n, middle) > alpha:
            low = middle
        else:
            high = middle
    return high


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm-adjusted p-values in the original order."""

    indexed = sorted(enumerate(p_values), key=lambda value: value[1])
    adjusted = [0.0] * len(indexed)
    running = 0.0
    total = len(indexed)
    for rank, (original_index, p_value) in enumerate(indexed):
        running = max(running, min(1.0, (total - rank) * p_value))
        adjusted[original_index] = running
    return adjusted


def _bootstrap_cost_saving(
    observations,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    by_pair = defaultdict(list)
    for value in observations:
        by_pair[value.pair_id].append(value)
    pair_ids = sorted(by_pair)

    def statistic(values) -> float:
        baseline = sum((value.baseline_list_cost for value in values), Decimal(0))
        saving = sum((value.net_saving for value in values), Decimal(0))
        return float(saving / baseline)

    estimate = statistic(observations)
    rng = random.Random(seed)
    samples = []
    for _ in range(replicates):
        selected = [rng.choice(pair_ids) for _ in pair_ids]
        samples.append(
            statistic([value for pair_id in selected for value in by_pair[pair_id]])
        )
    samples.sort()

    def percentile(probability: float) -> float:
        position = (len(samples) - 1) * probability
        lower, upper = math.floor(position), math.ceil(position)
        if lower == upper:
            return samples[lower]
        weight = position - lower
        return samples[lower] * (1 - weight) + samples[upper] * weight

    return {
        "estimate": estimate,
        "lower": percentile(0.025),
        "upper": percentile(0.975),
    }


def analyze_ratio_sweep(
    baseline_records: Sequence[Mapping[str, Any]],
    sweep_records: Sequence[Mapping[str, Any]],
    *,
    success_margin: float = 0.05,
    confidence_level: float = 0.95,
    bootstrap_replicates: int = 10_000,
    seed: int = 20260918,
) -> dict[str, Any]:
    """Analyze each channel/ratio cell against the frozen repeated B0 runs."""

    if confidence_level != 0.95:
        raise ValueError("the frozen follow-up currently supports 95% confidence only")
    baselines = [
        record
        for record in baseline_records
        if record["condition"] == "B0" and record.get("analysis_valid", True)
    ]
    groups: dict[tuple[str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for record in sweep_records:
        groups[(str(record["condition"]), float(record["target_ratio"]))].append(record)
    if not groups:
        raise ValueError("ratio sweep contains no cells")

    cells = []
    raw_p_values = []
    for cell_index, ((condition, ratio), planned_values) in enumerate(sorted(groups.items())):
        success_values = []
        for record in planned_values:
            value = dict(record)
            value["analysis_valid"] = True
            value["success"] = bool(record.get("analysis_valid", True) and record["success"])
            success_values.append(value)
        paired_success_records = [*baselines, *success_values]
        mcnemar = mcnemar_exact(paired_success_records, condition)
        raw_p_values.append(mcnemar.exact_two_sided_p)

        valid_values = [
            record for record in planned_values if record.get("analysis_valid", True)
        ]
        cost_records = [*baselines, *valid_values]
        observations = build_cost_transfer_observations(cost_records, condition)
        aggregate = aggregate_cost_transfer(observations)
        cdr_interval = bootstrap_cdr(
            observations, replicates=bootstrap_replicates, seed=seed + cell_index
        )
        cost_interval = _bootstrap_cost_saving(
            observations, replicates=bootstrap_replicates, seed=seed + cell_index
        )
        failures = sum(not bool(value["success"]) for value in success_values)
        harm_upper = clopper_pearson_upper(
            failures, len(success_values), alpha=1 - confidence_level
        )
        success_rate = 1 - failures / len(success_values)
        safe = harm_upper <= success_margin and cost_interval["lower"] > 0
        cells.append(
            {
                "condition": condition,
                "target_ratio": ratio,
                "planned_run_count": len(planned_values),
                "analysis_valid_count": len(valid_values),
                "success_rate_planned": success_rate,
                "observed_success_drop_vs_perfect_b0": 1 - success_rate,
                "one_sided_95_harm_upper": harm_upper,
                "success_noninferiority_margin": success_margin,
                "list_price_saving": cost_interval,
                "cdr": {
                    "estimate": float(aggregate.cdr) if aggregate.cdr is not None else None,
                    "lower": cdr_interval.lower,
                    "upper": cdr_interval.upper,
                },
                "mcnemar": asdict(mcnemar),
                "safe_and_cost_saving": safe,
            }
        )
    for cell, adjusted in zip(cells, holm_adjust(raw_p_values), strict=True):
        cell["mcnemar"]["holm_adjusted_p"] = adjusted

    eligible = [cell for cell in cells if cell["safe_and_cost_saving"]]
    recommended = (
        max(eligible, key=lambda cell: cell["list_price_saving"]["estimate"])
        if eligible
        else None
    )
    return {
        "protocol": {
            "success_gate_uses_all_planned_runs": True,
            "accounting_invalid_is_failure_for_success_gate": True,
            "cost_and_cdr_use_analysis_valid_runs": True,
            "multiple_comparison_correction": "holm",
        },
        "cells": cells,
        "recommended_cell": (
            None
            if recommended is None
            else {
                "condition": recommended["condition"],
                "target_ratio": recommended["target_ratio"],
            }
        ),
    }


def _paired_exact(first: Mapping[str, bool], second: Mapping[str, bool]) -> dict[str, Any]:
    task_ids = sorted(set(first) & set(second))
    first_only = sum(first[task_id] and not second[task_id] for task_id in task_ids)
    second_only = sum(second[task_id] and not first[task_id] for task_id in task_ids)
    discordant = first_only + second_only
    if discordant:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(first_only, second_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2 * tail)
    else:
        p_value = 1.0
    return {
        "matched_tasks": len(task_ids),
        "first_only_success": first_only,
        "second_only_success": second_only,
        "exact_two_sided_p": p_value,
    }


def analyze_contract_guard(
    baseline_records: Sequence[Mapping[str, Any]],
    uniform_records: Sequence[Mapping[str, Any]],
    guard_records: Sequence[Mapping[str, Any]],
    *,
    target_ratio: float = 0.8,
    success_margin: float = 0.05,
    bootstrap_replicates: int = 10_000,
    seed: int = 20260919,
) -> dict[str, Any]:
    """Compare the exploratory contract guard with uniform extraction."""

    baselines = [
        record
        for record in baseline_records
        if record["condition"] == "B0" and record.get("analysis_valid", True)
    ]
    uniform = [
        record
        for record in uniform_records
        if record["condition"] == "T-H"
        and abs(float(record["target_ratio"]) - target_ratio) < 1e-12
    ]
    guard = [record for record in guard_records if record["condition"] == "T-H"]
    if len(uniform) != 80 or len(guard) != 80:
        raise ValueError("architecture comparison requires 80 uniform and 80 guard runs")

    def success_map(records):
        return {
            str(record["task_id"]): bool(
                record.get("analysis_valid", True) and record["success"]
            )
            for record in records
        }

    def summarize_method(records, name, method_seed):
        outcomes = success_map(records)
        failures = sum(not value for value in outcomes.values())
        valid = [record for record in records if record.get("analysis_valid", True)]
        observations = build_cost_transfer_observations(
            [*baselines, *valid], "T-H"
        )
        aggregate = aggregate_cost_transfer(observations)
        cost_interval = _bootstrap_cost_saving(
            observations, replicates=bootstrap_replicates, seed=method_seed
        )
        cdr_interval = bootstrap_cdr(
            observations, replicates=bootstrap_replicates, seed=method_seed
        )
        harm_upper = clopper_pearson_upper(failures, len(records))
        algorithms = sorted(
            {
                str(event["algorithm"])
                for record in records
                for event in record.get("events", [])
                if event.get("event_type") == "compression"
            }
        )
        return {
            "method": name,
            "algorithms_observed": algorithms,
            "planned_run_count": len(records),
            "analysis_valid_count": len(valid),
            "success_rate_planned": 1 - failures / len(records),
            "one_sided_95_harm_upper": harm_upper,
            "mean_realized_ratio": mean(float(record["realized_ratio"]) for record in valid),
            "list_price_saving": cost_interval,
            "cdr": {
                "estimate": float(aggregate.cdr) if aggregate.cdr is not None else None,
                "lower": cdr_interval.lower,
                "upper": cdr_interval.upper,
            },
            "safe_and_cost_saving": (
                harm_upper <= success_margin and cost_interval["lower"] > 0
            ),
        }

    uniform_summary = summarize_method(uniform, "uniform_extract", seed)
    guard_summary = summarize_method(guard, "contract_guard", seed + 1)
    return {
        "status": "exploratory_post_ratio_sweep",
        "target_ratio": target_ratio,
        "success_noninferiority_margin": success_margin,
        "methods": [uniform_summary, guard_summary],
        "paired_success_test": _paired_exact(success_map(guard), success_map(uniform)),
        "contract_guard_recommended": guard_summary["safe_and_cost_saving"],
    }
