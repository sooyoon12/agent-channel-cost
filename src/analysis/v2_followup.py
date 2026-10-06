"""Exploratory v2 follow-up analysis with a concurrent repeated baseline."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import random
import statistics
from typing import Any, Mapping, Sequence

from src.experiments.v2 import V2_FOLLOWUP_METHOD_RATIOS
from src.tasks.generator import SemanticTask


def _cost(record: Mapping[str, Any], field: str = "list_price") -> float:
    return float(record["model_cost_usd"][field]) + float(record["compressor_cost_usd"][field])


def _ratio(value: Any) -> float | None:
    return None if value is None else float(value)


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _mcnemar(first_only: int, second_only: int) -> float:
    total = first_only + second_only
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, value) for value in range(min(first_only, second_only) + 1))
    return min(1.0, 2 * tail / (2**total))


def _holm(values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, value * (len(ordered) - rank)))
        adjusted[name] = running
    return adjusted


def _cell_label(method: str, ratio: float) -> str:
    return f"{method}@{ratio:.1f}"


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    successes = sum(bool(row.get("success")) for row in rows)
    cost = sum(_cost(row) for row in rows)
    return {
        "runs": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows) if rows else None,
        "mean_list_price_cost_per_run_usd": cost / len(rows) if rows else None,
        "list_price_cost_per_success_usd": cost / successes if successes else None,
        "mean_provider_input_tokens": statistics.fmean(
            float(row["tokens"]["provider_input"]) for row in rows
        ) if rows else None,
        "mean_provider_output_tokens": statistics.fmean(
            float(row["tokens"]["provider_total_output"]) for row in rows
        ) if rows else None,
    }


def _planned_outcomes(
    records: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Return one research outcome per plan ordinal.

    Retryable provider errors preceding a terminal result are attempts, not
    research rows. If every attempt for an ordinal is a provider error, the
    final attempt is retained as that planned run's failure.
    """

    by_ordinal: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    missing_ordinal: list[Mapping[str, Any]] = []
    for row in records:
        ordinal = row.get("plan_ordinal")
        if isinstance(ordinal, int):
            by_ordinal[ordinal].append(row)
        else:
            missing_ordinal.append(row)
    selected: list[Mapping[str, Any]] = list(missing_ordinal)
    for ordinal in sorted(by_ordinal):
        attempts = by_ordinal[ordinal]
        terminal = [row for row in attempts if row.get("failure_type") != "provider_error"]
        if terminal:
            # Retain multiple terminal rows so the duplicate-key gate fails
            # closed instead of silently choosing one.
            selected.extend(terminal)
        else:
            failed = dict(attempts[-1])
            failed["success"] = False
            failed["analysis_valid"] = False
            selected.append(failed)
    return selected


def summarize_v2_followup(
    main_records: Sequence[Mapping[str, Any]],
    followup_records: Sequence[Mapping[str, Any]],
    tasks: Sequence[SemanticTask],
    *,
    method_ratios: Sequence[tuple[str, float]] = V2_FOLLOWUP_METHOD_RATIOS,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 20261006,
) -> dict[str, Any]:
    """Validate seven compression cells and compare with concurrent B0."""

    historical_rows = [
        row for row in main_records
        if row.get("experiment_method") == "b0" and row.get("failure_type") != "provider_error"
    ]
    terminal = _planned_outcomes(followup_records)
    task_by_id = {task.pair_id: task for task in tasks}
    pair_ids = sorted(task_by_id)
    expected = {
        (task.pair_id, language, method, float(ratio))
        for task in tasks for language in ("ko", "en") for method, ratio in method_ratios
    }
    expected.update(
        (task.pair_id, language, "b0_repeat", 0.8)
        for task in tasks for language in ("ko", "en")
    )
    observed = [
        (
            str(row.get("pair_id")), str(row.get("language")),
            str(row.get("experiment_method")), _ratio(row.get("target_ratio")),
        )
        for row in terminal
    ]
    duplicates = sorted(key for key, count in Counter(observed).items() if count > 1)
    missing = sorted(expected - set(observed))
    unexpected = sorted(set(observed) - expected)
    checksum_mismatch = [
        row for row in terminal
        if str(row.get("pair_id")) in task_by_id
        and row.get("task_checksum") != task_by_id[str(row["pair_id"])].checksum()
    ]
    if missing or unexpected or duplicates or checksum_mismatch:
        raise ValueError(
            "follow-up completeness validation failed: "
            f"missing={len(missing)}, unexpected={len(unexpected)}, "
            f"duplicates={len(duplicates)}, checksum_mismatch={len(checksum_mismatch)}"
        )
    baseline_rows = [row for row in terminal if row.get("experiment_method") == "b0_repeat"]
    baseline_by_key = {
        (str(row["pair_id"]), str(row["language"])): row for row in baseline_rows
    }
    historical_by_key = {
        (str(row["pair_id"]), str(row["language"])): row for row in historical_rows
    }
    expected_baseline = {(pair_id, language) for pair_id in pair_ids for language in ("ko", "en")}
    if expected_baseline - set(baseline_by_key):
        raise ValueError("concurrent B0 is incomplete")
    if expected_baseline - set(historical_by_key):
        raise ValueError("historical frozen B0 is incomplete")

    cells: dict[str, Any] = {}
    raw_p: dict[str, float] = {}
    for cell_index, (method, ratio) in enumerate(method_ratios):
        label = _cell_label(method, ratio)
        rows = [
            row for row in terminal
            if row.get("experiment_method") == method and _ratio(row.get("target_ratio")) == ratio
        ]
        by_pair: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            by_pair[str(row["pair_id"])].append(row)
        successes = sum(bool(row.get("success")) for row in rows)
        total_cost = sum(_cost(row) for row in rows)
        baseline_cost = sum(
            _cost(baseline_by_key[(str(row["pair_id"]), str(row["language"]))]) for row in rows
        )
        baseline_successes = sum(
            bool(baseline_by_key[(str(row["pair_id"]), str(row["language"]))]["success"])
            for row in rows
        )

        def statistic(sampled: Sequence[str]) -> float:
            current = [row for pair_id in sampled for row in by_pair[pair_id]]
            base = [baseline_by_key[(pair_id, language)] for pair_id in sampled for language in ("ko", "en")]
            current_success = sum(bool(row["success"]) for row in current)
            base_success = sum(bool(row["success"]) for row in base)
            if not current_success or not base_success:
                return float("nan")
            return 100 * (
                (sum(_cost(row) for row in current) / current_success)
                / (sum(_cost(row) for row in base) / base_success) - 1
            )

        rng = random.Random(bootstrap_seed + cell_index)
        samples = [statistic(rng.choices(pair_ids, k=len(pair_ids))) for _ in range(bootstrap_replicates)]
        samples = [value for value in samples if math.isfinite(value)]
        base_only = method_only = 0
        for pair_id in pair_ids:
            base_ok = all(baseline_by_key[(pair_id, language)]["success"] for language in ("ko", "en"))
            current = by_pair.get(pair_id, [])
            method_ok = len(current) == 2 and all(row["success"] for row in current)
            base_only += int(base_ok and not method_ok)
            method_only += int(not base_ok and method_ok)
        raw_p[label] = _mcnemar(base_only, method_only)
        cps = total_cost / successes if successes else None
        baseline_cps = baseline_cost / baseline_successes if baseline_successes else None
        cells[label] = {
            "method": method,
            "target_ratio": ratio,
            "runs": len(rows),
            "analysis_invalid_rows": sum(not row.get("analysis_valid", True) for row in rows),
            "successes": successes,
            "success_rate": successes / len(rows) if rows else None,
            "mean_realized_ratio": statistics.fmean(float(row["realized_ratio"]) for row in rows) if rows else None,
            "mean_list_price_cost_per_run_usd": total_cost / len(rows) if rows else None,
            "list_price_cost_change_vs_b0_percent": 100 * (total_cost / baseline_cost - 1) if baseline_cost else None,
            "list_price_cost_per_success_usd": cps,
            "cost_per_success_change_vs_b0_percent": 100 * (cps / baseline_cps - 1) if cps is not None and baseline_cps else None,
            "cost_per_success_change_pair_clustered_95ci_percent": [
                _percentile(samples, 0.025), _percentile(samples, 0.975)
            ] if samples else [None, None],
            "strict_pair_b0_only_success": base_only,
            "strict_pair_method_only_success": method_only,
            "strict_pair_mcnemar_p": raw_p[label],
        }
    for label, value in _holm(raw_p).items():
        cells[label]["strict_pair_mcnemar_holm_p"] = value

    language = {}
    for field in ("local_i_input", "local_t_input", "provider_input"):
        ratios = [
            float(baseline_by_key[(pair_id, "ko")]["tokens"][field])
            / float(baseline_by_key[(pair_id, "en")]["tokens"][field])
            for pair_id in pair_ids
            if float(baseline_by_key[(pair_id, "en")]["tokens"][field])
        ]
        language[field] = {
            "ko_over_en_mean": statistics.fmean(ratios),
            "ko_over_en_median": statistics.median(ratios),
        }
    current_summary, historical_summary = _summary(baseline_rows), _summary(historical_rows)
    drift = {
        "historical_b0": historical_summary,
        "concurrent_b0_repeat": current_summary,
        "success_rate_change_percentage_points": 100 * (
            current_summary["success_rate"] - historical_summary["success_rate"]
        ),
        "mean_cost_per_run_change_percent": 100 * (
            current_summary["mean_list_price_cost_per_run_usd"]
            / historical_summary["mean_list_price_cost_per_run_usd"] - 1
        ),
        "interpretation": "Concurrent B0 is primary; historical B0 is drift diagnosis only.",
    }
    return {
        "completeness": {
            "expected_followup_rows": len(expected),
            "terminal_followup_rows": len(terminal),
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "duplicate_keys": duplicates,
            "checksum_mismatch_rows": len(checksum_mismatch),
        },
        "status": "exploratory_post_main",
        "primary_baseline": "concurrent b0_repeat",
        "baseline_source_runs": len(baseline_rows),
        "historical_baseline_source_runs": len(historical_rows),
        "baseline_drift_diagnostic": drift,
        "cells": cells,
        "baseline_language_token_ratios": language,
        "inference": {
            "bootstrap_unit": "pair_id",
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": bootstrap_seed,
            "multiplicity": "Holm across seven exploratory strict-pair comparisons",
        },
    }
