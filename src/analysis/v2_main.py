"""Paired, pair-clustered summary for the frozen v2 held-out experiment."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import random
import statistics
from typing import Any, Callable, Mapping, Sequence

from src.experiments.v2 import V2_METHODS
from src.tasks.generator import SemanticTask
from src.environment.tools import DeterministicToolEnvironment
from src.compression.fixed_ratio import FixedRatioCompressor
from src.compression.head_tail import HeadTailCompressor


def _cost(record: Mapping[str, Any], field: str) -> float:
    model = record["model_cost_usd"]
    compressor = record["compressor_cost_usd"]
    model_value = model.get(field, model.get("billed", 0))
    compressor_value = compressor.get(field, compressor.get("billed", 0))
    return float(model_value) + float(compressor_value)


def _exact_mcnemar(b: int, c: int) -> float:
    discordant = b + c
    if discordant == 0:
        return 1.0
    lower = min(b, c)
    tail = sum(math.comb(discordant, value) for value in range(lower + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def _cluster_bootstrap(
    pair_ids: Sequence[str],
    statistic: Callable[[Sequence[str]], float],
    *,
    replicates: int = 10_000,
    seed: int = 20261001,
) -> list[float]:
    rng = random.Random(seed)
    return sorted(statistic(rng.choices(pair_ids, k=len(pair_ids))) for _ in range(replicates))


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return float("nan")
    position = probability * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def _holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, value * (count - rank)))
        adjusted[name] = running
    return adjusted


def _gold_action_fields_survive(task: SemanticTask, compressed: Mapping[str, Any]) -> bool:
    if task.task_family == "long_observation":
        expected = task.expected_tool_calls[1]["arguments"]
        candidates = [
            record for record in compressed.get("records", [])
            if record.get("record_id") == expected["record_id"]
        ]
        if len(candidates) != 1:
            return False
        original = next(
            record for record in task.hidden_state["records"]
            if record["record_id"] == expected["record_id"]
        )
        decision_fields = {
            "record_id", "evidence_code", "region", "status", "category",
            "priority", "amount", "risk_score", "date",
        }
        return all(candidates[0].get(key) == original.get(key) for key in decision_fields)
    if task.task_family == "event_followup":
        expected_id = task.expected_tool_calls[1]["arguments"]["event_id"]
        candidates = [
            event for event in compressed.get("events", [])
            if event.get("event_id") == expected_id
        ]
        return (
            len(candidates) == 1
            and candidates[0].get("room") == task.semantic_form["room"]
        )
    raise ValueError(f"unsupported family: {task.task_family}")


def information_retention_report(
    records: Sequence[Mapping[str, Any]],
    tasks: Sequence[SemanticTask],
    tokenizer: Any,
    *,
    target_ratio: float = 0.8,
) -> dict[str, Any]:
    """Recompute whether the actionable gold item survives deterministic compression."""

    by_key = {
        (str(record["pair_id"]), str(record["language"]), str(record["experiment_method"])): record
        for record in records
        if record.get("failure_type") != "provider_error"
    }
    task_by_id = {task.pair_id: task for task in tasks}
    result: dict[str, Any] = {}
    for method, compressor in (
        ("uniform", FixedRatioCompressor()),
        ("head_tail", HeadTailCompressor()),
    ):
        survives: dict[str, bool] = {}
        for task in tasks:
            environment = DeterministicToolEnvironment(task.hidden_state)
            first_call = task.expected_tool_calls[0]
            observation = environment.execute(
                first_call["name"], first_call["arguments"]
            ).result
            compressed = compressor.compress(
                observation, target_ratio=target_ratio, tokenizer=tokenizer
            ).compressed
            survives[task.pair_id] = _gold_action_fields_survive(task, compressed)

        strata = {}
        for family in ("all", "long_observation", "event_followup"):
            family_tasks = [
                task for task in tasks if family == "all" or task.task_family == family
            ]
            for retained in (True, False):
                rows = [
                    by_key[(task.pair_id, language, method)]
                    for task in family_tasks
                    if survives[task.pair_id] is retained
                    for language in ("ko", "en")
                ]
                successes = sum(bool(row["success"]) for row in rows)
                strata[f"{family}:{'retained' if retained else 'removed'}"] = {
                    "runs": len(rows),
                    "successes": successes,
                    "success_rate": successes / len(rows) if rows else None,
                }

        retained_mcnemar = {}
        for language in ("ko", "en"):
            pairs = [
                (
                    bool(by_key[(task.pair_id, language, "b0")]["success"]),
                    bool(by_key[(task.pair_id, language, method)]["success"]),
                )
                for task in tasks
                if survives[task.pair_id]
            ]
            harm = sum(base and not current for base, current in pairs)
            improvement = sum(not base and current for base, current in pairs)
            retained_mcnemar[language] = {
                "pairs": len(pairs),
                "b0_success_method_failure": harm,
                "b0_failure_method_success": improvement,
                "exact_two_sided_p": _exact_mcnemar(harm, improvement),
            }
        result[method] = {
            "retained_pairs": sum(survives.values()),
            "removed_pairs": len(survives) - sum(survives.values()),
            "definition": "gold item and all action/decision fields remain exact",
            "strata": strata,
            "mcnemar_vs_b0_among_retained_by_language": retained_mcnemar,
        }
    return result


def residual_distribution_report(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Describe unallocated positive provider-input residuals without excluding rows."""

    def summarize(values: Sequence[float]) -> dict[str, float | int | None]:
        ordered = sorted(values)
        return {
            "n": len(ordered),
            "median": statistics.median(ordered) if ordered else None,
            "p95": _percentile(ordered, 0.95) if ordered else None,
            "max": max(ordered) if ordered else None,
        }

    report = {}
    for method in V2_METHODS:
        rows = [record for record in records if record.get("experiment_method") == method]
        run_tokens = [float(row["tokens"]["protocol_residual_input"]) for row in rows]
        run_fractions = [
            float(row["tokens"]["protocol_residual_input"])
            / float(row["tokens"]["provider_input"])
            for row in rows
            if row["tokens"]["provider_input"]
        ]
        warnings = [
            event
            for row in rows
            for event in row.get("events", [])
            if event.get("event_type") == "accounting_warning"
        ]
        warning_tokens = [float(event["protocol_residual_tokens"]) for event in warnings]
        warning_fractions = [
            float(event["protocol_residual_tokens"]) / float(event["provider_input_tokens"])
            for event in warnings
            if event["provider_input_tokens"]
        ]
        report[method] = {
            "runs": len(rows),
            "warning_runs": sum(bool(row.get("accounting_warning_count")) for row in rows),
            "warning_requests": len(warnings),
            "aggregate_run_residual_tokens": summarize(run_tokens),
            "aggregate_run_residual_fraction": summarize(run_fractions),
            "warning_request_residual_tokens": summarize(warning_tokens),
            "warning_request_residual_fraction": summarize(warning_fractions),
        }
    return report


def summarize_v2_main(
    records: Sequence[Mapping[str, Any]],
    tasks: Sequence[SemanticTask],
    *,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 20261001,
) -> dict[str, Any]:
    expected = {
        (task.pair_id, language, method)
        for task in tasks
        for language in ("ko", "en")
        for method in V2_METHODS
    }
    terminal = [record for record in records if record.get("failure_type") != "provider_error"]
    observed_keys = [
        (record.get("pair_id"), record.get("language"), record.get("experiment_method"))
        for record in terminal
    ]
    duplicate_keys = sorted(key for key, count in Counter(observed_keys).items() if count > 1)
    missing = sorted(expected - set(observed_keys))
    unexpected = sorted(set(observed_keys) - expected)
    invalid = [record for record in terminal if not bool(record.get("analysis_valid", True))]
    by_key = {
        (str(record["pair_id"]), str(record["language"]), str(record["experiment_method"])): record
        for record in terminal
    }
    task_by_id = {task.pair_id: task for task in tasks}
    pair_ids = sorted(task_by_id)
    checksum_mismatch = [
        record for record in terminal
        if record.get("task_checksum") != task_by_id[str(record.get("pair_id"))].checksum()
    ]

    methods: dict[str, Any] = {}
    for method in V2_METHODS:
        rows = [record for record in terminal if record.get("experiment_method") == method]
        successes = sum(bool(record.get("success")) for record in rows)
        billed = sum(
            _cost(record, "billed")
            for record in rows
        )
        list_price = sum(_cost(record, "list_price") for record in rows)
        breakdown: dict[str, dict[str, Any]] = {}
        grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for record in rows:
            grouped[(str(record["task_family"]), str(record["language"]))].append(record)
        for (family, language), values in sorted(grouped.items()):
            count = len(values)
            passed = sum(bool(value.get("success")) for value in values)
            breakdown[f"{family}:{language}"] = {
                "runs": count,
                "successes": passed,
                "success_rate": passed / count,
            }
        methods[method] = {
            "runs": len(rows),
            "successes": successes,
            "success_rate": successes / len(rows) if rows else None,
            "billed_cost_usd": billed,
            "list_price_cost_usd": list_price,
            "mean_billed_cost_per_run_usd": billed / len(rows) if rows else None,
            "mean_list_price_cost_per_run_usd": list_price / len(rows) if rows else None,
            "billed_cost_per_success_usd": billed / successes if successes else None,
            "list_price_cost_per_success_usd": list_price / successes if successes else None,
            "mean_provider_input_tokens": statistics.fmean(
                record["tokens"]["provider_input"] for record in rows
            ) if rows else None,
            "mean_local_t_input_tokens": statistics.fmean(
                record["tokens"]["local_t_input"] for record in rows
            ) if rows else None,
            "mean_realized_ratio": statistics.fmean(
                float(record["realized_ratio"]) for record in rows
            ) if rows else None,
            "accounting_warning_rows": sum(bool(record.get("accounting_warning_count")) for record in rows),
            "failure_types": dict(sorted(Counter(
                str(record.get("failure_type")) for record in rows if record.get("failure_type")
            ).items())),
            "breakdown": breakdown,
        }

        def sampled_cost_per_success(sample: Sequence[str]) -> float:
            sampled_rows = [
                by_key[(pair_id, language, method)]
                for pair_id in sample
                for language in ("ko", "en")
            ]
            sampled_successes = sum(bool(row["success"]) for row in sampled_rows)
            return sum(_cost(row, "list_price") for row in sampled_rows) / sampled_successes

        method_cps_bootstrap = _cluster_bootstrap(
            pair_ids,
            sampled_cost_per_success,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + 30 + V2_METHODS.index(method),
        )
        methods[method]["list_price_cost_per_success_pair_clustered_95ci_usd"] = [
            _percentile(method_cps_bootstrap, 0.025),
            _percentile(method_cps_bootstrap, 0.975),
        ]

    comparisons: dict[str, Any] = {}
    for method in V2_METHODS:
        if method == "b0":
            continue
        paired = [
            (by_key[(pair_id, language, "b0")], by_key[(pair_id, language, method)])
            for pair_id in pair_ids
            for language in ("ko", "en")
            if (pair_id, language, "b0") in by_key and (pair_id, language, method) in by_key
        ]
        mcnemar_by_language = {}
        for language in ("ko", "en"):
            language_pairs = [
                (by_key[(pair_id, language, "b0")], by_key[(pair_id, language, method)])
                for pair_id in pair_ids
            ]
            harm = sum(
                bool(base["success"]) and not bool(current["success"])
                for base, current in language_pairs
            )
            improvement = sum(
                not bool(base["success"]) and bool(current["success"])
                for base, current in language_pairs
            )
            mcnemar_by_language[language] = {
                "pairs": len(language_pairs),
                "b0_success_method_failure": harm,
                "b0_failure_method_success": improvement,
                "exact_two_sided_p": _exact_mcnemar(harm, improvement),
            }
        strict_pair_outcomes = [
            (
                all(bool(by_key[(pair_id, language, "b0")]["success"]) for language in ("ko", "en")),
                all(bool(by_key[(pair_id, language, method)]["success"]) for language in ("ko", "en")),
            )
            for pair_id in pair_ids
        ]
        strict_harm = sum(base and not current for base, current in strict_pair_outcomes)
        strict_improvement = sum(not base and current for base, current in strict_pair_outcomes)
        delta_by_pair = {
            pair_id: statistics.fmean(
                float(by_key[(pair_id, language, method)]["success"])
                - float(by_key[(pair_id, language, "b0")]["success"])
                for language in ("ko", "en")
            )
            for pair_id in pair_ids
        }
        cost_delta_by_pair = {
            pair_id: statistics.fmean(
                _cost(by_key[(pair_id, language, method)], "billed")
                - _cost(by_key[(pair_id, language, "b0")], "billed")
                for language in ("ko", "en")
            )
            for pair_id in pair_ids
        }
        list_cost_delta_by_pair = {
            pair_id: statistics.fmean(
                _cost(by_key[(pair_id, language, method)], "list_price")
                - _cost(by_key[(pair_id, language, "b0")], "list_price")
                for language in ("ko", "en")
            )
            for pair_id in pair_ids
        }
        method_cps = methods[method]["list_price_cost_per_success_usd"]
        baseline_cps = methods["b0"]["list_price_cost_per_success_usd"]

        def cost_per_success_for(sample: Sequence[str], current_method: str) -> float:
            sampled_rows = [
                by_key[(pair_id, language, current_method)]
                for pair_id in sample
                for language in ("ko", "en")
            ]
            successes = sum(bool(row["success"]) for row in sampled_rows)
            return sum(_cost(row, "list_price") for row in sampled_rows) / successes
        success_bootstrap = _cluster_bootstrap(
            pair_ids,
            lambda sample: statistics.fmean(delta_by_pair[pair_id] for pair_id in sample),
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        )
        cost_bootstrap = _cluster_bootstrap(
            pair_ids,
            lambda sample: statistics.fmean(cost_delta_by_pair[pair_id] for pair_id in sample),
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + 1,
        )
        list_cost_bootstrap = _cluster_bootstrap(
            pair_ids,
            lambda sample: statistics.fmean(list_cost_delta_by_pair[pair_id] for pair_id in sample),
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + 2,
        )
        cps_method_bootstrap = _cluster_bootstrap(
            pair_ids,
            lambda sample: cost_per_success_for(sample, method),
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + 3,
        )
        cps_delta_bootstrap = _cluster_bootstrap(
            pair_ids,
            lambda sample: (
                cost_per_success_for(sample, method)
                - cost_per_success_for(sample, "b0")
            ),
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + 4,
        )
        cps_change_bootstrap = _cluster_bootstrap(
            pair_ids,
            lambda sample: 100 * (
                cost_per_success_for(sample, method)
                / cost_per_success_for(sample, "b0")
                - 1
            ),
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + 5,
        )
        comparisons[method] = {
            "paired_runs": len(paired),
            "success_delta_percentage_points": 100 * statistics.fmean(delta_by_pair.values()),
            "success_delta_pair_clustered_95ci_percentage_points": [
                100 * _percentile(success_bootstrap, 0.025),
                100 * _percentile(success_bootstrap, 0.975),
            ],
            "mcnemar_exact_by_language": mcnemar_by_language,
            "mcnemar_exact_strict_bilingual_pair": {
                "pairs": len(strict_pair_outcomes),
                "definition": "success requires both Korean and English surfaces",
                "b0_success_method_failure": strict_harm,
                "b0_failure_method_success": strict_improvement,
                "exact_two_sided_p": _exact_mcnemar(strict_harm, strict_improvement),
            },
            "mean_billed_cost_delta_per_run_usd": statistics.fmean(cost_delta_by_pair.values()),
            "cost_delta_pair_clustered_95ci_usd": [
                _percentile(cost_bootstrap, 0.025),
                _percentile(cost_bootstrap, 0.975),
            ],
            "mean_list_price_cost_delta_per_run_usd_primary": statistics.fmean(
                list_cost_delta_by_pair.values()
            ),
            "list_price_cost_delta_pair_clustered_95ci_usd_primary": [
                _percentile(list_cost_bootstrap, 0.025),
                _percentile(list_cost_bootstrap, 0.975),
            ],
            "list_price_cost_per_success_usd": method_cps,
            "list_price_cost_per_success_pair_clustered_95ci_usd": [
                _percentile(cps_method_bootstrap, 0.025),
                _percentile(cps_method_bootstrap, 0.975),
            ],
            "list_price_cost_per_success_delta_vs_b0_usd": method_cps - baseline_cps,
            "list_price_cost_per_success_delta_pair_clustered_95ci_usd": [
                _percentile(cps_delta_bootstrap, 0.025),
                _percentile(cps_delta_bootstrap, 0.975),
            ],
            "list_price_cost_per_success_change_vs_b0_percent": 100 * (
                method_cps / baseline_cps - 1
            ),
            "list_price_cost_per_success_change_pair_clustered_95ci_percent": [
                _percentile(cps_change_bootstrap, 0.025),
                _percentile(cps_change_bootstrap, 0.975),
            ],
        }

    strict_p_values = {
        method: values["mcnemar_exact_strict_bilingual_pair"]["exact_two_sided_p"]
        for method, values in comparisons.items()
    }
    holm_adjusted = _holm_adjust(strict_p_values)
    for method, adjusted in holm_adjusted.items():
        comparisons[method]["mcnemar_strict_pair_holm_adjusted_p"] = adjusted

    reasons = []
    if missing or unexpected or duplicate_keys:
        reasons.append("coverage_mismatch")
    if invalid:
        reasons.append("analysis_invalid_rows")
    if checksum_mismatch:
        reasons.append("task_checksum_mismatch")
    return {
        "status": "complete" if not reasons else "blocked",
        "reasons": reasons,
        "expected_terminal_runs": len(expected),
        "terminal_runs": len(terminal),
        "raw_attempt_rows": len(records),
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "duplicate_key_count": len(duplicate_keys),
        "analysis_invalid_rows": len(invalid),
        "task_checksum_mismatch_rows": len(checksum_mismatch),
        "total_billed_cost_usd": sum(
            _cost(record, "billed")
            for record in records
        ),
        "total_list_price_cost_usd": sum(_cost(record, "list_price") for record in records),
        "methods": methods,
        "paired_comparisons_vs_b0": comparisons,
        "multiplicity": {
            "family": list(comparisons),
            "method": "Holm family-wise adjustment",
            "family_size": len(comparisons),
            "status": "specified after held-out result review; treat adjusted p-values as descriptive",
        },
        "inference": {
            "bootstrap_unit": "pair_id",
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": bootstrap_seed,
            "paired_binary_test": "exact two-sided McNemar by language and strict bilingual pair; no pooled-language independence assumption",
        },
    }
