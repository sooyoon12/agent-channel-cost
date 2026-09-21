"""Development-gate and payload diagnostics for the v2 study."""

from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

from src.tasks.generator import SemanticTask


def records_sha256(records: Iterable[Mapping[str, Any]]) -> str:
    lines = [json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for record in records]
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def development_baseline_report(
    records: Sequence[Mapping[str, Any]],
    tasks: Sequence[SemanticTask],
    *,
    baseline_floor: float = 0.90,
    ceiling_threshold: float = 0.99,
) -> dict[str, Any]:
    if not 0 <= baseline_floor < ceiling_threshold <= 1:
        raise ValueError("invalid development thresholds")
    expected = {(task.pair_id, language) for task in tasks for language in ("ko", "en")}
    terminal = [record for record in records if record.get("failure_type") != "provider_error"]
    observed = [(record.get("pair_id"), record.get("language")) for record in terminal]
    duplicates = sorted(key for key, count in Counter(observed).items() if count > 1)
    missing = sorted(expected - set(observed))
    unexpected = sorted(set(observed) - expected)
    invalid = sum(not bool(record.get("analysis_valid", True)) for record in terminal)
    checksum_mismatch = sum(
        record.get("task_checksum") != next(
            (task.checksum() for task in tasks if task.pair_id == record.get("pair_id")), None
        )
        for record in terminal
    )
    successes = sum(bool(record.get("success")) for record in terminal)
    total = len(terminal)
    success_rate = successes / total if total else 0.0

    groups: dict[tuple[str, str], list[bool]] = defaultdict(list)
    by_task = {task.pair_id: task for task in tasks}
    for record in terminal:
        task = by_task.get(str(record.get("pair_id")))
        if task is not None:
            groups[(task.task_family, str(record.get("language")))].append(bool(record.get("success")))
    breakdown = {
        f"{family}:{language}": {
            "successes": sum(values),
            "runs": len(values),
            "success_rate": sum(values) / len(values),
        }
        for (family, language), values in sorted(groups.items())
    }
    cost = sum(
        Decimal(str(record.get("model_cost_usd", {}).get("billed", 0)))
        + Decimal(str(record.get("compressor_cost_usd", {}).get("billed", 0)))
        for record in records
    )
    reasons: list[str] = []
    if missing or unexpected or duplicates:
        reasons.append("coverage_mismatch")
    if invalid:
        reasons.append("analysis_invalid_rows")
    if checksum_mismatch:
        reasons.append("task_checksum_mismatch")
    if success_rate < baseline_floor:
        reasons.append("baseline_below_floor")
    if success_rate >= ceiling_threshold:
        reasons.append("baseline_ceiling")
    return {
        "status": "pass" if not reasons else "blocked",
        "reasons": reasons,
        "preregistered_gate": {
            "baseline_floor_inclusive": baseline_floor,
            "ceiling_threshold_exclusive": ceiling_threshold,
            "unit_note": "descriptive run-level gate; inference clusters by pair_id",
        },
        "expected_runs": len(expected),
        "terminal_runs": total,
        "successes": successes,
        "success_rate": success_rate,
        "analysis_invalid_rows": invalid,
        "task_checksum_mismatch_rows": checksum_mismatch,
        "accounting_warning_rows": sum(bool(record.get("accounting_warning_count")) for record in terminal),
        "missing": missing,
        "unexpected": unexpected,
        "duplicates": duplicates,
        "breakdown": breakdown,
        "billed_cost_usd": float(cost),
        "raw_records_sha256": records_sha256(records),
    }


def non_contract_payload_fraction(value: Any, tokenizer: Any, *, key: str = "notes") -> float:
    """Token share attributable to a declared non-contract field."""

    original = tokenizer.count_json(value)

    def drop(item: Any) -> Any:
        if isinstance(item, dict):
            return {name: drop(child) for name, child in item.items() if name != key}
        if isinstance(item, list):
            return [drop(child) for child in item]
        return item

    without = tokenizer.count_json(drop(value))
    return (original - without) / original if original else 0.0
