"""Deterministic experiment plans, budget gates, and sequential execution."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
import random
from typing import Callable, Iterable, Mapping, Sequence

from src.agent.runtime import AgentRuntime, ExperimentCondition
from src.tasks.generator import SemanticTask

from .logger import JsonlRunLogger


@dataclass(frozen=True)
class RunPlanEntry:
    ordinal: int
    pair_id: str
    language: str
    condition: ExperimentCondition
    repetition: int


@dataclass(frozen=True)
class BudgetStatus:
    cumulative_usd: Decimal
    projected_total_usd: Decimal
    warning: bool
    allowed: bool
    reason: str | None


class BudgetGuard:
    def __init__(self, *, limit_usd: Decimal = Decimal("100"), warning_usd: Decimal = Decimal("70")) -> None:
        if limit_usd <= 0 or warning_usd < 0 or warning_usd > limit_usd:
            raise ValueError("invalid budget thresholds")
        self.limit_usd = limit_usd
        self.warning_usd = warning_usd

    def check(self, cumulative_usd: Decimal, projected_additional_usd: Decimal) -> BudgetStatus:
        if cumulative_usd < 0 or projected_additional_usd < 0:
            raise ValueError("costs cannot be negative")
        projected = cumulative_usd + projected_additional_usd
        allowed = projected <= self.limit_usd
        warning = cumulative_usd >= self.warning_usd or projected >= self.warning_usd
        reason = None if allowed else "projected cost exceeds the approved support limit"
        return BudgetStatus(cumulative_usd, projected, warning, allowed, reason)


def _base_entries(tasks: Sequence[SemanticTask]) -> list[tuple[str, str, ExperimentCondition, int]]:
    entries: list[tuple[str, str, ExperimentCondition, int]] = []
    for task in tasks:
        for language in ("ko", "en"):
            for repetition in (1, 2, 3):
                entries.append((task.pair_id, language, ExperimentCondition.BASELINE, repetition))
            for condition in (
                ExperimentCondition.INSTRUCTION_HISTORY,
                ExperimentCondition.TOOL_OBSERVATION,
                ExperimentCondition.CONCISE_OUTPUT,
            ):
                entries.append((task.pair_id, language, condition, 1))
    return entries


def build_main_plan(tasks: Sequence[SemanticTask], *, order_seed: int = 20260914) -> tuple[RunPlanEntry, ...]:
    if len(tasks) != 40:
        raise ValueError("main plan requires exactly 40 semantic pairs")
    entries = _base_entries(tasks)
    random.Random(order_seed).shuffle(entries)
    return tuple(
        RunPlanEntry(index, pair_id, language, condition, repetition)
        for index, (pair_id, language, condition, repetition) in enumerate(entries, 1)
    )


def build_smoke_plan(tasks: Sequence[SemanticTask], *, max_runs: int = 20) -> tuple[RunPlanEntry, ...]:
    if not 1 <= max_runs <= 20:
        raise ValueError("smoke plan is capped at 20 runs")
    selected_ids = ("precise-001", "precise-011", "long-001", "long-011")
    selected = [task for task in tasks if task.pair_id in selected_ids]
    candidates: list[tuple[str, str, ExperimentCondition, int]] = []
    # With a hard cap of 20, use one alternating language per representative
    # task. This balances family and sensitivity while covering both languages.
    selected_with_language = [
        (task, "ko" if index % 2 == 0 else "en") for index, task in enumerate(selected)
    ]
    for task, language in selected_with_language:
        for condition in ExperimentCondition:
            candidates.append((task.pair_id, language, condition, 1))
    # Four additional independent B0 repetitions expose baseline variance.
    for task, language in selected_with_language:
        candidates.append((task.pair_id, language, ExperimentCondition.BASELINE, 2))
    candidates = candidates[:max_runs]
    return tuple(
        RunPlanEntry(index, pair_id, language, condition, repetition)
        for index, (pair_id, language, condition, repetition) in enumerate(candidates, 1)
    )


def plan_checksum(plan: Iterable[RunPlanEntry]) -> str:
    values = [
        {
            "ordinal": entry.ordinal,
            "pair_id": entry.pair_id,
            "language": entry.language,
            "condition": entry.condition.value,
            "repetition": entry.repetition,
        }
        for entry in plan
    ]
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def provider_error_is_retryable(record: Mapping[str, object]) -> bool:
    events = record.get("events")
    if not isinstance(events, list):
        return False
    return any(
        isinstance(event, Mapping)
        and event.get("event_type") == "provider_error"
        and event.get("retryable") is True
        for event in events
    )


def canonical_plan_attempts(
    records: Iterable[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    """Keep retries through the first terminal attempt for each plan ordinal.

    Provider errors are retryable transport attempts. Every other outcome,
    including an accounting-invalid row, is terminal for that planned run.
    Later accidental replays remain in the raw log but are not research rows.
    """

    selected: list[Mapping[str, object]] = []
    terminal_ordinals: set[int] = set()
    for record in records:
        ordinal = record.get("plan_ordinal")
        if not isinstance(ordinal, int):
            raise ValueError("run record is missing an integer plan_ordinal")
        if ordinal in terminal_ordinals:
            continue
        selected.append(record)
        if record.get("failure_type") != "provider_error":
            terminal_ordinals.add(ordinal)
    return tuple(selected)


def execute_plan(
    plan: Iterable[RunPlanEntry],
    tasks: Sequence[SemanticTask],
    *,
    runtime_factory: Callable[[SemanticTask], AgentRuntime],
    logger: JsonlRunLogger,
    provider_error_retries: int = 1,
    budget_guard: BudgetGuard | None = None,
    projected_cost_per_run_usd: Decimal | None = None,
    starting_cost_usd: Decimal = Decimal("0"),
) -> tuple[Mapping[str, object], ...]:
    if provider_error_retries < 0:
        raise ValueError("provider_error_retries cannot be negative")
    if budget_guard is not None and projected_cost_per_run_usd is None:
        raise ValueError("budgeted execution requires projected_cost_per_run_usd")
    by_id = {task.pair_id: task for task in tasks}
    records: list[Mapping[str, object]] = []
    cumulative = starting_cost_usd
    for entry in plan:
        task = by_id[entry.pair_id]
        if budget_guard is not None:
            status = budget_guard.check(cumulative, projected_cost_per_run_usd or Decimal(0))
            if not status.allowed:
                raise RuntimeError(status.reason)
        for attempt in range(provider_error_retries + 1):
            runtime = runtime_factory(task)
            result = runtime.run(
                task,
                language=entry.language,
                condition=entry.condition,
                repetition=entry.repetition,
            )
            record = dict(result.record)
            record["plan_ordinal"] = entry.ordinal
            record["provider_attempt"] = attempt + 1
            record["analysis_valid"] = record.get("failure_type") not in {
                "provider_error",
                "accounting_invalid",
            }
            logger.append(record)
            cumulative += Decimal(str(record["model_cost_usd"]["billed"])) + Decimal(
                str(record["compressor_cost_usd"]["billed"])
            )
            if (
                record["failure_type"] == "provider_error"
                and provider_error_is_retryable(record)
                and attempt < provider_error_retries
            ):
                continue
            if record["failure_type"] == "provider_error":
                attempts = attempt + 1
                raise RuntimeError(
                    f"provider failed after {attempts} attempt(s) at plan ordinal {entry.ordinal}"
                )
            records.append(record)
            break
    return tuple(records)
