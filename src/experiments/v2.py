"""Preregistered plan builders for the T-H v2 development and held-out runs."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from src.agent.runtime import ExperimentCondition
from src.tasks.generator import SemanticTask

from .runner import RunPlanEntry


@dataclass(frozen=True)
class V2MainPlanEntry:
    ordinal: int
    pair_id: str
    language: str
    condition: ExperimentCondition
    repetition: int
    target_ratio: float | None
    method: str


V2_METHODS = ("b0", "uniform", "head_tail", "query_aware_filter", "oracle_contract")
V2_FOLLOWUP_METHOD_RATIOS = (
    ("uniform", 0.6),
    ("uniform", 0.9),
    ("head_tail", 0.6),
    ("head_tail", 0.9),
    ("surface_ranked", 0.8),
    ("surface_ranked", 0.4),
    ("surface_ranked", 0.2),
)


def _build_plan(
    tasks: Sequence[SemanticTask],
    *,
    condition: ExperimentCondition,
    order_seed: int,
    target_ratio: float | None,
) -> tuple[RunPlanEntry, ...]:
    if not tasks or any(task.task_family not in {"long_observation", "event_followup"} for task in tasks):
        raise ValueError("v2 plans require nonempty T-H-active task families only")
    entries = [
        (task.pair_id, language, condition, 1, target_ratio)
        for task in tasks
        for language in ("ko", "en")
    ]
    random.Random(order_seed).shuffle(entries)
    return tuple(
        RunPlanEntry(index, pair_id, language, current_condition, repetition, ratio)
        for index, (pair_id, language, current_condition, repetition, ratio) in enumerate(entries, 1)
    )


def build_v2_development_baseline_plan(
    tasks: Sequence[SemanticTask], *, order_seed: int = 20260921
) -> tuple[RunPlanEntry, ...]:
    return _build_plan(
        tasks,
        condition=ExperimentCondition.BASELINE,
        order_seed=order_seed,
        target_ratio=None,
    )


def build_v2_heldout_baseline_plan(
    tasks: Sequence[SemanticTask], *, order_seed: int = 20261001
) -> tuple[RunPlanEntry, ...]:
    return _build_plan(
        tasks,
        condition=ExperimentCondition.BASELINE,
        order_seed=order_seed,
        target_ratio=None,
    )


def build_v2_heldout_compression_plan(
    tasks: Sequence[SemanticTask],
    *,
    target_ratio: float = 0.8,
    order_seed: int = 20261001,
) -> tuple[RunPlanEntry, ...]:
    if not 0 < target_ratio < 1:
        raise ValueError("target_ratio must lie in (0, 1)")
    return _build_plan(
        tasks,
        condition=ExperimentCondition.TOOL_OBSERVATION,
        order_seed=order_seed,
        target_ratio=target_ratio,
    )


def build_v2_heldout_main_plan(
    tasks: Sequence[SemanticTask],
    *,
    target_ratio: float = 0.8,
    order_seed: int = 20261001,
) -> tuple[V2MainPlanEntry, ...]:
    """Interleave B0 and all comparison methods in one randomized plan."""

    if not tasks or any(task.task_family not in {"long_observation", "event_followup"} for task in tasks):
        raise ValueError("v2 plans require nonempty T-H-active task families only")
    entries = []
    for task in tasks:
        for language in ("ko", "en"):
            for method in V2_METHODS:
                condition = (
                    ExperimentCondition.BASELINE
                    if method == "b0"
                    else ExperimentCondition.TOOL_OBSERVATION
                )
                entries.append((task.pair_id, language, condition, 1, None if method == "b0" else target_ratio, method))
    random.Random(order_seed).shuffle(entries)
    return tuple(
        V2MainPlanEntry(index, *entry)
        for index, entry in enumerate(entries, 1)
    )


def build_v2_followup_plan(
    tasks: Sequence[SemanticTask],
    *,
    method_ratios: Sequence[tuple[str, float]] = V2_FOLLOWUP_METHOD_RATIOS,
    order_seed: int = 20261006,
    include_repeat_baseline: bool = True,
) -> tuple[V2MainPlanEntry, ...]:
    """Build the post-main sweep with an interleaved concurrent B0 repeat."""

    if not tasks or any(
        task.task_family not in {"long_observation", "event_followup"} for task in tasks
    ):
        raise ValueError("v2 plans require nonempty T-H-active task families only")
    allowed = {"uniform", "head_tail", "surface_ranked"}
    cells = tuple((str(method), float(ratio)) for method, ratio in method_ratios)
    if not cells or len(set(cells)) != len(cells):
        raise ValueError("follow-up cells must be nonempty and unique")
    if any(method not in allowed or not 0 < ratio < 1 for method, ratio in cells):
        raise ValueError("invalid v2 follow-up method or ratio")
    entries = [
        (
            task.pair_id,
            language,
            ExperimentCondition.TOOL_OBSERVATION,
            1,
            ratio,
            method,
        )
        for task in tasks
        for language in ("ko", "en")
        for method, ratio in cells
    ]
    if include_repeat_baseline:
        entries.extend(
            (
                task.pair_id,
                language,
                ExperimentCondition.BASELINE,
                1,
                0.8,
                "b0_repeat",
            )
            for task in tasks
            for language in ("ko", "en")
        )
    random.Random(order_seed).shuffle(entries)
    return tuple(V2MainPlanEntry(index, *entry) for index, entry in enumerate(entries, 1))


def build_v2_followup_smoke_plan(
    tasks: Sequence[SemanticTask],
    *,
    method_ratios: Sequence[tuple[str, float]] = V2_FOLLOWUP_METHOD_RATIOS,
    order_seed: int = 20261006,
    include_repeat_baseline: bool = True,
) -> tuple[V2MainPlanEntry, ...]:
    """One Korean and English task per family/cell for wiring validation."""

    representatives = []
    for family in ("long_observation", "event_followup"):
        representatives.append(next(task for task in tasks if task.task_family == family))
    return build_v2_followup_plan(
        representatives,
        method_ratios=method_ratios,
        order_seed=order_seed,
        include_repeat_baseline=include_repeat_baseline,
    )
