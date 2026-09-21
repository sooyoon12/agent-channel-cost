"""Unfrozen, leakage-audited candidates for the T-H follow-up study.

The v1 generator and frozen dataset remain untouched. V2 deliberately omits
the precise/I-H family: every retained task requires a tool observation to make
the next tool call, so every pair is informative for T-H compression.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
import hashlib
import random
from typing import Any, Iterable

from src.environment.evaluator import evaluate_trajectory
from src.environment.tools import DeterministicToolEnvironment

from .generator import REGIONS, STATUSES, SemanticTask
_NOTE_TEMPLATES = (
    "Auxiliary annotation class alpha for routine retention.",
    "Auxiliary annotation class beta for routine retention.",
    "Auxiliary annotation class gamma for routine retention.",
    "Auxiliary annotation class delta for routine retention.",
    "Routine processing memo group alpha with standard handling.",
    "Routine processing memo group beta with standard handling.",
    "Routine processing memo group gamma with standard handling.",
    "Routine processing memo group delta with standard handling.",
)


def _neutral_note(position: int) -> str:
    # Every template is reused, so correctness has no unique note literal/format.
    return _NOTE_TEMPLATES[position % len(_NOTE_TEMPLATES)]


def _opaque_code(namespace: str, seed: int, position: int) -> str:
    digest = hashlib.sha256(f"v2:{namespace}:{seed}:{position}".encode("ascii")).hexdigest()
    return f"EV-{digest[:12].upper()}"


def _long_task(index: int, *, sensitive: bool, record_count: int = 64) -> SemanticTask:
    """Create fresh hard-negative records without the v1 boundary shortcut."""

    seed = 5000 + index
    rng = random.Random(seed)
    target_position = rng.randrange(record_count)
    target_region = REGIONS[seed % len(REGIONS)]
    target_status = STATUSES[seed % len(STATUSES)]
    categories = ("ALPHA", "BETA", "GAMMA", "DELTA")
    priorities = ("P1", "P2", "P3", "P4")
    target_category = categories[(seed // 2) % len(categories)]
    target_priority = priorities[(seed // 3) % len(priorities)]
    target_amount = 12_000 + seed % 700
    minimum_amount = target_amount - 7
    target_risk = 40 + seed % 20
    maximum_risk = target_risk + 3
    target_date = date(2027, 2, 1) + timedelta(days=index % 20)
    on_or_after = target_date - timedelta(days=1)
    dataset_id = f"DS-V2-{seed}"

    records: list[dict[str, Any]] = []
    for position in range(record_count):
        record = {
            "record_id": f"REC-V2-{seed}-{position:02d}",
            "region": target_region,
            "status": target_status,
            "category": target_category,
            "priority": target_priority,
            "amount": target_amount + (position % 3),
            "risk_score": target_risk + (position % 3),
            "date": (target_date + timedelta(days=position % 2)).isoformat(),
            "evidence_code": _opaque_code("record", seed, position),
            "notes": _neutral_note(position),
        }
        if position != target_position:
            if sensitive:
                violated = position % 5
                if violated == 0:
                    record["region"] = REGIONS[(REGIONS.index(target_region) + 1) % len(REGIONS)]
                elif violated == 1:
                    record["status"] = STATUSES[(STATUSES.index(target_status) + 1) % len(STATUSES)]
                elif violated == 2:
                    record["amount"] = minimum_amount - 1
                elif violated == 3:
                    record["date"] = (on_or_after - timedelta(days=1)).isoformat()
                else:
                    record["risk_score"] = maximum_risk + 1
            else:
                violated = position % 4
                if violated == 0:
                    record["region"] = REGIONS[(REGIONS.index(target_region) + 1) % len(REGIONS)]
                elif violated == 1:
                    record["status"] = STATUSES[(STATUSES.index(target_status) + 1) % len(STATUSES)]
                elif violated == 2:
                    record["category"] = categories[(categories.index(target_category) + 1) % len(categories)]
                else:
                    record["priority"] = priorities[(priorities.index(target_priority) + 1) % len(priorities)]
        records.append(record)

    target = records[target_position]
    semantic: dict[str, Any] = {
        "dataset_id": dataset_id,
        "region": target_region,
        "status": target_status,
    }
    if sensitive:
        semantic.update(
            {
                "minimum_amount": minimum_amount,
                "on_or_after": on_or_after.isoformat(),
                "maximum_risk": maximum_risk,
            }
        )
    else:
        semantic.update({"category": target_category, "priority": target_priority})
    submission = {
        "record_id": target["record_id"],
        "evidence_code": target["evidence_code"],
    }
    if sensitive:
        ko = (
            f"데이터셋 {dataset_id}의 레코드를 읽고 지역={target_region}, 상태={target_status}, "
            f"금액 {minimum_amount} 이상, "
            f"날짜 {on_or_after.isoformat()} 이후(당일 포함), 위험점수 {maximum_risk} 이하를 모두 "
            "만족하는 유일한 레코드의 record_id와 evidence_code를 제출해."
        )
        en = (
            f"Read dataset {dataset_id} and submit the record_id and evidence_code of the unique "
            f"record satisfying all conditions: region={target_region}, status={target_status}, "
            f"amount at least {minimum_amount}, "
            f"date on or after {on_or_after.isoformat()}, and risk_score at most {maximum_risk}."
        )
    else:
        ko = (
            f"데이터셋 {dataset_id}의 레코드를 읽고 지역={target_region}, 상태={target_status}, "
            f"분류={target_category}, 우선순위={target_priority}를 모두 만족하는 유일한 레코드의 "
            "record_id와 evidence_code를 제출해."
        )
        en = (
            f"Read dataset {dataset_id} and submit the record_id and evidence_code of the unique "
            f"record with region={target_region}, status={target_status}, category={target_category}, "
            f"and priority={target_priority}."
        )
    return SemanticTask(
        pair_id=f"long-v2-{index:03d}",
        task_family="long_observation",
        sensitivity_group="compression_sensitive" if sensitive else "neutral",
        seed=seed,
        hidden_state={"dataset_id": dataset_id, "records": records, "submission": None},
        parameters={
            "generator_version": 2,
            "record_count": record_count,
            "boundary_offset_amount": target_amount - minimum_amount,
            "boundary_offset_days": (target_date - on_or_after).days,
            "candidate_revision": 4,
        },
        semantic_form=semantic,
        ko_surface=ko,
        en_surface=en,
        expected_tool_calls=(
            {"name": "read_records", "arguments": {"dataset_id": dataset_id}},
            {"name": "submit_evidence", "arguments": submission},
        ),
        allowed_equivalent_paths=(),
        expected_final_state={"submission": submission},
    )


def _event_followup_task(index: int) -> SemanticTask:
    """Require an event id observed in search results for the second tool call."""

    seed = 7000 + index
    rng = random.Random(seed)
    anchor = date(2027, 5, 1) + timedelta(days=index)
    reference_date = anchor.isoformat()
    target_position = rng.randrange(8)
    events: list[dict[str, Any]] = []
    for position in range(8):
        events.append(
            {
                "event_id": _opaque_code("event", seed, position),
                "date": (anchor + timedelta(days=position + 1)).isoformat(),
                "attendees": ["P02", "P03", "P04", "P06"],
                "declined": [],
                "cancelled": False,
                "room": "HALL-A" if position == target_position else "HALL-B",
                "notes": _neutral_note(position),
            }
        )
    events.extend(
        (
            {
                "event_id": _opaque_code("event", seed, 8),
                "date": (anchor - timedelta(days=1)).isoformat(),
                "attendees": ["P02", "P03", "P04", "P06"],
                "declined": [],
                "cancelled": False,
                "room": "HALL-A",
                "notes": _neutral_note(8),
            },
            {
                "event_id": _opaque_code("event", seed, 9),
                "date": (anchor + timedelta(days=10)).isoformat(),
                "attendees": ["P02", "P03", "P04", "P06"],
                "declined": [],
                "cancelled": True,
                "room": "HALL-A",
                "notes": _neutral_note(9),
            },
        )
    )
    state = {"events": events, "deleted_event_ids": [], "audit_preserved": {}}
    target = events[target_position]
    search_arguments: dict[str, Any] = {
        "person_id": "P02",
        "attendance": "attending",
        "date_operator": "after",
        "reference_date": reference_date,
        "attendee_operator": "at_least",
        "attendee_count": 3,
        "include_cancelled": False,
        "sort": "ascending",
    }
    followup_arguments = {
        "event_id": target["event_id"],
        "person_id": "P05",
        "status": "attending",
    }
    ko = (
        f"일정 도구에서 지우(P02)가 참석하며 {reference_date} 이후이고 참석자가 "
        "3명 이상인 취소되지 않은 일정을 검색해. 검색 결과 중 장소가 HALL-A인 일정에서 "
        "도윤(P05)을 참석 상태로 바꿔. 일정 ID를 추측하지 마."
    )
    en = (
        f"Search for a non-cancelled event attended by Jiwoo (P02), after {reference_date}, "
        "with at least 3 attendees. Among the returned events, select the one in room "
        "HALL-A and set Doyoon (P05) to attending. Do not guess the event id."
    )
    expected_events = deepcopy(events)
    expected_target = expected_events[target_position]
    expected_target["attendees"] = sorted([*expected_target["attendees"], "P05"])
    return SemanticTask(
        pair_id=f"event-v2-{index:03d}",
        task_family="event_followup",
        sensitivity_group="neutral" if index <= 10 else "compression_sensitive",
        seed=seed,
        hidden_state=state,
        parameters={"generator_version": 2, "search_result_count": 8},
        semantic_form={"search": search_arguments, "room": "HALL-A", "followup_person_id": "P05"},
        ko_surface=ko,
        en_surface=en,
        expected_tool_calls=(
            {"name": "search_events", "arguments": search_arguments},
            {"name": "set_attendance", "arguments": followup_arguments},
        ),
        allowed_equivalent_paths=(),
        expected_final_state={"events": expected_events},
    )


def generate_v2_candidates(
    *, long_pairs: int = 80, event_pairs: int = 20
) -> tuple[SemanticTask, ...]:
    """Generate unfrozen T-H candidates; no API outcome is used here."""

    counts = (long_pairs, event_pairs)
    if any(value <= 0 or value % 2 for value in counts):
        raise ValueError("pair counts must be positive even numbers")
    tasks: list[SemanticTask] = []
    tasks.extend(
        _long_task(index, sensitive=index > long_pairs // 2)
        for index in range(1, long_pairs + 1)
    )
    tasks.extend(_event_followup_task(index) for index in range(1, event_pairs + 1))
    return tuple(tasks)


def split_v2_candidates(
    tasks: Iterable[SemanticTask],
) -> tuple[tuple[SemanticTask, ...], tuple[SemanticTask, ...]]:
    """Reserve every fourth pair for development before observing API outcomes."""

    development: list[SemanticTask] = []
    held_out: list[SemanticTask] = []
    for task in tasks:
        index = int(task.pair_id.rsplit("-", 1)[1])
        (development if index % 4 == 0 else held_out).append(task)
    return tuple(development), tuple(held_out)


def _matches_long(record: dict[str, Any], semantic: dict[str, Any]) -> bool:
    return (
        record["region"] == semantic["region"]
        and record["status"] == semantic["status"]
        and ("category" not in semantic or record["category"] == semantic["category"])
        and ("priority" not in semantic or record["priority"] == semantic["priority"])
        and (
            "minimum_amount" not in semantic
            or (
                record["amount"] >= semantic["minimum_amount"]
                and record["date"] >= semantic["on_or_after"]
                and record["risk_score"] <= semantic["maximum_risk"]
            )
        )
    )


def audit_v2_candidates(tasks: Iterable[SemanticTask]) -> dict[str, int]:
    """Fail closed on leaks, shortcuts, broken paths, and final-state mismatch."""

    task_list = tuple(tasks)
    if len({task.pair_id for task in task_list}) != len(task_list):
        raise ValueError("duplicate pair_id")
    for task in task_list:
        if task.seed < 5000:
            raise ValueError(f"v1 seed range reused: {task.pair_id}")
        environment = DeterministicToolEnvironment(task.hidden_state)
        for call in task.expected_tool_calls:
            execution = environment.execute(call["name"], call["arguments"])
            if execution.error_code is not None:
                raise ValueError(f"gold path failed: {task.pair_id}: {execution.error_code}")
        evaluation = evaluate_trajectory(
            environment.executions,
            environment.snapshot(),
            expected_tool_calls=task.expected_tool_calls,
            allowed_equivalent_paths=task.allowed_equivalent_paths,
            expected_final_state=task.expected_final_state,
            turn_count=len(environment.executions),
        )
        if not evaluation.success:
            raise ValueError(f"gold final state failed: {task.pair_id}: {evaluation.failure_type}")

        if task.task_family == "long_observation":
            records = task.hidden_state["records"]
            notes = [record.get("notes") for record in records]
            codes = [record["evidence_code"] for record in records]
            if len(set(notes)) < 2 or any(
                marker in str(note).lower() for note in notes for marker in ("target", "distractor")
            ):
                raise ValueError(f"answer-bearing notes: {task.pair_id}")
            if len(set(codes)) != len(codes) or any(not code.startswith("EV-") for code in codes):
                raise ValueError(f"invalid evidence codes: {task.pair_id}")
            if any(
                code != _opaque_code("record", task.seed, position)
                for position, code in enumerate(codes)
            ):
                raise ValueError(f"inconsistent evidence-code format: {task.pair_id}")
            semantic = dict(task.semantic_form)
            matches = [record for record in records if _matches_long(record, semantic)]
            target_id = task.expected_final_state["submission"]["record_id"]
            if len(matches) != 1 or matches[0]["record_id"] != target_id:
                raise ValueError(f"long task has non-unique or incorrect gold record: {task.pair_id}")
            if "minimum_amount" in semantic:
                target = next(record for record in records if record["record_id"] == target_id)
                if target["amount"] == semantic["minimum_amount"] or target["date"] == semantic["on_or_after"]:
                    raise ValueError(f"target equals a boundary shortcut: {task.pair_id}")
                if sum(record["amount"] == target["amount"] for record in records) < 2:
                    raise ValueError(f"target amount is unique: {task.pair_id}")
        elif task.task_family == "event_followup":
            first_result = environment.executions[0].result
            observed = first_result["events"]
            target_id = task.expected_tool_calls[1]["arguments"]["event_id"]
            matches = [event for event in observed if event["room"] == task.semantic_form["room"]]
            if len(observed) < 2 or [event["event_id"] for event in matches] != [target_id]:
                raise ValueError(f"event follow-up lacks a unique observed id: {task.pair_id}")
            if len({event.get("notes") for event in observed}) < 2:
                raise ValueError(f"event notes are an artificial constant: {task.pair_id}")
            if target_id in task.ko_surface or target_id in task.en_surface:
                raise ValueError(f"event id leaked into prompt: {task.pair_id}")
        else:
            raise ValueError(f"unsupported v2 family: {task.task_family}")
    return {"pairs": len(task_list), "th_active_pairs": len(task_list)}
