"""Generate paired surfaces from shared semantic structures."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
import hashlib
import json
import random
from typing import Any, Iterable, Mapping

from .templates import render_long_observation, render_precise


@dataclass(frozen=True)
class SemanticTask:
    pair_id: str
    task_family: str
    sensitivity_group: str
    seed: int
    hidden_state: Mapping[str, Any]
    parameters: Mapping[str, Any]
    semantic_form: Mapping[str, Any]
    ko_surface: str
    en_surface: str
    expected_tool_calls: tuple[Mapping[str, Any], ...]
    allowed_equivalent_paths: tuple[tuple[Mapping[str, Any], ...], ...]
    expected_final_state: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def checksum(self) -> str:
        payload = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


PEOPLE = (
    {"id": "P01", "ko": "민지", "en": "Minji"},
    {"id": "P02", "ko": "지우", "en": "Jiwoo"},
    {"id": "P03", "ko": "서준", "en": "Seojun"},
    {"id": "P04", "ko": "하윤", "en": "Hayoon"},
    {"id": "P05", "ko": "도윤", "en": "Doyoon"},
    {"id": "P06", "ko": "수아", "en": "Sua"},
)
REGIONS = ("EAST", "WEST", "SOUTH", "NORTH")
STATUSES = ("REVIEW", "HOLD", "APPROVED")


def _event_state(index: int) -> dict[str, Any]:
    base = 100 + index
    anchor = date(2026, 10, 1) + timedelta(days=index)
    return {
        "events": [
            {
                "event_id": f"EV-{base}-A",
                "date": anchor.isoformat(),
                "attendees": ["P01", "P03"],
                "declined": ["P02"],
                "cancelled": False,
            },
            {
                "event_id": f"EV-{base}-B",
                "date": (anchor + timedelta(days=2)).isoformat(),
                "attendees": ["P01", "P02", "P03", "P04"],
                "declined": [],
                "cancelled": False,
            },
            {
                "event_id": f"EV-{base}-C",
                "date": (anchor + timedelta(days=3)).isoformat(),
                "attendees": ["P06"],
                "declined": ["P01"],
                "cancelled": True,
            },
        ],
        "deleted_event_ids": [],
        "audit_preserved": {},
    }


def _generate_precise(index: int, sensitive: bool) -> SemanticTask:
    seed = 1000 + index
    rng = random.Random(seed)
    state = _event_state(index)
    if not sensitive or index % 3 == 1:
        person = PEOPLE[0] if sensitive else rng.choice(PEOPLE[:4])
        semantic = {
            "action": "search",
            "person_id": person["id"],
            "person_ko": person["ko"],
            "person_en": person["en"],
            "attendance": "not_attending" if sensitive else "attending",
            "date_operator": "on_or_after" if sensitive else "after",
            "reference_date": (
                state["events"][2]["date"]
                if sensitive
                else (date.fromisoformat(state["events"][1]["date"]) - timedelta(days=1)).isoformat()
            ),
            "attendee_operator": "at_most" if sensitive else "at_least",
            "attendee_count": 1 if sensitive else rng.randint(2, 4),
            "include_cancelled": sensitive,
            "sort": "descending" if sensitive else rng.choice(("ascending", "descending")),
        }
        name = "search_events"
        arguments = {
            key: value
            for key, value in semantic.items()
            if key not in {"action", "person_ko", "person_en"}
        }
        expected_state: dict[str, Any] = {}
    elif index % 3 == 2:
        event_id = state["events"][1]["event_id"]
        person = rng.choice(PEOPLE)
        semantic = {
            "action": "set_attendance",
            "event_id": event_id,
            "person_id": person["id"],
            "person_ko": person["ko"],
            "person_en": person["en"],
            "status": "not_attending",
        }
        name = "set_attendance"
        arguments = {
            key: value
            for key, value in semantic.items()
            if key not in {"action", "person_ko", "person_en"}
        }
        expected_state = {}
        # Compute the exact expected event subset after the deterministic update.
        target = next(event for event in state["events"] if event["event_id"] == event_id)
        target = dict(target)
        target["attendees"] = sorted(
            person_id for person_id in target["attendees"] if person_id != semantic["person_id"]
        )
        target["declined"] = sorted(set(target["declined"] + [semantic["person_id"]]))
        expected_state = {"events": [state["events"][0], target, state["events"][2]]}
    else:
        event_id = state["events"][0]["event_id"]
        semantic = {"action": "delete", "event_id": event_id, "preserve_audit": True}
        name = "delete_event"
        arguments = {"event_id": event_id, "preserve_audit": True}
        expected_state = {
            "deleted_event_ids": [event_id],
            "audit_preserved": {event_id: True},
        }
    ko, en = render_precise(semantic)
    return SemanticTask(
        pair_id=f"precise-{index:03d}",
        task_family="precise_argument",
        sensitivity_group="compression_sensitive" if sensitive else "neutral",
        seed=seed,
        hidden_state=state,
        parameters={"generator_version": 1},
        semantic_form=semantic,
        ko_surface=ko,
        en_surface=en,
        expected_tool_calls=({"name": name, "arguments": arguments},),
        allowed_equivalent_paths=(),
        expected_final_state=expected_state,
    )


def _make_records(seed: int, count: int = 36) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    target_position = rng.randrange(count)
    target = {
        "record_id": f"REC-{seed}-{target_position:02d}",
        "region": REGIONS[seed % len(REGIONS)],
        "status": STATUSES[seed % len(STATUSES)],
        "amount": 9000 + seed % 500,
        "date": "2026-11-18",
        "evidence_code": f"EV{seed:04d}X{target_position:02d}",
        "notes": "target evidence with all required fields",
    }
    records: list[dict[str, Any]] = []
    for position in range(count):
        if position == target_position:
            records.append(target)
            continue
        # Each distractor violates at least one predicate, while many remain
        # close enough that record_id or evidence_code cannot be guessed.
        violated = position % 4
        record = {
            "record_id": f"REC-{seed}-{position:02d}",
            "region": target["region"] if violated != 0 else REGIONS[(seed + 1) % len(REGIONS)],
            "status": target["status"] if violated != 1 else STATUSES[(seed + 1) % len(STATUSES)],
            "amount": target["amount"] + rng.randrange(0, 800) if violated != 2 else target["amount"] - 1,
            "date": "2026-11-18" if violated != 3 else "2026-11-17",
            "evidence_code": f"EV{seed:04d}D{position:02d}",
            "notes": "synthetic distractor record " + ("detail " * (3 + position % 5)).strip(),
        }
        records.append(record)
    return records, target


def _generate_long(index: int, sensitive: bool) -> SemanticTask:
    seed = 2000 + index
    records, target = _make_records(seed)
    dataset_id = f"DS-{seed}"
    semantic = {"dataset_id": dataset_id, "region": target["region"], "status": target["status"]}
    if sensitive:
        semantic.update({"minimum_amount": target["amount"], "on_or_after": target["date"]})
    else:
        # Neutral tasks use two positive predicates. Preserve the same long
        # observation shape while making their answer unique without a boundary.
        for record in records:
            if (
                record["record_id"] != target["record_id"]
                and record["region"] == target["region"]
                and record["status"] == target["status"]
            ):
                record["region"] = REGIONS[(REGIONS.index(target["region"]) + 1) % len(REGIONS)]
    state = {"dataset_id": dataset_id, "records": records, "submission": None}
    submission = {"record_id": target["record_id"], "evidence_code": target["evidence_code"]}
    ko, en = render_long_observation(semantic)
    return SemanticTask(
        pair_id=f"long-{index:03d}",
        task_family="long_observation",
        sensitivity_group="compression_sensitive" if sensitive else "neutral",
        seed=seed,
        hidden_state=state,
        parameters={"generator_version": 1, "record_count": len(records)},
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


def generate_task_set(pairs_per_family: int = 20) -> tuple[SemanticTask, ...]:
    if pairs_per_family <= 0 or pairs_per_family % 2:
        raise ValueError("pairs_per_family must be a positive even number")
    half = pairs_per_family // 2
    tasks: list[SemanticTask] = []
    for index in range(1, pairs_per_family + 1):
        tasks.append(_generate_precise(index, sensitive=index > half))
    for index in range(1, pairs_per_family + 1):
        tasks.append(_generate_long(index, sensitive=index > half))
    return tuple(tasks)


def generate_preview() -> tuple[SemanticTask, ...]:
    """Three review pairs per family without declaring the dataset frozen."""

    all_tasks = generate_task_set(20)
    selected_ids = {"precise-001", "precise-011", "precise-012", "long-001", "long-011", "long-012"}
    return tuple(task for task in all_tasks if task.pair_id in selected_ids)


def task_set_checksum(tasks: Iterable[SemanticTask]) -> str:
    lines = [
        json.dumps(task.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for task in tasks
    ]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
