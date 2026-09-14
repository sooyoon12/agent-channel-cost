"""Deterministic, network-free tools used by synthetic tasks."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import time
from typing import Any, Callable, Mapping


JsonObject = dict[str, Any]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class ToolExecution:
    sequence: int
    name: str
    arguments: JsonObject
    argument_hash: str
    result: JsonObject
    error_code: str | None
    latency_ms: float


def _object_schema(properties: Mapping[str, Any], required: list[str]) -> JsonObject:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": required,
        "additionalProperties": False,
    }


TOOL_SPECS: dict[str, ToolSpec] = {
    "search_events": ToolSpec(
        "search_events",
        "Search the isolated synthetic schedule using exact filters.",
        _object_schema(
            {
                "person_id": {"type": "string"},
                "attendance": {"type": "string", "enum": ["attending", "not_attending"]},
                "date_operator": {
                    "type": "string",
                    "enum": ["before", "after", "on_or_before", "on_or_after"],
                },
                "reference_date": {"type": "string"},
                "attendee_operator": {"type": "string", "enum": ["at_least", "at_most"]},
                "attendee_count": {"type": "integer", "minimum": 0},
                "include_cancelled": {"type": "boolean"},
                "sort": {"type": "string", "enum": ["ascending", "descending"]},
            },
            [
                "person_id",
                "attendance",
                "date_operator",
                "reference_date",
                "attendee_operator",
                "attendee_count",
                "include_cancelled",
                "sort",
            ],
        ),
    ),
    "set_attendance": ToolSpec(
        "set_attendance",
        "Set one person's attendance state for one event.",
        _object_schema(
            {
                "event_id": {"type": "string"},
                "person_id": {"type": "string"},
                "status": {"type": "string", "enum": ["attending", "not_attending"]},
            },
            ["event_id", "person_id", "status"],
        ),
    ),
    "delete_event": ToolSpec(
        "delete_event",
        "Delete one event while explicitly choosing whether to preserve its audit record.",
        _object_schema(
            {
                "event_id": {"type": "string"},
                "preserve_audit": {"type": "boolean"},
            },
            ["event_id", "preserve_audit"],
        ),
    ),
    "read_records": ToolSpec(
        "read_records",
        "Return the deterministic long observation for one synthetic dataset.",
        _object_schema({"dataset_id": {"type": "string"}}, ["dataset_id"]),
    ),
    "submit_evidence": ToolSpec(
        "submit_evidence",
        "Submit the record id and evidence code selected from a tool observation.",
        _object_schema(
            {
                "record_id": {"type": "string"},
                "evidence_code": {"type": "string"},
            },
            ["record_id", "evidence_code"],
        ),
    ),
}


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _error(code: str, message: str, *, details: Any = None) -> JsonObject:
    value: JsonObject = {"ok": False, "error": {"code": code, "message": message}}
    if details is not None:
        value["error"]["details"] = details
    return value


def _is_type(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, list)
    return False


def validate_arguments(arguments: Any, schema: Mapping[str, Any]) -> list[JsonObject]:
    """Validate the strict schema subset used by this experiment."""

    errors: list[JsonObject] = []
    if not isinstance(arguments, Mapping):
        return [{"path": "$", "message": "expected object"}]
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    for key in required:
        if key not in arguments:
            errors.append({"path": f"$.{key}", "message": "required property is missing"})
    if schema.get("additionalProperties") is False:
        for key in arguments:
            if key not in properties:
                errors.append({"path": f"$.{key}", "message": "additional property is not allowed"})
    for key, value in arguments.items():
        field = properties.get(key)
        if field is None:
            continue
        expected = field.get("type")
        if expected and not _is_type(value, expected):
            errors.append({"path": f"$.{key}", "message": f"expected {expected}"})
            continue
        if "enum" in field and value not in field["enum"]:
            errors.append({"path": f"$.{key}", "message": "value is not in enum"})
        if "minimum" in field and isinstance(value, int) and value < field["minimum"]:
            errors.append({"path": f"$.{key}", "message": f"must be >= {field['minimum']}"})
    return errors


class DeterministicToolEnvironment:
    """A fresh instance is one isolated run.

    All data is copied at construction and no wall clock, random generator,
    network, or process-global state is consulted during tool execution.
    """

    def __init__(self, hidden_state: Mapping[str, Any]) -> None:
        self._initial_state: JsonObject = deepcopy(dict(hidden_state))
        self._state: JsonObject = deepcopy(dict(hidden_state))
        self._executions: list[ToolExecution] = []
        self._handlers: dict[str, Callable[[Mapping[str, Any]], JsonObject]] = {
            "search_events": self._search_events,
            "set_attendance": self._set_attendance,
            "delete_event": self._delete_event,
            "read_records": self._read_records,
            "submit_evidence": self._submit_evidence,
        }

    @property
    def executions(self) -> tuple[ToolExecution, ...]:
        return tuple(self._executions)

    def initial_state(self) -> JsonObject:
        return deepcopy(self._initial_state)

    def snapshot(self) -> JsonObject:
        return deepcopy(self._state)

    def execute(self, name: str, arguments: Any) -> ToolExecution:
        started = time.perf_counter_ns()
        argument_object = deepcopy(dict(arguments)) if isinstance(arguments, Mapping) else {}
        if name not in TOOL_SPECS:
            result = _error("unknown_tool", f"unknown tool: {name}")
            return self._record(name, argument_object, result, "unknown_tool", started)
        validation_errors = validate_arguments(arguments, TOOL_SPECS[name].parameters)
        if validation_errors:
            result = _error("schema_error", "tool arguments failed schema validation", details=validation_errors)
            return self._record(name, argument_object, result, "schema_error", started)
        result = self._handlers[name](arguments)
        error = result.get("error", {}).get("code") if not result.get("ok", False) else None
        return self._record(name, argument_object, result, error, started)

    def _record(
        self,
        name: str,
        arguments: JsonObject,
        result: JsonObject,
        error_code: str | None,
        started_ns: int,
    ) -> ToolExecution:
        execution = ToolExecution(
            sequence=len(self._executions) + 1,
            name=name,
            arguments=deepcopy(arguments),
            argument_hash=_stable_hash(arguments),
            result=deepcopy(result),
            error_code=error_code,
            latency_ms=(time.perf_counter_ns() - started_ns) / 1_000_000,
        )
        self._executions.append(execution)
        return execution

    def _event(self, event_id: str) -> JsonObject | None:
        return next((event for event in self._state.get("events", []) if event["event_id"] == event_id), None)

    def _search_events(self, arguments: Mapping[str, Any]) -> JsonObject:
        reference = arguments["reference_date"]
        date_operator = arguments["date_operator"]
        attendee_operator = arguments["attendee_operator"]

        def date_matches(value: str) -> bool:
            return {
                "before": value < reference,
                "after": value > reference,
                "on_or_before": value <= reference,
                "on_or_after": value >= reference,
            }[date_operator]

        def count_matches(value: int) -> bool:
            if attendee_operator == "at_least":
                return value >= arguments["attendee_count"]
            return value <= arguments["attendee_count"]

        matches = []
        deleted = set(self._state.get("deleted_event_ids", []))
        for event in self._state.get("events", []):
            if event["event_id"] in deleted:
                continue
            attends = arguments["person_id"] in event.get("attendees", [])
            attendance_matches = attends if arguments["attendance"] == "attending" else not attends
            if not arguments["include_cancelled"] and event.get("cancelled", False):
                continue
            if attendance_matches and date_matches(event["date"]) and count_matches(len(event.get("attendees", []))):
                matches.append(deepcopy(event))
        reverse = arguments["sort"] == "descending"
        matches.sort(key=lambda event: (event["date"], event["event_id"]), reverse=reverse)
        return {"ok": True, "events": matches, "count": len(matches)}

    def _set_attendance(self, arguments: Mapping[str, Any]) -> JsonObject:
        event = self._event(arguments["event_id"])
        if event is None or arguments["event_id"] in self._state.get("deleted_event_ids", []):
            return _error("not_found", "event does not exist")
        person = arguments["person_id"]
        attendees = list(event.get("attendees", []))
        declined = list(event.get("declined", []))
        if arguments["status"] == "attending":
            if person not in attendees:
                attendees.append(person)
            declined = [value for value in declined if value != person]
        else:
            attendees = [value for value in attendees if value != person]
            if person not in declined:
                declined.append(person)
        event["attendees"] = sorted(attendees)
        event["declined"] = sorted(declined)
        return {
            "ok": True,
            "event_id": event["event_id"],
            "person_id": person,
            "status": arguments["status"],
        }

    def _delete_event(self, arguments: Mapping[str, Any]) -> JsonObject:
        if self._event(arguments["event_id"]) is None:
            return _error("not_found", "event does not exist")
        deleted = list(self._state.get("deleted_event_ids", []))
        if arguments["event_id"] not in deleted:
            deleted.append(arguments["event_id"])
        self._state["deleted_event_ids"] = sorted(deleted)
        audit = dict(self._state.get("audit_preserved", {}))
        audit[arguments["event_id"]] = arguments["preserve_audit"]
        self._state["audit_preserved"] = audit
        return {
            "ok": True,
            "event_id": arguments["event_id"],
            "deleted": True,
            "audit_preserved": arguments["preserve_audit"],
        }

    def _read_records(self, arguments: Mapping[str, Any]) -> JsonObject:
        if arguments["dataset_id"] != self._state.get("dataset_id"):
            return _error("not_found", "dataset does not exist")
        return {
            "ok": True,
            "dataset_id": arguments["dataset_id"],
            "records": deepcopy(self._state.get("records", [])),
        }

    def _submit_evidence(self, arguments: Mapping[str, Any]) -> JsonObject:
        record = next(
            (value for value in self._state.get("records", []) if value["record_id"] == arguments["record_id"]),
            None,
        )
        if record is None:
            return _error("not_found", "record does not exist")
        if record["evidence_code"] != arguments["evidence_code"]:
            return _error("argument_error", "evidence code does not match the record")
        submission = {
            "record_id": arguments["record_id"],
            "evidence_code": arguments["evidence_code"],
        }
        self._state["submission"] = submission
        return {"ok": True, "submission": deepcopy(submission)}
