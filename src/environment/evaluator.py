"""Deterministic task evaluation without an LLM judge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .tools import ToolExecution


@dataclass(frozen=True)
class EvaluationResult:
    success: bool
    failure_type: str | None
    details: str | None


def _is_subset(expected: Any, actual: Any) -> bool:
    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and all(
            key in actual and _is_subset(value, actual[key]) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return expected == actual
    return expected == actual


def _call_shape(execution: ToolExecution) -> dict[str, Any]:
    return {"name": execution.name, "arguments": execution.arguments}


def _path_matches(expected: Sequence[Mapping[str, Any]], actual: Sequence[ToolExecution]) -> bool:
    if len(expected) != len(actual):
        return False
    return all(_is_subset(wanted, _call_shape(observed)) for wanted, observed in zip(expected, actual))


def evaluate_trajectory(
    executions: Iterable[ToolExecution],
    final_state: Mapping[str, Any],
    *,
    expected_tool_calls: Sequence[Mapping[str, Any]],
    allowed_equivalent_paths: Sequence[Sequence[Mapping[str, Any]]],
    expected_final_state: Mapping[str, Any],
    turn_count: int,
    max_turns: int = 6,
    max_identical_calls: int = 2,
    provider_error: bool = False,
) -> EvaluationResult:
    calls = tuple(executions)
    if provider_error:
        return EvaluationResult(False, "provider_error", "provider failure is not research behavior")
    if turn_count > max_turns:
        return EvaluationResult(False, "max_turn/budget", f"turn count {turn_count} exceeds {max_turns}")
    for call in calls:
        if call.error_code == "schema_error":
            return EvaluationResult(False, "schema_error", f"call {call.sequence} failed schema validation")
    frequencies: dict[tuple[str, str], int] = {}
    for call in calls:
        key = (call.name, call.argument_hash)
        frequencies[key] = frequencies.get(key, 0) + 1
        if frequencies[key] > max_identical_calls:
            return EvaluationResult(False, "repeated_call", f"identical call repeated {frequencies[key]} times")
    paths = [expected_tool_calls, *allowed_equivalent_paths]
    if not calls and expected_tool_calls:
        return EvaluationResult(False, "premature_stop", "no required tool call was made")
    if not any(_path_matches(path, calls) for path in paths):
        expected_names = {call["name"] for path in paths for call in path}
        if any(call.name not in expected_names for call in calls):
            return EvaluationResult(False, "wrong_tool", "trajectory contains an unexpected tool")
        return EvaluationResult(False, "argument_error", "tool arguments or call sequence differ from expected")
    if any(call.error_code for call in calls):
        return EvaluationResult(False, "argument_error", "a required tool returned a structured error")
    if not _is_subset(expected_final_state, final_state):
        return EvaluationResult(False, "observation_omission", "expected final state was not reached")
    return EvaluationResult(True, None, None)

