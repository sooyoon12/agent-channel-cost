"""Append-only JSONL run logging with minimal schema enforcement."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


REQUIRED_TOP_LEVEL = frozenset(
    {
        "run_id",
        "pair_id",
        "task_id",
        "task_family",
        "sensitivity_group",
        "language",
        "condition",
        "repetition",
        "provider",
        "model",
        "temperature",
        "reasoning_effort",
        "tokenizer_name",
        "tokenizer_version",
        "target_ratio",
        "realized_ratio",
        "success",
        "failure_type",
        "turn_count",
        "tool_call_count",
        "model_cost_usd",
        "compressor_cost_usd",
        "tokens",
        "shadow",
        "events",
    }
)

REQUIRED_TOKEN_FIELDS = frozenset(
    {
        "local_i_input",
        "local_t_input",
        "protocol_residual_input",
        "provider_input",
        "provider_cached_input",
        "provider_cache_write_input",
        "assistant_natural_output",
        "assistant_tool_call_output",
        "provider_reasoning_output",
        "provider_total_output",
        "compressor_input",
        "compressor_output",
    }
)


def validate_run_record(record: Mapping[str, Any]) -> None:
    missing = REQUIRED_TOP_LEVEL - record.keys()
    if missing:
        raise ValueError(f"run record missing fields: {sorted(missing)}")
    missing_tokens = REQUIRED_TOKEN_FIELDS - record["tokens"].keys()
    if missing_tokens:
        raise ValueError(f"run record missing token fields: {sorted(missing_tokens)}")
    if record["language"] not in {"ko", "en"}:
        raise ValueError("language must be ko or en")
    if record["condition"] not in {"B0", "I-H", "T-H", "O"}:
        raise ValueError("unknown experiment condition")
    if not isinstance(record["events"], list):
        raise ValueError("events must be a list")


class JsonlRunLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, record: Mapping[str, Any]) -> None:
        validate_run_record(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
