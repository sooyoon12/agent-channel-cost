"""Small strict loader for the mapping-only YAML used by experiment configs."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any


_INTEGER = re.compile(r"^[+-]?\d+$")
_FLOAT = re.compile(r"^[+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][+-]?\d+)$|^[+-]?(?:\d+\.\d*|\d*\.\d+)$")


def _scalar(value: str) -> Any:
    value = value.strip()
    if value in {"null", "~"}:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if _INTEGER.fullmatch(value):
        return int(value)
    if _FLOAT.fullmatch(value):
        return float(value)
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def load_mapping_yaml(path: str | Path) -> dict[str, Any]:
    """Load nested mappings and scalars; reject lists and malformed indentation."""

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if "\t" in raw_line[: len(raw_line) - len(raw_line.lstrip())]:
            raise ValueError(f"tabs are not allowed in YAML indentation at line {number}")
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if indent % 2:
            raise ValueError(f"indentation must use multiples of two at line {number}")
        content = raw_line.strip()
        if content.startswith("-") or ":" not in content:
            raise ValueError(f"only mapping YAML is supported at line {number}")
        key, raw_value = content.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"empty key at line {number}")
        while stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        if key in parent:
            raise ValueError(f"duplicate key {key!r} at line {number}")
        if not raw_value.strip():
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _scalar(raw_value)
    return root

