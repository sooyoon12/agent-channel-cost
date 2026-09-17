"""Minimal, dependency-free loading for one local secret file."""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Mapping


_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def read_dotenv(path: str | Path) -> dict[str, str]:
    """Read simple KEY=VALUE entries without mutating process environment."""

    source = Path(path)
    if not source.exists():
        return {}
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"invalid .env entry at line {line_number}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not _KEY.fullmatch(key):
            raise ValueError(f"invalid .env key at line {line_number}")
        if value.startswith(("'", '"')):
            quote = value[0]
            if len(value) < 2 or value[-1] != quote:
                raise ValueError(f"unterminated quoted .env value at line {line_number}")
            value = value[1:-1]
        values[key] = value
    return values


def environment_or_dotenv(
    name: str,
    *,
    dotenv_path: str | Path = ".env",
    environment: Mapping[str, str] | None = None,
) -> str | None:
    """Prefer the process environment, then fall back to a local .env file."""

    current = os.environ if environment is None else environment
    value = current.get(name)
    if value:
        return value
    return read_dotenv(dotenv_path).get(name) or None
