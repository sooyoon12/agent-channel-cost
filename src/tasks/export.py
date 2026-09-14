"""Write and verify the frozen semantic-pair dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .generator import SemanticTask, generate_task_set, task_set_checksum


DATASET_VERSION = "ksc2026-v1"


def _json_line(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _payload(lines: Iterable[str]) -> bytes:
    return ("\n".join(lines) + "\n").encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _instance(task: SemanticTask, language: str) -> dict[str, Any]:
    value = task.to_dict()
    value["task_id"] = f"{task.pair_id}-{language}"
    value["language"] = language
    value["surface"] = task.ko_surface if language == "ko" else task.en_surface
    value.pop("ko_surface")
    value.pop("en_surface")
    return value


def build_dataset_payloads(tasks: Iterable[SemanticTask]) -> dict[str, bytes]:
    task_list = tuple(tasks)
    semantic_lines = [_json_line(task.to_dict()) for task in task_list]
    ko_lines = [_json_line(_instance(task, "ko")) for task in task_list]
    en_lines = [_json_line(_instance(task, "en")) for task in task_list]
    return {
        "semantic_tasks.jsonl": _payload(semantic_lines),
        "generated_tasks_ko.jsonl": _payload(ko_lines),
        "generated_tasks_en.jsonl": _payload(en_lines),
    }


def export_frozen_dataset(output_dir: str | Path) -> dict[str, Any]:
    """Create once, or verify that an existing frozen export is unchanged."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    tasks = generate_task_set(20)
    payloads = build_dataset_payloads(tasks)
    manifest = {
        "dataset_version": DATASET_VERSION,
        "generator_version": 1,
        "semantic_pair_count": len(tasks),
        "task_instance_count": len(tasks) * 2,
        "semantic_set_checksum": task_set_checksum(tasks),
        "files": {name: _sha256(payload) for name, payload in sorted(payloads.items())},
    }
    manifest_payload = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    payloads["manifest.json"] = manifest_payload
    for name, payload in payloads.items():
        path = directory / name
        if path.exists():
            if path.read_bytes() != payload:
                raise RuntimeError(f"frozen dataset differs from generator output: {path}")
            continue
        with path.open("xb") as stream:
            stream.write(payload)
    return manifest


def verify_frozen_dataset(output_dir: str | Path) -> dict[str, Any]:
    directory = Path(output_dir)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        actual = _sha256((directory / name).read_bytes())
        if actual != expected:
            raise RuntimeError(f"checksum mismatch: {name}")
    return manifest

