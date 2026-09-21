"""Freeze and verify the v2 held-out dataset only after the development gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .export import build_dataset_payloads
from .generator import SemanticTask, task_set_checksum
from .v2 import audit_v2_candidates


DATASET_VERSION = "ksc2026-v2-th"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def export_frozen_v2_dataset(
    output_dir: str | Path,
    tasks: Sequence[SemanticTask],
    *,
    development_report: Mapping[str, Any],
) -> dict[str, Any]:
    if development_report.get("status") != "pass":
        raise RuntimeError("development gate did not pass; held-out data must remain unfrozen")
    audit = audit_v2_candidates(tasks)
    payloads = build_dataset_payloads(tasks)
    manifest = {
        "dataset_version": DATASET_VERSION,
        "generator_version": 2,
        "status": "frozen_after_development_gate",
        "scope": "T-H active tasks only; I-H excluded",
        "semantic_pair_count": len(tasks),
        "task_instance_count": len(tasks) * 2,
        "semantic_set_checksum": task_set_checksum(tasks),
        "audit": audit,
        "development_raw_records_sha256": development_report["raw_records_sha256"],
        "development_gate": development_report["preregistered_gate"],
        "files": {name: _sha256(payload) for name, payload in sorted(payloads.items())},
    }
    manifest_payload = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    payloads["manifest.json"] = manifest_payload
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        path = directory / name
        if path.exists() and path.read_bytes() != payload:
            raise RuntimeError(f"frozen v2 dataset differs from requested payload: {path}")
        if not path.exists():
            with path.open("xb") as stream:
                stream.write(payload)
    return manifest


def verify_frozen_v2_dataset(output_dir: str | Path, tasks: Sequence[SemanticTask]) -> dict[str, Any]:
    directory = Path(output_dir)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset_version") != DATASET_VERSION:
        raise RuntimeError("unexpected v2 dataset version")
    if manifest.get("semantic_set_checksum") != task_set_checksum(tasks):
        raise RuntimeError("v2 generator output differs from frozen semantic checksum")
    for name, expected in manifest["files"].items():
        if _sha256((directory / name).read_bytes()) != expected:
            raise RuntimeError(f"checksum mismatch: {name}")
    return manifest
