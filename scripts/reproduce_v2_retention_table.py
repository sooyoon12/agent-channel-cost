#!/usr/bin/env python3
"""Reproduce paper Table 3 from run logs without making API calls.

The script independently regenerates each decision-bearing tool observation,
applies the deterministic compressor, and matches its SHA-256 handles against
the handles recorded in the run log.  Only then does it cross-tabulate gold
field retention and the observed run outcome.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.accounting.tokenizer import resolve_tiktoken
from src.analysis.v2_followup import _planned_outcomes
from src.analysis.v2_main import _gold_action_fields_survive
from src.compression.base import content_handle
from src.compression.fixed_ratio import FixedRatioCompressor
from src.compression.head_tail import HeadTailCompressor
from src.compression.surface_ranked import SurfaceRankedCompressor
from src.environment.tools import DeterministicToolEnvironment
from src.experiments.config import load_mapping_yaml
from src.tasks.export_v2 import verify_frozen_v2_dataset
from src.tasks.v2 import generate_v2_candidates, split_v2_candidates


EXPECTED = {
    ("uniform", 0.6): (76, 0, 0, 74),
    ("uniform", 0.9): (134, 0, 0, 16),
    ("head_tail", 0.6): (70, 2, 0, 78),
    ("head_tail", 0.9): (129, 7, 0, 14),
    ("surface_ranked", 0.2): (112, 0, 0, 38),
    ("surface_ranked", 0.4): (130, 1, 0, 19),
    ("surface_ranked", 0.8): (148, 2, 0, 0),
}


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _compressor(method: str, prompt: str):
    if method == "uniform":
        return FixedRatioCompressor()
    if method == "head_tail":
        return HeadTailCompressor()
    if method == "surface_ranked":
        return SurfaceRankedCompressor(prompt)
    raise ValueError(f"unsupported follow-up method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "log", nargs="?", type=Path, default=Path("results/v2/followup-main.jsonl")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/v2/followup-retention-outcome-audit.json"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/v2/followup-retention-outcome-audit.csv"),
    )
    args = parser.parse_args()

    model = load_mapping_yaml("configs/model.yaml")
    tokenizer = resolve_tiktoken(
        model["model_id"], proxy_encoding=model["tokenizer"].get("proxy_encoding")
    )
    _, tasks = split_v2_candidates(generate_v2_candidates())
    verify_frozen_v2_dataset("data/v2", tasks)
    task_by_id = {task.pair_id: task for task in tasks}

    rows = [
        row
        for row in _planned_outcomes(_load_jsonl(args.log))
        if row.get("experiment_method") != "b0_repeat"
    ]
    observed_keys = [
        (
            str(row.get("pair_id")),
            str(row.get("language")),
            str(row.get("experiment_method")),
            float(row.get("target_ratio")),
        )
        for row in rows
    ]
    expected_keys = {
        (task.pair_id, language, method, ratio)
        for task in tasks
        for language in ("ko", "en")
        for method, ratio in EXPECTED
    }
    duplicates = [key for key, count in Counter(observed_keys).items() if count > 1]
    missing = expected_keys - set(observed_keys)
    unexpected = set(observed_keys) - expected_keys
    if duplicates or missing or unexpected or len(rows) != 1050:
        raise ValueError(
            "follow-up log is incomplete: "
            f"rows={len(rows)}, missing={len(missing)}, "
            f"unexpected={len(unexpected)}, duplicates={len(duplicates)}"
        )

    counts: dict[tuple[str, float], Counter[str]] = defaultdict(Counter)
    matched = 0
    for row in rows:
        task = task_by_id[str(row["pair_id"])]
        language = str(row["language"])
        method = str(row["experiment_method"])
        ratio = float(row["target_ratio"])
        environment = DeterministicToolEnvironment(task.hidden_state)
        first_call = task.expected_tool_calls[0]
        observation = environment.execute(
            first_call["name"], first_call["arguments"]
        ).result
        prompt = task.ko_surface if language == "ko" else task.en_surface
        result = _compressor(method, prompt).compress(
            observation, target_ratio=ratio, tokenizer=tokenizer
        )

        original_handle = content_handle(observation)
        events = [
            event
            for event in row.get("events", [])
            if event.get("event_type") == "compression"
            and event.get("original_handle") == original_handle
        ]
        if not events:
            raise ValueError(f"no decision-observation handle match for run {row['run_id']}")
        if any(
            event.get("compressed_handle") != result.compressed_handle
            or event.get("algorithm") != result.algorithm
            or event.get("algorithm_version") != result.algorithm_version
            for event in events
        ):
            raise ValueError(f"compressor hash mismatch for run {row['run_id']}")
        matched += 1

        retained = _gold_action_fields_survive(task, result.compressed)
        outcome = "success" if bool(row.get("success")) else "failure"
        counts[(method, ratio)][f"{'retained' if retained else 'removed'}_{outcome}"] += 1

    result_rows = []
    paper_match = True
    for method, ratio in EXPECTED:
        cell = counts[(method, ratio)]
        values = (
            cell["retained_success"],
            cell["retained_failure"],
            cell["removed_success"],
            cell["removed_failure"],
        )
        matches = values == EXPECTED[(method, ratio)]
        paper_match &= matches
        result_rows.append(
            {
                "method": method,
                "target_ratio": ratio,
                "retained_success": values[0],
                "retained_failure": values[1],
                "removed_success": values[2],
                "removed_failure": values[3],
                "runs": sum(values),
                "paper_table_match": matches,
            }
        )
    if not paper_match:
        raise ValueError("recomputed counts do not match paper Table 3")

    report = {
        "status": "offline_only_no_api_calls",
        "input": str(args.log),
        "terminal_compression_runs": len(rows),
        "sha256_matched_runs": matched,
        "paper_table_match": paper_match,
        "totals": {
            "retained_success": sum(row["retained_success"] for row in result_rows),
            "retained_failure": sum(row["retained_failure"] for row in result_rows),
            "removed_success": sum(row["removed_success"] for row in result_rows),
            "removed_failure": sum(row["removed_failure"] for row in result_rows),
        },
        "cells": result_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with args.csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
