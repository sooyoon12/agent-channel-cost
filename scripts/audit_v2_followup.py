#!/usr/bin/env python3
"""Offline-only retention and plan audit for the v2 exploratory follow-up."""

from __future__ import annotations

from collections import defaultdict
import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.accounting.tokenizer import resolve_tiktoken
from src.analysis.v2_main import _gold_action_fields_survive
from src.compression.fixed_ratio import FixedRatioCompressor
from src.compression.head_tail import HeadTailCompressor
from src.compression.surface_ranked import SurfaceRankedCompressor
from src.environment.tools import DeterministicToolEnvironment
from src.experiments.config import load_mapping_yaml
from src.experiments.runner import plan_checksum
from src.experiments.v2 import build_v2_followup_plan
from src.tasks.export_v2 import verify_frozen_v2_dataset
from src.tasks.v2 import generate_v2_candidates, split_v2_candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    model = load_mapping_yaml("configs/model.yaml")
    config = load_mapping_yaml("configs/v2_followup.yaml")
    tokenizer = resolve_tiktoken(
        model["model_id"], proxy_encoding=model["tokenizer"].get("proxy_encoding")
    )
    _, tasks = split_v2_candidates(generate_v2_candidates())
    verify_frozen_v2_dataset("data/v2", tasks)
    method_ratios = tuple(
        (method, float(ratio))
        for method, ratios in config["methods"].items()
        for ratio in str(ratios).split(",")
    )
    plan = build_v2_followup_plan(
        tasks,
        method_ratios=method_ratios,
        order_seed=int(config["order_seed"]),
        include_repeat_baseline=bool(config["repeat_baseline"]),
    )
    summaries = defaultdict(lambda: {"ratios": [], "retained": 0, "total": 0})
    for task in tasks:
        environment = DeterministicToolEnvironment(task.hidden_state)
        first = task.expected_tool_calls[0]
        observation = environment.execute(first["name"], first["arguments"]).result
        for language in ("ko", "en"):
            prompt = task.ko_surface if language == "ko" else task.en_surface
            for method, ratio in method_ratios:
                compressor = {
                    "uniform": FixedRatioCompressor(),
                    "head_tail": HeadTailCompressor(),
                    "surface_ranked": SurfaceRankedCompressor(prompt),
                }[method]
                result = compressor.compress(
                    observation, target_ratio=ratio, tokenizer=tokenizer
                )
                key = f"{method}@{ratio:.1f}"
                summaries[key]["total"] += 1
                summaries[key]["retained"] += int(
                    _gold_action_fields_survive(task, result.compressed)
                )
                summaries[key]["ratios"].append(result.realized_ratio)
    report = {
        "status": "offline_only_no_api_calls",
        "plan_runs": len(plan),
        "plan_checksum": plan_checksum(plan),
        "cells": {
            key: {
                "runs": values["total"],
                "gold_action_fields_retained": values["retained"],
                "gold_action_field_retention_rate": values["retained"] / values["total"],
                "mean_realized_ratio": statistics.fmean(values["ratios"]),
            }
            for key, values in sorted(summaries.items())
        },
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
