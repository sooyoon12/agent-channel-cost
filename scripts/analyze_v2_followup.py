#!/usr/bin/env python3
"""Analyze the v2 exploratory sweep against its concurrent repeated B0."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.v2_followup import summarize_v2_followup
from src.analysis.figures import write_v2_frontier_figure
from src.experiments.config import load_mapping_yaml
from src.tasks.v2 import generate_v2_candidates, split_v2_candidates


def _records(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("followup", type=Path)
    parser.add_argument("--main", type=Path, default=Path("results/v2/heldout-main-2026-09-21.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/v2/followup-summary.json"))
    parser.add_argument("--csv", type=Path, default=Path("results/v2/followup-summary.csv"))
    parser.add_argument("--figure", type=Path, default=Path("results/figures/figure_v2_followup_frontier.svg"))
    args = parser.parse_args()
    config = load_mapping_yaml("configs/v2_followup.yaml")
    method_ratios = tuple(
        (method, float(ratio))
        for method, ratios in config["methods"].items()
        for ratio in str(ratios).split(",")
    )
    _, tasks = split_v2_candidates(generate_v2_candidates())
    report = summarize_v2_followup(
        _records(args.main), _records(args.followup), tasks,
        method_ratios=method_ratios,
        bootstrap_replicates=int(config["bootstrap_replicates"]),
        bootstrap_seed=int(config["order_seed"]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = list(report["cells"].values())
    with args.csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_v2_frontier_figure(report, args.figure)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
