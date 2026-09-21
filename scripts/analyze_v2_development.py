#!/usr/bin/env python3
"""Apply the preregistered v2 B0 ceiling/difficulty gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.v2 import development_baseline_report
from src.experiments.config import load_mapping_yaml
from src.tasks.v2 import generate_v2_candidates, split_v2_candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    development, _ = split_v2_candidates(generate_v2_candidates())
    config = load_mapping_yaml("configs/v2.yaml")
    report = development_baseline_report(
        records,
        development,
        baseline_floor=float(config["baseline_floor"]),
        ceiling_threshold=float(config["ceiling_threshold"]),
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
