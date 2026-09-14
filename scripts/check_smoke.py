#!/usr/bin/env python3
"""Evaluate smoke gates and print a machine-readable report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.config import load_mapping_yaml
from src.experiments.smoke import evaluate_smoke_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    experiment = load_mapping_yaml("configs/experiment.yaml")
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    report = evaluate_smoke_run(
        records,
        target_ratio=experiment["target_ratio"],
        realized_ratio_absolute_tolerance=experiment["realized_ratio_absolute_tolerance"],
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True, indent=2))
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

