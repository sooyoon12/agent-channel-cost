#!/usr/bin/env python3
"""Analyze an append-only experiment JSONL file without changing it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.figures import write_required_figures
from src.analysis.reporting import write_analysis_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--processed", type=Path, default=Path("results/processed"))
    parser.add_argument("--figures", type=Path, default=Path("results/figures"))
    parser.add_argument("--bootstrap", type=int, default=10_000)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    analysis = write_analysis_outputs(records, args.processed, bootstrap_replicates=args.bootstrap)
    write_required_figures(analysis, args.figures)


if __name__ == "__main__":
    main()
