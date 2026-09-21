#!/usr/bin/env python3
"""Freeze the untouched v2 held-out set after a passing development report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tasks.export_v2 import export_frozen_v2_dataset
from src.tasks.v2 import generate_v2_candidates, split_v2_candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("development_report", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/v2"))
    args = parser.parse_args()
    report = json.loads(args.development_report.read_text(encoding="utf-8"))
    _, held_out = split_v2_candidates(generate_v2_candidates())
    manifest = export_frozen_v2_dataset(
        args.output_dir, held_out, development_report=report
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
