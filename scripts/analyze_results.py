#!/usr/bin/env python3
"""Analyze an append-only experiment JSONL file without changing it."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.figures import write_required_figures
from src.analysis.reporting import write_analysis_outputs
from src.experiments.config import load_mapping_yaml
from src.experiments.runner import canonical_plan_attempts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--processed", type=Path, default=Path("results/processed"))
    parser.add_argument("--figures", type=Path, default=Path("results/figures"))
    parser.add_argument("--bootstrap", type=int, default=10_000)
    args = parser.parse_args()
    raw_records = [
        json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line
    ]
    records = canonical_plan_attempts(raw_records)
    pricing = load_mapping_yaml("configs/pricing.yaml")["standard"]
    analysis = write_analysis_outputs(
        records,
        args.processed,
        bootstrap_replicates=args.bootstrap,
        uncached_input_price_per_million=float(pricing["uncached_input_per_unit"]),
        output_price_per_million=float(pricing["output_per_unit"]),
        reasoning_price_per_million=(
            None
            if pricing.get("reasoning_output_per_unit") is None
            else float(pricing["reasoning_output_per_unit"])
        ),
    )
    terminal_records = tuple(
        record for record in records if record.get("failure_type") != "provider_error"
    )
    valid_records = tuple(
        record for record in terminal_records if record.get("analysis_valid", True)
    )

    def spend(values):
        return sum(
            float(record["model_cost_usd"]["billed"])
            + float(record["compressor_cost_usd"]["billed"])
            for record in values
        )

    audit = {
        "raw_rows": len(raw_records),
        "canonical_attempt_rows": len(records),
        "completed_plan_ordinals": len(
            {record["plan_ordinal"] for record in terminal_records}
        ),
        "analysis_valid_rows": len(valid_records),
        "excluded_terminal_rows": len(terminal_records) - len(valid_records),
        "ignored_replay_rows": len(raw_records) - len(records),
        "provider_error_attempts": sum(
            record.get("failure_type") == "provider_error" for record in records
        ),
        "terminal_failure_types": dict(
            sorted(
                Counter(
                    str(record.get("failure_type"))
                    for record in terminal_records
                    if record.get("failure_type") is not None
                ).items()
            )
        ),
        "actual_raw_spend_usd": spend(raw_records),
        "canonical_spend_usd": spend(records),
        "research_spend_usd": spend(valid_records),
    }
    args.processed.mkdir(parents=True, exist_ok=True)
    (args.processed / "run_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    write_required_figures(analysis, args.figures)


if __name__ == "__main__":
    main()
