"""Write processed JSON and the paper summary table."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .metrics import serializable_analysis


def write_analysis_outputs(
    records: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    bootstrap_replicates: int = 10_000,
) -> dict[str, Any]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    analysis = serializable_analysis(records, bootstrap_replicates=bootstrap_replicates)
    (directory / "summary.json").write_text(
        json.dumps(analysis, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with (directory / "condition_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        fieldnames = [
            "condition",
            "run_count",
            "success_rate",
            "mean_model_list_cost_usd",
            "mean_model_billed_cost_usd",
            "mean_turn_count",
            "mean_tool_call_count",
            "mean_realized_ratio",
            "mean_reasoning_output_tokens",
            "aggregate_cdr",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for condition, summary in analysis["conditions"].items():
            row = {
                key: value
                for key, value in {"condition": condition, **summary, "aggregate_cdr": None}.items()
                if key in fieldnames
            }
            if condition in analysis["cost_transfer"]:
                row["aggregate_cdr"] = analysis["cost_transfer"][condition]["aggregate"]["cdr"]
            writer.writerow(row)
    return analysis
