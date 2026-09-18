#!/usr/bin/env python3
"""Create the KSC evaluation comparison and analyze the frozen ratio sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.followup import (
    analyze_contract_guard,
    analyze_ratio_sweep,
    evaluation_method_comparison,
)
from src.analysis.figures import write_followup_success_figure
from src.experiments.config import load_mapping_yaml
from src.experiments.runner import canonical_plan_attempts


def read_canonical(path: Path):
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
    ]
    return canonical_plan_attempts(records)


def write_csv(path: Path, rows) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline", type=Path, default=Path("results/raw/2026-09-17.jsonl")
    )
    parser.add_argument("--sweep", type=Path)
    parser.add_argument("--guard", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/followup"))
    args = parser.parse_args()

    config = load_mapping_yaml("configs/followup.yaml")
    baseline = read_canonical(args.baseline)
    comparison = evaluation_method_comparison(
        baseline, success_margin=float(config["success_noninferiority_margin"])
    )
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "evaluation_method_comparison.csv", comparison)
    (args.output / "evaluation_method_comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.sweep is None:
        print("wrote evaluation-method comparison; ratio sweep not supplied")
        return

    sweep = read_canonical(args.sweep)
    result = analyze_ratio_sweep(
        baseline,
        sweep,
        success_margin=float(config["success_noninferiority_margin"]),
        confidence_level=float(config["confidence_level"]),
        bootstrap_replicates=int(config["bootstrap_replicates"]),
        seed=int(config["execution_order_seed"]),
    )
    (args.output / "ratio_sweep_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    flat_cells = []
    for cell in result["cells"]:
        flat_cells.append(
            {
                "condition": cell["condition"],
                "target_ratio": cell["target_ratio"],
                "planned_run_count": cell["planned_run_count"],
                "analysis_valid_count": cell["analysis_valid_count"],
                "success_rate_planned": cell["success_rate_planned"],
                "one_sided_95_harm_upper": cell["one_sided_95_harm_upper"],
                "list_price_saving": cell["list_price_saving"]["estimate"],
                "list_price_saving_ci_lower": cell["list_price_saving"]["lower"],
                "list_price_saving_ci_upper": cell["list_price_saving"]["upper"],
                "cdr": cell["cdr"]["estimate"],
                "safe_and_cost_saving": cell["safe_and_cost_saving"],
            }
        )
    write_csv(args.output / "ratio_sweep_summary.csv", flat_cells)
    if args.guard is not None:
        architecture = load_mapping_yaml("configs/architecture.yaml")
        guard_result = analyze_contract_guard(
            baseline,
            sweep,
            read_canonical(args.guard),
            target_ratio=float(architecture["target_ratio"]),
            success_margin=float(config["success_noninferiority_margin"]),
            bootstrap_replicates=int(config["bootstrap_replicates"]),
            seed=int(architecture["execution_order_seed"]),
        )
        (args.output / "contract_guard_summary.json").write_text(
            json.dumps(guard_result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        write_followup_success_figure(
            comparison,
            result,
            guard_result,
            args.output / "figure3_success_frontier.svg",
        )
    print(f"wrote follow-up analysis: {args.output}")


if __name__ == "__main__":
    main()
