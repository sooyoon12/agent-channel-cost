#!/usr/bin/env python3
"""Analyze a complete frozen v2 held-out main log."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.accounting.tokenizer import resolve_tiktoken
from src.analysis.v2_main import (
    information_retention_report,
    residual_distribution_report,
    summarize_v2_main,
)
from src.experiments.config import load_mapping_yaml
from src.tasks.v2 import generate_v2_candidates, split_v2_candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--retention-csv", type=Path)
    parser.add_argument("--residual-csv", type=Path)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    _, held_out = split_v2_candidates(generate_v2_candidates())
    report = summarize_v2_main(records, held_out)
    model = load_mapping_yaml("configs/model.yaml")
    tokenizer = resolve_tiktoken(
        model["model_id"], proxy_encoding=model["tokenizer"].get("proxy_encoding")
    )
    report["information_retention"] = information_retention_report(
        records, held_out, tokenizer
    )
    report["protocol_residual_distribution"] = residual_distribution_report(records)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        baseline = report["methods"]["b0"]
        with args.csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "method",
                    "runs",
                    "successes",
                    "success_rate",
                    "success_delta_pp_vs_b0",
                    "mean_realized_ratio",
                    "mean_local_t_input_tokens",
                    "mean_list_price_cost_per_run_usd",
                    "list_price_cost_change_percent_vs_b0",
                    "list_price_cost_per_success_usd",
                    "list_price_cost_per_success_ci_low",
                    "list_price_cost_per_success_ci_high",
                    "cost_per_success_change_percent_vs_b0",
                    "cost_per_success_change_ci_low_percent",
                    "cost_per_success_change_ci_high_percent",
                    "strict_pair_mcnemar_holm_p",
                    "role",
                ),
            )
            writer.writeheader()
            for method, values in report["methods"].items():
                comparison = report["paired_comparisons_vs_b0"].get(method, {})
                cps_ci = values[
                    "list_price_cost_per_success_pair_clustered_95ci_usd"
                ]
                cps_change_ci = comparison.get(
                    "list_price_cost_per_success_change_pair_clustered_95ci_percent",
                    [None, None],
                )
                cost_change = 100 * (
                    values["mean_list_price_cost_per_run_usd"]
                    / baseline["mean_list_price_cost_per_run_usd"]
                    - 1
                )
                writer.writerow(
                    {
                        "method": method,
                        "runs": values["runs"],
                        "successes": values["successes"],
                        "success_rate": values["success_rate"],
                        "success_delta_pp_vs_b0": comparison.get(
                            "success_delta_percentage_points", 0
                        ),
                        "mean_realized_ratio": values["mean_realized_ratio"],
                        "mean_local_t_input_tokens": values["mean_local_t_input_tokens"],
                        "mean_list_price_cost_per_run_usd": values[
                            "mean_list_price_cost_per_run_usd"
                        ],
                        "list_price_cost_change_percent_vs_b0": cost_change,
                        "list_price_cost_per_success_usd": values[
                            "list_price_cost_per_success_usd"
                        ],
                        "list_price_cost_per_success_ci_low": cps_ci[0],
                        "list_price_cost_per_success_ci_high": cps_ci[1],
                        "cost_per_success_change_percent_vs_b0": comparison.get(
                            "list_price_cost_per_success_change_vs_b0_percent", 0
                        ),
                        "cost_per_success_change_ci_low_percent": cps_change_ci[0],
                        "cost_per_success_change_ci_high_percent": cps_change_ci[1],
                        "strict_pair_mcnemar_holm_p": comparison.get(
                            "mcnemar_strict_pair_holm_adjusted_p"
                        ),
                        "role": {
                            "b0": "main baseline",
                            "uniform": "main destructive baseline",
                            "head_tail": "main destructive baseline",
                            "oracle_contract": "schema-oracle upper bound",
                            "query_aware_filter": "theoretical lower-bound reference",
                        }[method],
                    }
                )
    if args.retention_csv:
        args.retention_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.retention_csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=("method", "stratum", "runs", "successes", "success_rate"),
            )
            writer.writeheader()
            for method, values in report["information_retention"].items():
                for stratum, stratum_values in values["strata"].items():
                    writer.writerow({"method": method, "stratum": stratum, **stratum_values})
    if args.residual_csv:
        args.residual_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.residual_csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "method", "runs", "warning_runs", "warning_requests",
                    "run_residual_tokens_median", "run_residual_tokens_p95", "run_residual_tokens_max",
                    "run_residual_fraction_median", "run_residual_fraction_p95", "run_residual_fraction_max",
                    "warning_request_tokens_max", "warning_request_fraction_max",
                ),
            )
            writer.writeheader()
            for method, values in report["protocol_residual_distribution"].items():
                run_tokens = values["aggregate_run_residual_tokens"]
                run_fractions = values["aggregate_run_residual_fraction"]
                warning_tokens = values["warning_request_residual_tokens"]
                warning_fractions = values["warning_request_residual_fraction"]
                writer.writerow(
                    {
                        "method": method,
                        "runs": values["runs"],
                        "warning_runs": values["warning_runs"],
                        "warning_requests": values["warning_requests"],
                        "run_residual_tokens_median": run_tokens["median"],
                        "run_residual_tokens_p95": run_tokens["p95"],
                        "run_residual_tokens_max": run_tokens["max"],
                        "run_residual_fraction_median": run_fractions["median"],
                        "run_residual_fraction_p95": run_fractions["p95"],
                        "run_residual_fraction_max": run_fractions["max"],
                        "warning_request_tokens_max": warning_tokens["max"],
                        "warning_request_fraction_max": warning_fractions["max"],
                    }
                )
    print(rendered, end="")


if __name__ == "__main__":
    main()
