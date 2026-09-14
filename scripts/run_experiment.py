#!/usr/bin/env python3
"""Preflight or execute the frozen smoke/main plan.

The default mode is read-only preflight. `smoke` and `main` can spend provider
credits and therefore require an explicit output path and projected per-run cost.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.accounting.pricing import PriceTier, PricingSchedule
from src.accounting.tokenizer import resolve_tiktoken
from src.agent.runtime import AgentRuntime, RuntimeSettings
from src.compression.fixed_ratio import FixedRatioCompressor
from src.experiments.logger import JsonlRunLogger
from src.experiments.config import load_mapping_yaml
from src.experiments.openai_provider import OpenAIResponsesProvider, UrllibJsonTransport
from src.experiments.preflight import require_live_ready, validate_live_configuration
from src.experiments.runner import BudgetGuard, build_main_plan, build_smoke_plan, execute_plan
from src.tasks.export import verify_frozen_dataset
from src.tasks.generator import generate_task_set


def price_tier(value):
    reasoning = value.get("reasoning_output_per_unit")
    return PriceTier(
        Decimal(str(value["uncached_input_per_unit"])),
        Decimal(str(value["cached_input_per_unit"])),
        Decimal(str(value["output_per_unit"])),
        Decimal(str(reasoning)) if reasoning is not None else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "smoke", "main"), default="preflight", nargs="?")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--projected-cost-per-run", type=Decimal)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    model = load_mapping_yaml(Path("configs/model.yaml"))
    pricing_config = load_mapping_yaml(Path("configs/pricing.yaml"))
    experiment = load_mapping_yaml(Path("configs/experiment.yaml"))
    compression = load_mapping_yaml(Path("configs/compression.yaml"))
    report = validate_live_configuration(
        model=model, pricing=pricing_config, experiment=experiment, compression=compression
    )
    if args.mode == "preflight":
        if report.ready:
            print("live preflight ready")
        else:
            print("live preflight blocked")
            for issue in report.issues:
                print("-", issue)
        return
    require_live_ready(
        model=model, pricing=pricing_config, experiment=experiment, compression=compression
    )
    if args.output is None or args.projected_cost_per_run is None:
        raise SystemExit("paid modes require --output and --projected-cost-per-run")
    if args.output.exists() and not args.resume:
        raise SystemExit(f"output exists; pass --resume after verifying it: {args.output}")
    api_key = os.environ.get(model["api_key_env"])
    if not api_key:
        raise SystemExit(f"missing API key environment variable: {model['api_key_env']}")
    verify_frozen_dataset("data")
    tokenizer_config = model["tokenizer"]
    tokenizer = resolve_tiktoken(
        model["model_id"], proxy_encoding=tokenizer_config.get("proxy_encoding")
    )
    standard = price_tier(pricing_config["standard"])
    long_context = (
        price_tier(pricing_config["long_context"])
        if pricing_config.get("long_context", {}).get("uncached_input_per_unit") is not None
        else None
    )
    pricing = PricingSchedule(
        standard=standard,
        long_context=long_context,
        long_context_threshold=pricing_config.get("long_context", {}).get("threshold_input_tokens"),
        version=pricing_config["price_version"],
    )
    transport = UrllibJsonTransport(model["base_url"], api_key)
    tasks = generate_task_set()
    plan = build_smoke_plan(tasks) if args.mode == "smoke" else build_main_plan(
        tasks, order_seed=experiment["execution_order_seed"]
    )
    starting_cost = Decimal(0)
    if args.output.exists():
        existing = [
            json.loads(line)
            for line in args.output.read_text(encoding="utf-8").splitlines()
            if line
        ]
        by_ordinal = {entry.ordinal: entry for entry in plan}
        completed: set[int] = set()
        for record in existing:
            ordinal = record.get("plan_ordinal")
            if ordinal not in by_ordinal:
                raise SystemExit(f"existing log has an ordinal outside this plan: {ordinal}")
            expected = by_ordinal[ordinal]
            observed_key = (
                record.get("pair_id"),
                record.get("language"),
                record.get("condition"),
                record.get("repetition"),
            )
            expected_key = (
                expected.pair_id,
                expected.language,
                expected.condition.value,
                expected.repetition,
            )
            if observed_key != expected_key:
                raise SystemExit(f"existing log does not match plan ordinal {ordinal}")
            starting_cost += Decimal(str(record["model_cost_usd"]["billed"])) + Decimal(
                str(record["compressor_cost_usd"]["billed"])
            )
            if record.get("analysis_valid", True) and record.get("failure_type") != "provider_error":
                completed.add(ordinal)
        plan = tuple(entry for entry in plan if entry.ordinal not in completed)
        print(f"resume verified: {len(completed)} complete, {len(plan)} remaining")

    def runtime_factory(task):
        return AgentRuntime(
            provider=OpenAIResponsesProvider(transport),
            tokenizer=tokenizer,
            pricing=pricing,
            compressor=FixedRatioCompressor(),
            settings=RuntimeSettings(
                model=model["model_id"],
                temperature=model["temperature"],
                reasoning_effort=model["reasoning_effort"],
                max_turns=experiment["max_agent_turns"],
                max_identical_calls=experiment["max_identical_tool_calls"],
                max_request_input_tokens=experiment["max_request_input_tokens"],
                target_ratio=compression["target_ratio"],
                residual_absolute_tolerance=experiment["residual_tolerance"]["absolute_tokens"],
                residual_fractional_tolerance=experiment["residual_tolerance"]["fraction_of_provider_input"],
                shadow_cost_tolerance=Decimal(str(experiment["shadow_cost_tolerance_usd"])),
                allow_non_offline_provider=True,
            ),
            tokenizer_name=tokenizer.name,
            tokenizer_version=tokenizer.version,
        )

    records = execute_plan(
        plan,
        tasks,
        runtime_factory=runtime_factory,
        logger=JsonlRunLogger(args.output),
        provider_error_retries=experiment["provider_error_retries"],
        budget_guard=BudgetGuard(),
        projected_cost_per_run_usd=args.projected_cost_per_run,
        starting_cost_usd=starting_cost,
    )
    print(f"completed {len(records)} research runs: {args.output}")


if __name__ == "__main__":
    main()
