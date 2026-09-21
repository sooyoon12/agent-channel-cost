#!/usr/bin/env python3
"""Preflight or execute the v2 development/held-out T-H plan."""

from __future__ import annotations

import argparse
from decimal import Decimal
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.accounting.pricing import PriceTier, PricingSchedule
from src.accounting.tokenizer import resolve_tiktoken
from src.agent.runtime import AgentRuntime, RuntimeSettings
from src.compression.contract_aware import OracleContractCompressor
from src.compression.fixed_ratio import FixedRatioCompressor
from src.compression.head_tail import HeadTailCompressor
from src.compression.identity import IdentityCompressor
from src.compression.query_aware import QueryAwareCompressor
from src.experiments.config import load_mapping_yaml
from src.experiments.dotenv import environment_or_dotenv
from src.experiments.logger import JsonlRunLogger
from src.experiments.openai_provider import OpenAIResponsesProvider, UrllibJsonTransport
from src.experiments.preflight import require_live_ready, validate_live_configuration
from src.experiments.runner import BudgetGuard, execute_plan, execute_plan_concurrent
from src.experiments.v2 import (
    build_v2_development_baseline_plan,
    build_v2_heldout_baseline_plan,
    build_v2_heldout_compression_plan,
    build_v2_heldout_main_plan,
)
from src.tasks.export_v2 import verify_frozen_v2_dataset
from src.tasks.v2 import audit_v2_candidates, generate_v2_candidates, split_v2_candidates


LIVE_MODES = {
    "dev-b0",
    "heldout-b0",
    "heldout-uniform",
    "heldout-head-tail",
    "heldout-query-aware",
    "heldout-oracle",
    "heldout-main",
}


def _price_tier(value):
    reasoning = value.get("reasoning_output_per_unit")
    cache_write = value.get("cache_write_input_per_unit")
    return PriceTier(
        Decimal(str(value["uncached_input_per_unit"])),
        Decimal(str(value["cached_input_per_unit"])),
        Decimal(str(value["output_per_unit"])),
        Decimal(str(reasoning)) if reasoning is not None else None,
        Decimal(str(cache_write)) if cache_write is not None else None,
    )


def _tokenizer(model):
    config = model["tokenizer"]
    tokenizer = resolve_tiktoken(model["model_id"], proxy_encoding=config.get("proxy_encoding"))
    expected = config.get("exact_encoding") or config.get("proxy_encoding")
    if tokenizer.name != expected or tokenizer.version != config["version"]:
        raise RuntimeError(
            f"tokenizer mismatch: resolved {tokenizer.name}/{tokenizer.version}, "
            f"expected {expected}/{config['version']}"
        )
    return tokenizer


def _compressor(mode, task, method=None):
    if mode == "heldout-main":
        mode = {
            "b0": "heldout-b0",
            "uniform": "heldout-uniform",
            "head_tail": "heldout-head-tail",
            "query_aware_filter": "heldout-query-aware",
            "oracle_contract": "heldout-oracle",
        }[method]
    if mode.endswith("b0"):
        return IdentityCompressor()
    if mode == "heldout-uniform":
        return FixedRatioCompressor()
    if mode == "heldout-head-tail":
        return HeadTailCompressor()
    if mode == "heldout-query-aware":
        return QueryAwareCompressor(task.semantic_form)
    if mode == "heldout-oracle":
        return OracleContractCompressor()
    raise ValueError(f"unsupported mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", *sorted(LIVE_MODES)), nargs="?", default="preflight")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--projected-cost-per-run", type=Decimal)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    model = load_mapping_yaml("configs/model.yaml")
    pricing_config = load_mapping_yaml("configs/pricing.yaml")
    experiment = load_mapping_yaml("configs/experiment.yaml")
    compression = load_mapping_yaml("configs/compression.yaml")
    v2 = load_mapping_yaml("configs/v2.yaml")
    report = validate_live_configuration(
        model=model, pricing=pricing_config, experiment=experiment, compression=compression
    )
    tokenizer = None
    tokenizer_issue = None
    if report.ready:
        try:
            tokenizer = _tokenizer(model)
        except Exception as exc:
            tokenizer_issue = str(exc)

    tasks = generate_v2_candidates()
    development, held_out = split_v2_candidates(tasks)
    if args.mode == "preflight":
        audit = audit_v2_candidates(tasks)
        status = "ready" if report.ready and tokenizer_issue is None else "blocked"
        print(json.dumps(
            {
                "status": status,
                "issues": [*report.issues, *([tokenizer_issue] if tokenizer_issue else [])],
                "audit": audit,
                "development_pairs": len(development),
                "heldout_pairs": len(held_out),
                "development_runs": len(development) * 2,
                "heldout_runs_per_method": len(held_out) * 2,
                "i_h_in_scope": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ))
        return

    require_live_ready(
        model=model, pricing=pricing_config, experiment=experiment, compression=compression
    )
    if tokenizer_issue is not None or tokenizer is None:
        raise SystemExit(f"tokenizer preflight failed: {tokenizer_issue}")
    if args.output is None or args.projected_cost_per_run is None:
        raise SystemExit("paid modes require --output and --projected-cost-per-run")
    if args.projected_cost_per_run <= 0:
        raise SystemExit("projected cost per run must be positive")
    if args.output.exists() and not args.resume:
        raise SystemExit(f"output exists; pass --resume after verifying it: {args.output}")
    api_key = environment_or_dotenv(model["api_key_env"])
    if not api_key:
        raise SystemExit(f"missing API key environment variable: {model['api_key_env']}")

    if args.mode == "dev-b0":
        selected = development
        audit_v2_candidates(selected)
        plan = build_v2_development_baseline_plan(
            selected, order_seed=int(v2["development_order_seed"])
        )
        budget_limit = Decimal(str(v2["development_budget_usd"]))
    else:
        selected = held_out
        verify_frozen_v2_dataset("data/v2", selected)
        if args.mode == "heldout-main":
            plan = build_v2_heldout_main_plan(
                selected,
                target_ratio=float(v2["target_ratio"]),
                order_seed=int(v2["heldout_order_seed"]),
            )
        elif args.mode == "heldout-b0":
            plan = build_v2_heldout_baseline_plan(
                selected, order_seed=int(v2["heldout_order_seed"])
            )
        else:
            plan = build_v2_heldout_compression_plan(
                selected,
                target_ratio=float(v2["target_ratio"]),
                order_seed=int(v2["heldout_order_seed"]),
            )
        budget_limit = Decimal(str(v2["heldout_budget_usd"]))

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
            entry = by_ordinal[ordinal]
            expected_key = (entry.pair_id, entry.language, entry.condition.value, entry.repetition)
            observed_key = (
                record.get("pair_id"), record.get("language"),
                record.get("condition"), record.get("repetition"),
            )
            if observed_key != expected_key:
                raise SystemExit(f"existing log does not match plan ordinal {ordinal}")
            expected_algorithm = _compressor(
                args.mode,
                next(t for t in selected if t.pair_id == entry.pair_id),
                getattr(entry, "method", None),
            ).name
            if record.get("compression_algorithm") != expected_algorithm:
                raise SystemExit(f"existing log compressor mismatch at plan ordinal {ordinal}")
            starting_cost += Decimal(str(record["model_cost_usd"]["billed"])) + Decimal(
                str(record["compressor_cost_usd"]["billed"])
            )
            if record.get("failure_type") != "provider_error":
                completed.add(ordinal)
        plan = tuple(entry for entry in plan if entry.ordinal not in completed)
        print(f"resume verified: {len(completed)} complete, {len(plan)} remaining")

    standard = _price_tier(pricing_config["standard"])
    long_context = _price_tier(pricing_config["long_context"])
    pricing = PricingSchedule(
        standard=standard,
        long_context=long_context,
        long_context_threshold=pricing_config["long_context"]["threshold_input_tokens"],
        version=pricing_config["price_version"],
    )
    transport = UrllibJsonTransport(model["base_url"], api_key)

    def runtime_factory(task, entry):
        return AgentRuntime(
            provider=OpenAIResponsesProvider(transport),
            tokenizer=tokenizer,
            pricing=pricing,
            compressor=_compressor(args.mode, task, getattr(entry, "method", None)),
            settings=RuntimeSettings(
                model=model["model_id"],
                temperature=model["temperature"],
                reasoning_effort=model["reasoning_effort"],
                max_turns=experiment["max_agent_turns"],
                max_identical_calls=experiment["max_identical_tool_calls"],
                max_request_input_tokens=experiment["max_request_input_tokens"],
                target_ratio=float(v2["target_ratio"]),
                residual_absolute_tolerance=experiment["residual_tolerance"]["absolute_tokens"],
                residual_fractional_tolerance=experiment["residual_tolerance"]["fraction_of_provider_input"],
                residual_upper_bound_is_fatal=False,
                shadow_cost_tolerance=Decimal(str(experiment["shadow_cost_tolerance_usd"])),
                allow_non_offline_provider=True,
            ),
            tokenizer_name=tokenizer.name,
            tokenizer_version=tokenizer.version,
        )

    execution_kwargs = dict(
        runtime_factory=runtime_factory,
        logger=JsonlRunLogger(args.output),
        provider_error_retries=experiment["provider_error_retries"],
        budget_guard=BudgetGuard(limit_usd=budget_limit, warning_usd=budget_limit * Decimal("0.7")),
        projected_cost_per_run_usd=args.projected_cost_per_run,
        starting_cost_usd=starting_cost,
    )
    if args.workers == 1:
        records = execute_plan(plan, selected, **execution_kwargs)
    else:
        records = execute_plan_concurrent(
            plan, selected, workers=args.workers, **execution_kwargs
        )
    print(f"completed {len(records)} v2 runs: {args.output}")


if __name__ == "__main__":
    main()
