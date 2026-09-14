"""Primary and secondary metrics with pair-level resampling."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from decimal import Decimal
import math
import random
from statistics import mean, variance
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class CostTransferObservation:
    pair_id: str
    task_id: str
    language: str
    condition: str
    baseline_list_cost: Decimal
    compressed_list_cost: Decimal
    shadow_list_cost: Decimal
    compressor_list_cost: Decimal
    gross_saving: Decimal
    behavioral_displacement: Decimal
    net_saving: Decimal


@dataclass(frozen=True)
class AggregateCostTransfer:
    condition: str
    observation_count: int
    pair_count: int
    gross_saving: Decimal
    behavioral_displacement: Decimal
    net_saving: Decimal
    cdr: Decimal | None


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float | None
    upper: float | None
    valid_replicates: int
    requested_replicates: int
    seed: int


@dataclass(frozen=True)
class McNemarResult:
    baseline_only_success: int
    condition_only_success: int
    concordant_success: int
    concordant_failure: int
    exact_two_sided_p: float


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value))


def _baseline_groups(records: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if (
            record["condition"] == "B0"
            and record.get("shadow", {}).get("valid", False)
            and record.get("analysis_valid", True)
        ):
            groups[record["task_id"]].append(record)
    return groups


def baseline_list_costs(
    records: Sequence[Mapping[str, Any]], *, require_repetitions: int | None = 3
) -> dict[str, Decimal]:
    groups = _baseline_groups(records)
    result: dict[str, Decimal] = {}
    for task_id, values in groups.items():
        if require_repetitions is not None and len(values) != require_repetitions:
            raise ValueError(
                f"{task_id} has {len(values)} valid B0 records; expected {require_repetitions}"
            )
        result[task_id] = sum(
            (_decimal(value["model_cost_usd"]["list_price"]) for value in values), Decimal(0)
        ) / Decimal(len(values))
    return result


def build_cost_transfer_observations(
    records: Iterable[Mapping[str, Any]],
    condition: str,
    *,
    require_baseline_repetitions: int | None = 3,
) -> tuple[CostTransferObservation, ...]:
    if condition not in {"I-H", "T-H"}:
        raise ValueError("CDR is defined only for I-H and T-H")
    values = tuple(records)
    baselines = baseline_list_costs(values, require_repetitions=require_baseline_repetitions)
    observations: list[CostTransferObservation] = []
    for record in values:
        if record["condition"] != condition:
            continue
        if not record.get("shadow", {}).get("valid", False) or not record.get("analysis_valid", True):
            continue
        if record["task_id"] not in baselines:
            raise ValueError(f"missing B0 baseline for {record['task_id']}")
        baseline = baselines[record["task_id"]]
        compressed = _decimal(record["model_cost_usd"]["list_price"])
        shadow = _decimal(record["shadow"]["path_a_list_cost"])
        compressor = _decimal(record["compressor_cost_usd"]["list_price"])
        gross = shadow - compressed
        displacement = shadow - baseline + compressor
        observations.append(
            CostTransferObservation(
                pair_id=record["pair_id"],
                task_id=record["task_id"],
                language=record["language"],
                condition=condition,
                baseline_list_cost=baseline,
                compressed_list_cost=compressed,
                shadow_list_cost=shadow,
                compressor_list_cost=compressor,
                gross_saving=gross,
                behavioral_displacement=displacement,
                net_saving=baseline - compressed - compressor,
            )
        )
    return tuple(observations)


def aggregate_cost_transfer(
    observations: Iterable[CostTransferObservation],
    *,
    minimum_gross_saving: Decimal = Decimal("0"),
) -> AggregateCostTransfer:
    values = tuple(observations)
    if not values:
        raise ValueError("no cost-transfer observations")
    conditions = {value.condition for value in values}
    if len(conditions) != 1:
        raise ValueError("aggregate requires exactly one condition")
    gross = sum((value.gross_saving for value in values), Decimal(0))
    displacement = sum((value.behavioral_displacement for value in values), Decimal(0))
    net = sum((value.net_saving for value in values), Decimal(0))
    cdr = None if gross <= minimum_gross_saving else displacement / gross
    return AggregateCostTransfer(
        condition=next(iter(conditions)),
        observation_count=len(values),
        pair_count=len({value.pair_id for value in values}),
        gross_saving=gross,
        behavioral_displacement=displacement,
        net_saving=net,
        cdr=cdr,
    )


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take percentile of an empty sequence")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def bootstrap_cdr(
    observations: Iterable[CostTransferObservation],
    *,
    replicates: int = 10_000,
    seed: int = 20260914,
    minimum_gross_saving: Decimal = Decimal("0"),
) -> BootstrapInterval:
    values = tuple(observations)
    estimate_value = aggregate_cost_transfer(values, minimum_gross_saving=minimum_gross_saving).cdr
    if estimate_value is None:
        return BootstrapInterval(float("nan"), None, None, 0, replicates, seed)
    by_pair: dict[str, list[CostTransferObservation]] = defaultdict(list)
    for value in values:
        by_pair[value.pair_id].append(value)
    pair_ids = sorted(by_pair)
    if not pair_ids:
        raise ValueError("no pair ids for bootstrap")
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(replicates):
        selected = [rng.choice(pair_ids) for _ in pair_ids]
        gross = sum(
            (value.gross_saving for pair_id in selected for value in by_pair[pair_id]), Decimal(0)
        )
        if gross <= minimum_gross_saving:
            continue
        displacement = sum(
            (
                value.behavioral_displacement
                for pair_id in selected
                for value in by_pair[pair_id]
            ),
            Decimal(0),
        )
        samples.append(float(displacement / gross))
    samples.sort()
    if not samples:
        return BootstrapInterval(float(estimate_value), None, None, 0, replicates, seed)
    return BootstrapInterval(
        estimate=float(estimate_value),
        lower=_percentile(samples, 0.025),
        upper=_percentile(samples, 0.975),
        valid_replicates=len(samples),
        requested_replicates=replicates,
        seed=seed,
    )


def summarize_conditions(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["condition"]].append(record)
    summaries: dict[str, dict[str, Any]] = {}
    for condition, values in sorted(groups.items()):
        list_costs = [float(value["model_cost_usd"]["list_price"]) for value in values]
        billed_costs = [float(value["model_cost_usd"]["billed"]) for value in values]
        summaries[condition] = {
            "run_count": len(values),
            "success_rate": sum(bool(value["success"]) for value in values) / len(values),
            "mean_model_list_cost_usd": mean(list_costs),
            "mean_model_billed_cost_usd": mean(billed_costs),
            "mean_turn_count": mean(float(value["turn_count"]) for value in values),
            "mean_tool_call_count": mean(float(value["tool_call_count"]) for value in values),
            "mean_realized_ratio": mean(float(value["realized_ratio"]) for value in values),
            "mean_local_i_input_tokens": _mean_nested(values, "tokens", "local_i_input"),
            "mean_local_t_input_tokens": _mean_nested(values, "tokens", "local_t_input"),
            "mean_provider_input_tokens": _mean_nested(values, "tokens", "provider_input"),
            "mean_reasoning_output_tokens": _mean_nested(
                values, "tokens", "provider_reasoning_output"
            ),
            "mean_provider_output_tokens": _mean_nested(values, "tokens", "provider_total_output"),
        }
    return summaries


def _mean_nested(
    values: Sequence[Mapping[str, Any]], container: str, field: str
) -> float | None:
    present = [
        float(value[container][field])
        for value in values
        if container in value and value[container].get(field) is not None
    ]
    return mean(present) if present else None


def _shadow_channel_tokens(record: Mapping[str, Any], channel: str) -> float:
    field = "local_i_input" if channel == "I" else "local_t_input"
    tokens = float(record["tokens"][field])
    for event in record.get("events", []):
        if event.get("event_type") == "compression" and event.get("channel") == channel:
            tokens += float(event["original_tokens"]) - float(event["compressed_tokens"])
    return tokens


def _provider_output_allocation(record: Mapping[str, Any]) -> dict[str, float] | None:
    tokens = record["tokens"]
    reasoning_value = tokens.get("provider_reasoning_output")
    if reasoning_value is None:
        return None
    reasoning = float(reasoning_value)
    visible = max(0.0, float(tokens["provider_total_output"]) - reasoning)
    natural_local = float(tokens["assistant_natural_output"])
    tool_local = float(tokens["assistant_tool_call_output"])
    denominator = natural_local + tool_local
    if denominator == 0:
        return {"natural_output": 0.0, "tool_call_output": 0.0, "other_output": visible, "reasoning": reasoning}
    return {
        "natural_output": visible * natural_local / denominator,
        "tool_call_output": visible * tool_local / denominator,
        "other_output": 0.0,
        "reasoning": reasoning,
    }


def displacement_decomposition(
    records: Iterable[Mapping[str, Any]],
    condition: str,
    *,
    uncached_input_price_per_million: float,
    output_price_per_million: float,
    reasoning_price_per_million: float | None = None,
) -> dict[str, Any]:
    """Decompose shadow-minus-B0 behavior using exact input and allocated output.

    Providers expose total visible output rather than natural/tool-call billing
    subcounts. Visible provider tokens are therefore allocated in proportion to
    their local-token subcounts; the method is surfaced in the returned metadata.
    """

    values = tuple(records)
    baseline_groups = _baseline_groups(values)

    def components(record: Mapping[str, Any], *, shadow: bool) -> dict[str, float] | None:
        allocation = _provider_output_allocation(record)
        if allocation is None:
            return None
        return {
            "instruction_history_input": (
                _shadow_channel_tokens(record, "I")
                if shadow
                else float(record["tokens"]["local_i_input"])
            ),
            "tool_observation_input": (
                _shadow_channel_tokens(record, "T")
                if shadow
                else float(record["tokens"]["local_t_input"])
            ),
            "protocol_residual_input": float(record["tokens"]["protocol_residual_input"]),
            **allocation,
        }

    baseline_components: dict[str, dict[str, float]] = {}
    for task_id, group in baseline_groups.items():
        component_rows = [components(record, shadow=False) for record in group]
        if any(row is None for row in component_rows):
            continue
        rows = [row for row in component_rows if row is not None]
        baseline_components[task_id] = {
            key: mean(row[key] for row in rows) for key in rows[0]
        }
    token_delta: dict[str, float] = defaultdict(float)
    model_difference = 0.0
    compressor_cost = 0.0
    observation_count = 0
    for record in values:
        if record["condition"] != condition or not record["shadow"]["valid"]:
            continue
        task_id = record["task_id"]
        row = components(record, shadow=True)
        if row is None or task_id not in baseline_components:
            continue
        for key, value in row.items():
            token_delta[key] += value - baseline_components[task_id][key]
        baseline_cost = mean(
            float(value["model_cost_usd"]["list_price"])
            for value in baseline_groups[task_id]
        )
        model_difference += float(record["shadow"]["path_a_list_cost"]) - baseline_cost
        compressor_cost += float(record["compressor_cost_usd"]["list_price"])
        observation_count += 1
    reasoning_rate = reasoning_price_per_million or output_price_per_million
    cost_components = {
        "instruction_history_input": token_delta["instruction_history_input"]
        * uncached_input_price_per_million
        / 1_000_000,
        "tool_observation_input": token_delta["tool_observation_input"]
        * uncached_input_price_per_million
        / 1_000_000,
        "protocol_residual_input": token_delta["protocol_residual_input"]
        * uncached_input_price_per_million
        / 1_000_000,
        "natural_output": token_delta["natural_output"] * output_price_per_million / 1_000_000,
        "tool_call_output": token_delta["tool_call_output"] * output_price_per_million / 1_000_000,
        "other_output": token_delta["other_output"] * output_price_per_million / 1_000_000,
        "reasoning": token_delta["reasoning"] * reasoning_rate / 1_000_000,
    }
    allocated = sum(cost_components.values())
    cost_components["other_model_billing"] = model_difference - allocated
    return {
        "condition": condition,
        "observation_count": observation_count,
        "token_delta": dict(token_delta),
        "model_cost_delta_usd": cost_components,
        "compressor_list_cost_usd": compressor_cost,
        "behavioral_displacement_usd": model_difference + compressor_cost,
        "visible_output_allocation": "proportional_to_local_natural_and_tool_call_tokens",
    }


def baseline_repeat_variance(records: Iterable[Mapping[str, Any]]) -> dict[str, float | None]:
    groups = _baseline_groups(tuple(records))
    return {
        task_id: (
            variance(float(value["model_cost_usd"]["list_price"]) for value in values)
            if len(values) >= 2
            else None
        )
        for task_id, values in groups.items()
    }


def mcnemar_exact(records: Iterable[Mapping[str, Any]], condition: str) -> McNemarResult:
    values = tuple(records)
    baseline_groups = _baseline_groups(values)
    baseline_majority = {
        task_id: sum(bool(value["success"]) for value in group) * 2 >= len(group)
        for task_id, group in baseline_groups.items()
    }
    condition_values = [value for value in values if value["condition"] == condition]
    seen: set[str] = set()
    cells = {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0}
    for value in condition_values:
        task_id = value["task_id"]
        if task_id in seen:
            raise ValueError(f"multiple {condition} outcomes for {task_id}")
        if task_id not in baseline_majority:
            raise ValueError(f"missing B0 outcome for {task_id}")
        seen.add(task_id)
        cells[(baseline_majority[task_id], bool(value["success"]))] += 1
    b = cells[(True, False)]
    c = cells[(False, True)]
    discordant = b + c
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(0, min(b, c) + 1)) / (2**discordant)
        p_value = min(1.0, 2 * tail)
    return McNemarResult(
        baseline_only_success=b,
        condition_only_success=c,
        concordant_success=cells[(True, True)],
        concordant_failure=cells[(False, False)],
        exact_two_sided_p=p_value,
    )


def serializable_analysis(
    records: Sequence[Mapping[str, Any]], *, bootstrap_replicates: int = 10_000
) -> dict[str, Any]:
    result: dict[str, Any] = {"conditions": summarize_conditions(records), "cost_transfer": {}}
    for condition in ("I-H", "T-H"):
        observations = build_cost_transfer_observations(records, condition)
        aggregate = aggregate_cost_transfer(observations)
        interval = bootstrap_cdr(observations, replicates=bootstrap_replicates)
        aggregate_value = asdict(aggregate)
        for key, value in aggregate_value.items():
            if isinstance(value, Decimal):
                aggregate_value[key] = float(value)
        result["cost_transfer"][condition] = {
            "aggregate": aggregate_value,
            "cdr_bootstrap_95": asdict(interval),
            "mcnemar": asdict(mcnemar_exact(records, condition)),
        }
    return result
