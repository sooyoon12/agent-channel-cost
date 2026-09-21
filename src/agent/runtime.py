"""Channel-aware agent loop shared by offline rehearsal and real providers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
import json
from typing import Any, Mapping
import uuid

from src.accounting.channels import (
    Channel,
    classify_input_item,
    classify_output_item,
    count_input_channels,
    count_input_item_content,
)
from src.accounting.invariants import check_input_residual
from src.accounting.pricing import PricingSchedule, calculate_model_cost
from src.accounting.shadow import (
    ReplayShadowRequest,
    ShadowContent,
    validate_retokenized_shadow_paths,
)
from src.compression.base import CompressionResult, Compressor
from src.compression.identity import IdentityCompressor
from src.environment.evaluator import EvaluationResult, evaluate_trajectory
from src.environment.tools import DeterministicToolEnvironment, TOOL_SPECS
from src.experiments.provider import Provider, ProviderRequest
from src.tasks.generator import SemanticTask


class ExperimentCondition(str, Enum):
    BASELINE = "B0"
    INSTRUCTION_HISTORY = "I-H"
    TOOL_OBSERVATION = "T-H"
    CONCISE_OUTPUT = "O"


@dataclass(frozen=True)
class RuntimeSettings:
    model: str
    temperature: float | None = None
    reasoning_effort: str = "low"
    max_output_tokens: int = 2048
    max_turns: int = 6
    max_identical_calls: int = 2
    max_request_input_tokens: int = 240_000
    target_ratio: float = 0.4
    residual_absolute_tolerance: int = 8
    residual_fractional_tolerance: float = 0.01
    residual_upper_bound_is_fatal: bool = True
    shadow_cost_tolerance: Decimal = Decimal("1e-9")
    provider_protocol_reserve_tokens: int = 1024
    allow_non_offline_provider: bool = False


@dataclass(frozen=True)
class RuntimeResult:
    record: Mapping[str, Any]


def _replace_payload(item: Mapping[str, Any], compressed_value: Any) -> dict[str, Any]:
    result = deepcopy(dict(item))
    if "content" in result:
        result["content"] = compressed_value
    elif "output" in result:
        result["output"] = compressed_value
    else:
        raise ValueError("item has no compressible content/output payload")
    return result


def _payload(item: Mapping[str, Any]) -> Any:
    if "content" in item:
        return item["content"]
    if "output" in item:
        return item["output"]
    raise ValueError("item has no compressible content/output payload")


class AgentRuntime:
    def __init__(
        self,
        *,
        provider: Provider,
        tokenizer: Any,
        pricing: PricingSchedule,
        compressor: Compressor,
        settings: RuntimeSettings,
        tokenizer_name: str,
        tokenizer_version: str,
    ) -> None:
        self.provider = provider
        self.tokenizer = tokenizer
        self.pricing = pricing
        self.compressor = compressor
        self.settings = settings
        self.tokenizer_name = tokenizer_name
        self.tokenizer_version = tokenizer_version

    def _should_compress(self, item: Mapping[str, Any], condition: ExperimentCondition) -> bool:
        classified = classify_input_item(item)
        if condition is ExperimentCondition.INSTRUCTION_HISTORY:
            return classified.channel is Channel.INSTRUCTION_HISTORY and "content" in item
        if condition is ExperimentCondition.TOOL_OBSERVATION:
            return classified.channel is Channel.TOOL_OBSERVATION and "output" in item
        return False

    def _materialize_request(
        self,
        history: list[Mapping[str, Any]],
        condition: ExperimentCondition,
    ) -> tuple[
        tuple[Mapping[str, Any], ...],
        tuple[ShadowContent, ...],
        list[tuple[Channel, CompressionResult]],
    ]:
        sent: list[Mapping[str, Any]] = []
        shadow: list[ShadowContent] = []
        compressions: list[tuple[Channel, CompressionResult]] = []
        identity = IdentityCompressor()
        for original in history:
            if self._should_compress(original, condition):
                result = self.compressor.compress(
                    _payload(original), target_ratio=self.settings.target_ratio, tokenizer=self.tokenizer
                )
                compressed_item = _replace_payload(original, result.compressed)
                compressions.append((classify_input_item(original).channel, result))
            else:
                result = identity.compress(original, target_ratio=1, tokenizer=self.tokenizer)
                compressed_item = result.compressed
            sent.append(compressed_item)
            shadow.append(ShadowContent(compressed=compressed_item, original=deepcopy(original)))
        return tuple(sent), tuple(shadow), compressions

    def run(
        self,
        task: SemanticTask,
        *,
        language: str,
        condition: ExperimentCondition,
        repetition: int,
        run_id: str | None = None,
    ) -> RuntimeResult:
        if language not in {"ko", "en"}:
            raise ValueError("language must be ko or en")
        if not self.settings.model:
            raise ValueError("a fixed model id is required")
        if self.provider.name != "scripted-offline" and not self.settings.allow_non_offline_provider:
            raise RuntimeError("non-offline provider is disabled until live preflight is explicitly approved")
        prompt = task.ko_surface if language == "ko" else task.en_surface
        history: list[Mapping[str, Any]] = [
            {
                "role": "system",
                "content": "Use only the supplied deterministic tools. Follow their schemas exactly.",
            },
            {"role": "user", "content": prompt},
            *(
                {
                    "type": "tool_schema",
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": deepcopy(spec.parameters),
                }
                for spec in TOOL_SPECS.values()
            ),
        ]
        environment = DeterministicToolEnvironment(task.hidden_state)
        events: list[dict[str, Any]] = []
        shadow_requests: list[ReplayShadowRequest] = []
        compression_results: list[CompressionResult] = []
        total_i = total_t = total_residual = total_provider_input = 0
        total_cached = total_cache_write = total_output = total_reasoning = 0
        cached_available = cache_write_available = reasoning_available = True
        natural_output = tool_call_output = 0
        billed_model = list_model = Decimal(0)
        turn_count = 0
        provider_error = False
        accounting_invalid_reason: str | None = None
        accounting_warning_count = 0
        repeat_blocked = False
        ended_with_final_output = False
        attempted_call_counts: dict[tuple[str, str], int] = {}

        while turn_count < self.settings.max_turns:
            turn_count += 1
            sent, shadow_contents, turn_compressions = self._materialize_request(history, condition)
            for compressed_channel, compression in turn_compressions:
                compression_results.append(compression)
                events.append(
                    {
                        "turn": turn_count,
                        "event_type": "compression",
                        "channel": compressed_channel.value,
                        "algorithm": compression.algorithm,
                        "algorithm_version": compression.algorithm_version,
                        "original_handle": compression.original_handle,
                        "compressed_handle": compression.compressed_handle,
                        "original_tokens": compression.original_tokens,
                        "compressed_tokens": compression.compressed_tokens,
                        "target_ratio": compression.target_ratio,
                        "realized_ratio": compression.realized_ratio,
                        "latency_ms": compression.latency_ms,
                    }
                )
            channel_counts = count_input_channels(sent, self.tokenizer)
            if (
                channel_counts.total_input_tokens + self.settings.provider_protocol_reserve_tokens
                >= self.settings.max_request_input_tokens
            ):
                raise ValueError("request input would reach or exceed the configured preflight token limit")
            request = ProviderRequest(
                items=sent,
                model=self.settings.model,
                temperature=self.settings.temperature,
                reasoning_effort=self.settings.reasoning_effort,
                max_output_tokens=self.settings.max_output_tokens,
                output_style="concise" if condition is ExperimentCondition.CONCISE_OUTPUT else "normal",
            )
            try:
                response = self.provider.complete(request)
            except Exception as exc:  # Provider adapters must translate details into the event log.
                provider_error = True
                events.append(
                    {
                        "turn": turn_count,
                        "event_type": "provider_error",
                        "message": str(exc),
                        "retryable": bool(getattr(exc, "retryable", False)),
                    }
                )
                break
            residual = check_input_residual(
                channel_counts.i_tokens,
                channel_counts.t_tokens,
                response.usage.input_tokens,
                absolute_tolerance=self.settings.residual_absolute_tolerance,
                fractional_tolerance=self.settings.residual_fractional_tolerance,
            )
            positive_residual_warning = (
                not residual.valid
                and residual.residual_tokens >= 0
                and not self.settings.residual_upper_bound_is_fatal
            )
            if positive_residual_warning:
                accounting_warning_count += 1
                events.append(
                    {
                        "turn": turn_count,
                        "event_type": "accounting_warning",
                        "message": residual.reason,
                        "local_i_tokens": channel_counts.i_tokens,
                        "local_t_tokens": channel_counts.t_tokens,
                        "provider_input_tokens": response.usage.input_tokens,
                        "protocol_residual_tokens": residual.residual_tokens,
                        "allowed_tokens": residual.allowed_tokens,
                    }
                )
            if not residual.valid and not positive_residual_warning:
                item_diagnostics = [
                    {
                        "index": index,
                        "type": item.get("type", "message"),
                        "role": item.get("role"),
                        "channel": classify_input_item(item).channel.value,
                        "content_tokens": count_input_item_content(item, self.tokenizer),
                        "id_tokens": self.tokenizer.count_text(str(item.get("id", ""))),
                        "call_id_tokens": self.tokenizer.count_text(
                            str(item.get("call_id", ""))
                        ),
                        "status_tokens": self.tokenizer.count_text(
                            str(item.get("status", ""))
                        ),
                        "encrypted_tokens": self.tokenizer.count_text(
                            str(item.get("encrypted_content", ""))
                        ),
                    }
                    for index, item in enumerate(sent)
                ]
                accounting_invalid_reason = (
                    f"{residual.reason}: turn={turn_count}, "
                    f"local_i={channel_counts.i_tokens}, local_t={channel_counts.t_tokens}, "
                    f"provider_input={response.usage.input_tokens}, "
                    f"residual={residual.residual_tokens}, allowed={residual.allowed_tokens}"
                )
                request_cost = calculate_model_cost(response.usage, self.pricing)
                billed_model += request_cost.billed.total
                list_model += request_cost.list_price.total
                total_i += channel_counts.i_tokens
                total_t += channel_counts.t_tokens
                total_residual += residual.residual_tokens
                total_provider_input += response.usage.input_tokens
                if response.usage.cached_input_tokens is None:
                    cached_available = False
                else:
                    total_cached += response.usage.cached_input_tokens
                if response.usage.cache_write_input_tokens is None:
                    cache_write_available = False
                else:
                    total_cache_write += response.usage.cache_write_input_tokens
                total_output += response.usage.output_tokens
                if response.usage.reasoning_output_tokens is None:
                    reasoning_available = False
                else:
                    total_reasoning += response.usage.reasoning_output_tokens
                events.append(
                    {
                        "turn": turn_count,
                        "event_type": "accounting_error",
                        "message": accounting_invalid_reason,
                        "items": item_diagnostics,
                        "provider_output_tokens": response.usage.output_tokens,
                        "provider_reasoning_tokens": response.usage.reasoning_output_tokens,
                        "latency_ms": response.latency_ms,
                        "price_version": self.pricing.version,
                    }
                )
                break
            if response.usage.input_tokens >= self.settings.max_request_input_tokens:
                raise ValueError("provider input reached or exceeded the 240K analysis limit")
            request_cost = calculate_model_cost(response.usage, self.pricing)
            billed_model += request_cost.billed.total
            list_model += request_cost.list_price.total
            reasoning_for_shadow = response.usage.reasoning_output_tokens or 0
            shadow_requests.append(
                ReplayShadowRequest(
                    shadow_contents,
                    residual.residual_tokens,
                    response.usage.output_tokens,
                    reasoning_for_shadow,
                )
            )
            total_i += channel_counts.i_tokens
            total_t += channel_counts.t_tokens
            total_residual += residual.residual_tokens
            total_provider_input += response.usage.input_tokens
            if response.usage.cached_input_tokens is None:
                cached_available = False
            else:
                total_cached += response.usage.cached_input_tokens
            if response.usage.cache_write_input_tokens is None:
                cache_write_available = False
            else:
                total_cache_write += response.usage.cache_write_input_tokens
            total_output += response.usage.output_tokens
            if response.usage.reasoning_output_tokens is None:
                reasoning_available = False
            total_reasoning += reasoning_for_shadow

            current_tool_calls: list[Mapping[str, Any]] = []
            for output_item in response.output_items:
                classify_output_item(output_item)
                output_tokens = self.tokenizer.count_json(output_item)
                if output_item.get("type") == "function_call":
                    tool_call_output += output_tokens
                    current_tool_calls.append(output_item)
                elif output_item.get("type") != "reasoning":
                    natural_output += output_tokens
                events.append(
                    {
                        "turn": turn_count,
                        "event_type": "model_output",
                        "channel": "O",
                        "item_type": output_item.get("type", "message"),
                        "local_tokens": output_tokens,
                        "provider_input_tokens": response.usage.input_tokens,
                        "provider_cached_input_tokens": response.usage.cached_input_tokens,
                        "provider_cache_write_input_tokens": response.usage.cache_write_input_tokens,
                        "provider_output_tokens": response.usage.output_tokens,
                        "provider_reasoning_tokens": response.usage.reasoning_output_tokens,
                        "latency_ms": response.latency_ms,
                        "price_version": self.pricing.version,
                    }
                )
                history.append(deepcopy(output_item))
            if not current_tool_calls:
                ended_with_final_output = True
                break
            for call in current_tool_calls:
                raw_arguments = call["arguments"]
                if isinstance(raw_arguments, str):
                    try:
                        parsed_arguments = json.loads(raw_arguments)
                    except json.JSONDecodeError:
                        parsed_arguments = None
                else:
                    parsed_arguments = raw_arguments
                normalized_arguments = json.dumps(
                    parsed_arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                call_key = (str(call["name"]), normalized_arguments)
                attempted_call_counts[call_key] = attempted_call_counts.get(call_key, 0) + 1
                if attempted_call_counts[call_key] > self.settings.max_identical_calls:
                    repeat_blocked = True
                    events.append(
                        {
                            "turn": turn_count,
                            "event_type": "tool_call_blocked",
                            "channel": "O",
                            "tool_name": call["name"],
                            "reason": "identical call repetition limit",
                        }
                    )
                    break
                execution = environment.execute(call["name"], parsed_arguments)
                tool_result = {
                    "type": "function_call_output",
                    "call_id": call.get("call_id"),
                    "output": execution.result,
                }
                history.append(tool_result)
                events.append(
                    {
                        "turn": turn_count,
                        "event_type": "tool_result",
                        "channel": "T",
                        "tool_name": execution.name,
                        "argument_hash": execution.argument_hash,
                        "error_code": execution.error_code,
                        "local_tokens": self.tokenizer.count_json(tool_result),
                        "latency_ms": execution.latency_ms,
                    }
                )
            if repeat_blocked:
                break

        evaluation = evaluate_trajectory(
            environment.executions,
            environment.snapshot(),
            expected_tool_calls=task.expected_tool_calls,
            allowed_equivalent_paths=task.allowed_equivalent_paths,
            expected_final_state=task.expected_final_state,
            turn_count=turn_count,
            max_turns=self.settings.max_turns,
            max_identical_calls=self.settings.max_identical_calls,
            provider_error=provider_error,
        )
        if repeat_blocked:
            evaluation = EvaluationResult(
                False,
                "repeated_call",
                "an identical tool call beyond the configured limit was blocked before execution",
            )
        elif accounting_invalid_reason is not None:
            evaluation = EvaluationResult(
                False,
                "accounting_invalid",
                accounting_invalid_reason,
            )
        elif (
            not provider_error
            and turn_count >= self.settings.max_turns
            and not ended_with_final_output
        ):
            evaluation = EvaluationResult(
                False,
                "max_turn/budget",
                "the model requested another tool call on the final allowed turn",
            )
        shadow = validate_retokenized_shadow_paths(
            shadow_requests,
            self.pricing,
            self.tokenizer,
            tolerance=self.settings.shadow_cost_tolerance,
        )
        compressed_token_total = sum(result.compressed_tokens for result in compression_results)
        original_token_total = sum(result.original_tokens for result in compression_results)
        realized_ratio = (
            compressed_token_total / original_token_total if original_token_total else 1.0
        )
        compressor_billed = sum((value.billed_cost_usd for value in compression_results), Decimal(0))
        compressor_list = sum((value.list_price_cost_usd for value in compression_results), Decimal(0))
        record = {
            "run_id": run_id or str(uuid.uuid4()),
            "pair_id": task.pair_id,
            "task_checksum": task.checksum(),
            "task_id": f"{task.pair_id}-{language}",
            "task_family": task.task_family,
            "sensitivity_group": task.sensitivity_group,
            "language": language,
            "condition": condition.value,
            "repetition": repetition,
            "provider": self.provider.name,
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "reasoning_effort": self.settings.reasoning_effort,
            "tokenizer_name": self.tokenizer_name,
            "tokenizer_version": self.tokenizer_version,
            "target_ratio": self.settings.target_ratio,
            "compression_algorithm": self.compressor.name,
            "compression_algorithm_version": self.compressor.version,
            "realized_ratio": realized_ratio,
            "success": evaluation.success and shadow.valid,
            "failure_type": evaluation.failure_type,
            "turn_count": turn_count,
            "tool_call_count": len(environment.executions),
            "model_cost_usd": {"billed": float(billed_model), "list_price": float(list_model)},
            "compressor_cost_usd": {
                "billed": float(compressor_billed),
                "list_price": float(compressor_list),
            },
            "tokens": {
                "local_i_input": total_i,
                "local_t_input": total_t,
                "protocol_residual_input": total_residual,
                "provider_input": total_provider_input,
                "provider_cached_input": total_cached if cached_available else None,
                "provider_cache_write_input": (
                    total_cache_write if cache_write_available else None
                ),
                "assistant_natural_output": natural_output,
                "assistant_tool_call_output": tool_call_output,
                "provider_reasoning_output": total_reasoning if reasoning_available else None,
                "provider_total_output": total_output,
                "compressor_input": sum(value.compressor_input_tokens for value in compression_results),
                "compressor_output": sum(value.compressor_output_tokens for value in compression_results),
            },
            "shadow": {
                "path_a_list_cost": float(shadow.path_a_list_cost),
                "path_b_list_cost": float(shadow.path_b_list_cost),
                "difference": float(shadow.difference),
                "valid": shadow.valid and accounting_invalid_reason is None,
            },
            "accounting_warning_count": accounting_warning_count,
            "events": events,
        }
        if evaluation.success and not shadow.valid:
            record["failure_type"] = "shadow_mismatch"
        return RuntimeResult(record)
