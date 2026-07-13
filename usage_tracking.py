"""Utilities for model-call latency, token usage and cost estimates."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any


PRICE_ENV_NAMES = (
    "MODEL_PRICING_USD_PER_1M",
    "MODEL_PRICING_USD_PER_MILLION",
)

MODEL_CALL_SCORE_FIELDS = (
    "model_call_provider",
    "model_call_latency_ms",
    "model_call_input_tokens",
    "model_call_output_tokens",
    "model_call_total_tokens",
    "model_call_cached_input_tokens",
    "model_call_reasoning_tokens",
    "model_call_estimated_cost_usd",
    "model_call_cost_source",
    "model_call_usage",
)


@dataclass(frozen=True)
class ModelCallResult:
    text: str
    metrics: dict[str, Any]


def latency_ms(started_at: float) -> int:
    return int(round((time.perf_counter() - started_at) * 1000))


def object_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {
            str(key): object_to_dict(item) if _is_mapping_like(item) else item
            for key, item in value.items()
        }
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            return value.to_dict()
        except Exception:
            pass

    out: dict[str, Any] = {}
    for name in (
        "input_tokens",
        "output_tokens",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "prompt_tokens_details",
        "completion_tokens_details",
        "output_tokens_details",
        "cached_tokens",
        "reasoning_tokens",
    ):
        if hasattr(value, name):
            item = getattr(value, name)
            out[name] = object_to_dict(item) if _is_mapping_like(item) else item
    return out


def _is_mapping_like(value: Any) -> bool:
    return isinstance(value, dict) or hasattr(value, "model_dump") or hasattr(value, "to_dict")


def extract_usage(response: Any) -> dict[str, Any]:
    return object_to_dict(getattr(response, "usage", None))


def normalized_token_usage(usage: dict[str, Any]) -> dict[str, int | None]:
    details_prompt = object_to_dict(usage.get("prompt_tokens_details"))
    details_completion = object_to_dict(usage.get("completion_tokens_details"))
    details_output = object_to_dict(usage.get("output_tokens_details"))

    input_tokens = _int_or_none(usage.get("input_tokens"))
    if input_tokens is None:
        input_tokens = _int_or_none(usage.get("prompt_tokens"))

    output_tokens = _int_or_none(usage.get("output_tokens"))
    if output_tokens is None:
        output_tokens = _int_or_none(usage.get("completion_tokens"))

    total_tokens = _int_or_none(usage.get("total_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    cached_input_tokens = _int_or_none(usage.get("cache_read_input_tokens"))
    if cached_input_tokens is None:
        cached_input_tokens = _int_or_none(details_prompt.get("cached_tokens"))

    reasoning_tokens = _int_or_none(details_completion.get("reasoning_tokens"))
    if reasoning_tokens is None:
        reasoning_tokens = _int_or_none(details_output.get("reasoning_tokens"))

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_input_tokens": cached_input_tokens,
        "reasoning_tokens": reasoning_tokens,
    }


def build_model_call_metrics(
    *,
    provider: str,
    model: str,
    latency_ms_value: int,
    usage: dict[str, Any],
) -> dict[str, Any]:
    tokens = normalized_token_usage(usage)
    cost, source = estimate_cost_usd(provider=provider, model=model, token_usage=tokens)
    return {
        "provider": provider,
        "model": model,
        "latency_ms": latency_ms_value,
        "usage": usage,
        **tokens,
        "estimated_cost_usd": cost,
        "cost_source": source,
    }


def score_update_from_model_call(metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not metrics:
        return {}
    return {
        "model_call_provider": metrics.get("provider"),
        "model_call_latency_ms": metrics.get("latency_ms"),
        "model_call_input_tokens": metrics.get("input_tokens"),
        "model_call_output_tokens": metrics.get("output_tokens"),
        "model_call_total_tokens": metrics.get("total_tokens"),
        "model_call_cached_input_tokens": metrics.get("cached_input_tokens"),
        "model_call_reasoning_tokens": metrics.get("reasoning_tokens"),
        "model_call_estimated_cost_usd": metrics.get("estimated_cost_usd"),
        "model_call_cost_source": metrics.get("cost_source"),
        "model_call_usage": metrics.get("usage") or {},
    }


def with_model_call_metrics(score: Any, metrics: dict[str, Any] | None) -> Any:
    update = score_update_from_model_call(metrics)
    if not update:
        return score
    if hasattr(score, "model_copy"):
        return score.model_copy(update=update)
    for key, value in update.items():
        setattr(score, key, value)
    return score


def score_model_dump(score: Any, *, include_model_call_fields: bool = True) -> dict[str, Any]:
    if hasattr(score, "model_dump"):
        data = score.model_dump()
    else:
        data = dict(score)
    if not include_model_call_fields:
        for field in MODEL_CALL_SCORE_FIELDS:
            data.pop(field, None)
    return data


def aggregate_model_call_metrics(scores: list[Any]) -> dict[str, Any]:
    latencies = _numeric_values(scores, "model_call_latency_ms")
    costs = _numeric_values(scores, "model_call_estimated_cost_usd")
    return {
        "model_call_count": sum(
            1 for score in scores if getattr(score, "model_call_latency_ms", None) is not None
        ),
        "model_call_total_latency_ms": int(sum(latencies)) if latencies else 0,
        "model_call_mean_latency_ms": (sum(latencies) / len(latencies)) if latencies else None,
        "model_call_input_tokens": int(sum(_numeric_values(scores, "model_call_input_tokens"))),
        "model_call_output_tokens": int(sum(_numeric_values(scores, "model_call_output_tokens"))),
        "model_call_total_tokens": int(sum(_numeric_values(scores, "model_call_total_tokens"))),
        "model_call_cached_input_tokens": int(
            sum(_numeric_values(scores, "model_call_cached_input_tokens"))
        ),
        "model_call_reasoning_tokens": int(
            sum(_numeric_values(scores, "model_call_reasoning_tokens"))
        ),
        "model_call_estimated_cost_usd": sum(costs) if costs else None,
    }


def estimate_cost_usd(
    *,
    provider: str,
    model: str,
    token_usage: dict[str, int | None],
) -> tuple[float | None, str | None]:
    price, source = _lookup_price(provider, model)
    if not price:
        return None, None

    input_tokens = token_usage.get("input_tokens") or 0
    output_tokens = token_usage.get("output_tokens") or 0
    cached_input_tokens = min(token_usage.get("cached_input_tokens") or 0, input_tokens)
    regular_input_tokens = input_tokens

    input_price = _price_value(price, "input")
    output_price = _price_value(price, "output")
    cached_input_price = _price_value(price, "cached_input")

    cost = 0.0
    if cached_input_price is not None:
        regular_input_tokens = max(0, input_tokens - cached_input_tokens)
        cost += cached_input_tokens / 1_000_000 * cached_input_price
    if input_price is not None:
        cost += regular_input_tokens / 1_000_000 * input_price
    if output_price is not None:
        cost += output_tokens / 1_000_000 * output_price

    return round(cost, 8), source


def _lookup_price(provider: str, model: str) -> tuple[dict[str, Any] | None, str | None]:
    table, source = _load_price_table()
    if not table:
        return None, None
    keys = (
        model,
        model.replace("/", "-"),
        f"{provider}/{model}",
        f"{provider}:{model}",
        provider,
    )
    for key in keys:
        value = table.get(key)
        if isinstance(value, dict):
            return value, source
    return None, None


def _load_price_table() -> tuple[dict[str, Any], str | None]:
    for name in PRICE_ENV_NAMES:
        raw = os.environ.get(name)
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}, name
        if isinstance(parsed, dict):
            return parsed, name
    return {}, None


def _price_value(price: dict[str, Any], name: str) -> float | None:
    for key in (name, f"{name}_per_1m", f"{name}_per_million"):
        value = price.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _numeric_values(items: list[Any], attr: str) -> list[float]:
    values = []
    for item in items:
        value = getattr(item, attr, None)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values
