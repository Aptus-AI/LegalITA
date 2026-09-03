"""Request-shape helpers for evaluated models.

These helpers are intentionally used only by answer-generating runners, not by
LLM-as-judge adapters.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


OPENAI_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")
OPENAI_MAX_SUPPORTED_REASONING_EFFORT = "high"
GEMINI_MAX_SUPPORTED_REASONING_EFFORT = "high"
PROVIDER_MAX_REASONING_EFFORT = "max"
ANTHROPIC_REASONING_PREFIXES = ("claude-sonnet-4", "claude-opus-4")
GEMINI_REASONING_PREFIXES = ("gemini-2.5", "gemini-3")
NOVITA_REASONING_PREFIXES = ("deepseek/", "qwen/", "zai-org/")


def anthropic_message_kwargs(model: str, query: str, max_tokens: int) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": query}],
    }
    if _starts_with_any(model, ANTHROPIC_REASONING_PREFIXES):
        kwargs["thinking"] = {"type": "adaptive", "display": "omitted"}
        kwargs["output_config"] = {"effort": PROVIDER_MAX_REASONING_EFFORT}
    return kwargs


def openai_completion_kwargs(model: str, query: str, max_tokens: int) -> dict[str, object]:
    normalized = model.strip().lower()
    token_param = (
        "max_completion_tokens"
        if normalized.startswith(OPENAI_REASONING_PREFIXES)
        else "max_tokens"
    )
    kwargs: dict[str, object] = {
        "model": model,
        token_param: max_tokens,
        "messages": [{"role": "user", "content": query}],
    }
    if normalized.startswith(OPENAI_REASONING_PREFIXES):
        kwargs["reasoning_effort"] = OPENAI_MAX_SUPPORTED_REASONING_EFFORT
    return kwargs


def gemini_completion_kwargs(model: str, query: str, max_tokens: int) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": query}],
    }
    if _starts_with_any(model, GEMINI_REASONING_PREFIXES):
        kwargs["reasoning_effort"] = GEMINI_MAX_SUPPORTED_REASONING_EFFORT
    return kwargs


def novita_completion_kwargs(
    model: str,
    query: str,
    *,
    max_tokens: int,
    glm_max_tokens: int,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": query}],
    }

    if _starts_with_any(model, NOVITA_REASONING_PREFIXES):
        kwargs["reasoning_effort"] = PROVIDER_MAX_REASONING_EFFORT
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}

    if model.startswith("zai-org/"):
        extra_body = dict(kwargs.get("extra_body") or {})
        extra_body.update(
            {
                "top_k": 50,
                "repetition_penalty": 1,
                "min_p": 0,
            }
        )
        kwargs.update(
            {
                "max_tokens": glm_max_tokens,
                "temperature": 1,
                "top_p": 1,
                "presence_penalty": 0,
                "frequency_penalty": 0,
                "response_format": {"type": "text"},
                "extra_body": extra_body,
            }
        )

    return kwargs


def request_config_for_summary(kwargs: dict[str, object]) -> dict[str, object]:
    """Return a JSON-safe request summary without prompt/message content."""
    summary = deepcopy(kwargs)
    summary.pop("messages", None)
    summary.pop("input", None)
    return summary


def _starts_with_any(value: str, prefixes: tuple[str, ...]) -> bool:
    return value.strip().lower().startswith(prefixes)
