"""Shared runtime helpers for model adapters used by benchmark runners."""

from __future__ import annotations

import logging
from types import ModuleType
from typing import Any

import anthropic
import openai


ANTHROPIC_STREAMING_REQUIRED_ERROR = (
    "Streaming is required for operations that may take longer than 10 minutes"
)


def _exception_classes(*items: tuple[object, str]) -> tuple[type[BaseException], ...]:
    classes: list[type[BaseException]] = []
    for module, name in items:
        cls = getattr(module, name, None)
        if isinstance(cls, type) and issubclass(cls, BaseException):
            classes.append(cls)
    return tuple(classes)


NON_RETRYABLE_API_ERRORS = _exception_classes(
    (anthropic, "BadRequestError"),
    (anthropic, "AuthenticationError"),
    (anthropic, "PermissionDeniedError"),
    (anthropic, "NotFoundError"),
    (openai, "BadRequestError"),
    (openai, "AuthenticationError"),
    (openai, "PermissionDeniedError"),
    (openai, "NotFoundError"),
)


def is_non_retryable_model_error(exc: Exception) -> bool:
    """Classify local/configuration errors that backoff cannot fix."""
    message = str(exc)
    if isinstance(exc, NON_RETRYABLE_API_ERRORS):
        return True
    if isinstance(exc, TypeError):
        return True
    if isinstance(exc, ValueError) and ANTHROPIC_STREAMING_REQUIRED_ERROR in message:
        return True
    if isinstance(exc, RuntimeError) and "API_KEY non impostata" in message:
        return True
    return False


def anthropic_response_text(response: object) -> str:
    return "".join(
        getattr(block, "text", "")
        for block in getattr(response, "content", []) or []
        if getattr(block, "type", None) == "text"
    ).strip()


def log_anthropic_request_diagnostics(
    logger: logging.Logger,
    *,
    config_module: ModuleType,
    model: str,
    model_max_tokens: object,
    request: dict[str, object],
) -> None:
    logger.info(
        "Diagnostica Anthropic: sdk_version=%s sdk_path=%s config_path=%s "
        "MODEL_MAX_TOKENS=%r MODEL_MAX_TOKENS_type=%s model=%s request_max_tokens=%r",
        getattr(anthropic, "__version__", "unknown"),
        getattr(anthropic, "__file__", "unknown"),
        getattr(config_module, "__file__", "unknown"),
        model_max_tokens,
        type(model_max_tokens).__name__,
        model,
        request.get("max_tokens"),
    )


def stream_anthropic_message(
    client: Any,
    *,
    logger: logging.Logger,
    config_module: ModuleType,
    model: str,
    model_max_tokens: object,
    request: dict[str, object],
) -> Any:
    log_anthropic_request_diagnostics(
        logger,
        config_module=config_module,
        model=model,
        model_max_tokens=model_max_tokens,
        request=request,
    )
    with client.messages.stream(**request) as stream:
        return stream.get_final_message()
