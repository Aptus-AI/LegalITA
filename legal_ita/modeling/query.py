"""
Layer condiviso di interrogazione dei modelli sotto esame.

Contiene gli adapter provider (Anthropic, OpenAI, Gemini via endpoint
OpenAI-compatibile, Novita) e il ciclo di retry con backoff esponenziale
usati da run_benchmark.py e run_bullshit_v2.py. I runner mantengono nel
proprio namespace gli alias storici (_query_anthropic, ...) e il routing
per prefisso, cosi' i test possono sostituire i singoli adapter sul modulo
runner senza conoscere questo layer.
"""

import logging
import os
import time
from collections.abc import Callable

import anthropic
from legal_ita import config as benchmark_config
import openai

from legal_ita.config import MODEL_MAX_TOKENS
from legal_ita.modeling.request_config import (
    anthropic_message_kwargs,
    gemini_completion_kwargs,
    novita_completion_kwargs,
    openai_completion_kwargs,
)
from legal_ita.modeling.runtime import (
    anthropic_response_text,
    is_non_retryable_model_error,
    stream_anthropic_message,
)
from legal_ita.modeling.usage import (
    ModelCallResult,
    build_model_call_metrics,
    extract_usage,
    latency_ms,
)

log = logging.getLogger(__name__)

NOVITA_PROVIDERS = ("deepseek/", "meta-llama/", "qwen/", "mistralai/", "zai-org/")
NOVITA_BASE_URL = "https://api.novita.ai/openai"
NOVITA_GLM_52_MAX_TOKENS = 65536
GEMINI_PROVIDER_PREFIX = "gemini-"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def run_query_with_retries(
    adapter: Callable[[str, str], object],
    model: str,
    query: str,
    *,
    max_retries: int,
    log: logging.Logger,
    validate: Callable[[object], None] | None = None,
) -> object | None:
    """
    Esegue adapter(model, query) con retry e backoff esponenziale.

    `validate` puo' sollevare per forzare il retry su risposte non valide
    (es. testo vuoto). Gli errori non recuperabili interrompono subito i
    tentativi; in ogni caso di fallimento il risultato e' None.
    """
    for attempt in range(1, max_retries + 1):
        try:
            result = adapter(model, query)
            if validate is not None:
                validate(result)
            return result
        except Exception as exc:
            if is_non_retryable_model_error(exc):
                log.error(
                    "Errore non recuperabile query %s (tentativo %d/%d): %s",
                    model,
                    attempt,
                    max_retries,
                    exc,
                )
                return None
            if attempt >= max_retries:
                log.warning(
                    "Errore ritentabile query %s (tentativo %d/%d): %s; tentativi esauriti",
                    model,
                    attempt,
                    max_retries,
                    exc,
                )
                break
            delay = 2.0 * (2 ** (attempt - 1))
            log.warning(
                "Errore ritentabile query %s (tentativo %d/%d): %s; attendo %.1fs",
                model,
                attempt,
                max_retries,
                exc,
                delay,
            )
            time.sleep(delay)

    log.error("Query fallita dopo %d tentativi: %s", max_retries, model)
    return None


def require_result_text(result: ModelCallResult) -> None:
    if not result.text:
        raise ValueError("Risposta vuota dal modello.")


def require_answer_text(answer: str) -> None:
    if not answer:
        raise ValueError("Risposta vuota dal modello.")


def query_anthropic(model: str, query: str) -> str:
    return query_anthropic_with_metrics(model, query).text


def query_anthropic_with_metrics(model: str, query: str) -> ModelCallResult:
    client = anthropic.Anthropic()
    started_at = time.perf_counter()
    request = default_anthropic_message_kwargs(model, query)
    response = stream_anthropic_message(
        client,
        logger=log,
        config_module=benchmark_config,
        model=model,
        model_max_tokens=MODEL_MAX_TOKENS,
        request=request,
    )
    text = anthropic_response_text(response)
    if not text:
        content_types = [
            getattr(block, "type", type(block).__name__)
            for block in getattr(response, "content", []) or []
        ]
        raise ValueError(
            "Risposta Anthropic priva di blocchi testuali finali "
            f"(content_types={content_types})."
        )
    return ModelCallResult(
        text=text,
        metrics=build_model_call_metrics(
            provider="anthropic",
            model=model,
            latency_ms_value=latency_ms(started_at),
            usage=extract_usage(response),
        ),
    )


def default_anthropic_message_kwargs(model: str, query: str) -> dict[str, object]:
    return anthropic_message_kwargs(model, query, MODEL_MAX_TOKENS)


def query_openai(model: str, query: str) -> str:
    return query_openai_with_metrics(model, query).text


def query_openai_with_metrics(model: str, query: str) -> ModelCallResult:
    client = openai.OpenAI()
    started_at = time.perf_counter()
    response = client.chat.completions.create(
        **default_openai_completion_kwargs(model, query),
    )
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("Risposta OpenAI priva di contenuto testuale.")
    return ModelCallResult(
        text=content.strip(),
        metrics=build_model_call_metrics(
            provider="openai",
            model=model,
            latency_ms_value=latency_ms(started_at),
            usage=extract_usage(response),
        ),
    )


def default_openai_completion_kwargs(model: str, query: str) -> dict[str, object]:
    return openai_completion_kwargs(model, query, MODEL_MAX_TOKENS)


def query_gemini(model: str, query: str) -> str:
    return query_gemini_with_metrics(model, query).text


def default_gemini_completion_kwargs(model: str, query: str) -> dict[str, object]:
    return gemini_completion_kwargs(model, query, MODEL_MAX_TOKENS)


def query_gemini_with_metrics(model: str, query: str) -> ModelCallResult:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY non impostata in .env")

    client = openai.OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)
    started_at = time.perf_counter()
    response = client.chat.completions.create(**default_gemini_completion_kwargs(model, query))
    choice = response.choices[0]
    if getattr(choice, "finish_reason", None) == "length":
        log.warning("Risposta %s troncata dal provider: finish_reason=length", model)
    content = choice.message.content
    if content is None:
        raise ValueError("Risposta Gemini priva di contenuto testuale.")
    return ModelCallResult(
        text=content.strip(),
        metrics=build_model_call_metrics(
            provider="google",
            model=model,
            latency_ms_value=latency_ms(started_at),
            usage=extract_usage(response),
        ),
    )


def default_novita_completion_kwargs(model: str, query: str) -> dict[str, object]:
    return novita_completion_kwargs(
        model,
        query,
        max_tokens=MODEL_MAX_TOKENS,
        glm_max_tokens=NOVITA_GLM_52_MAX_TOKENS,
    )


def query_novita(model: str, query: str) -> str:
    return query_novita_with_metrics(model, query).text


def query_novita_with_metrics(model: str, query: str) -> ModelCallResult:
    api_key = os.environ.get("NOVITA_API_KEY")
    if not api_key:
        raise RuntimeError("NOVITA_API_KEY non impostata in .env")

    client = openai.OpenAI(api_key=api_key, base_url=NOVITA_BASE_URL)
    started_at = time.perf_counter()
    response = client.chat.completions.create(**default_novita_completion_kwargs(model, query))
    choice = response.choices[0]
    if getattr(choice, "finish_reason", None) == "length":
        log.warning("Risposta %s troncata dal provider: finish_reason=length", model)
    content = choice.message.content
    if content is None:
        raise ValueError("Risposta Novita priva di contenuto testuale.")
    return ModelCallResult(
        text=content.strip(),
        metrics=build_model_call_metrics(
            provider="novita",
            model=model,
            latency_ms_value=latency_ms(started_at),
            usage=extract_usage(response),
        ),
    )


def model_request_kwargs_for_summary(model: str) -> dict[str, object]:
    if model.startswith(NOVITA_PROVIDERS):
        return default_novita_completion_kwargs(model, "")
    if model.startswith(GEMINI_PROVIDER_PREFIX):
        return default_gemini_completion_kwargs(model, "")
    if model.startswith("claude"):
        return default_anthropic_message_kwargs(model, "")
    if any(model.startswith(prefix) for prefix in ("gpt", "o1", "o3", "o4")):
        return default_openai_completion_kwargs(model, "")
    return {"model": model}
