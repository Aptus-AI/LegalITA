from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Iterable

from .ecli import (
    OUTSIDE_INDEX_NAMESPACES,
    SUPPORTED_NAMESPACES,
    VENUE_CODES,
    derive_authority_codes,
    infer_namespace_from_ecli,
    normalize_cass_sector,
    normalize_code,
    normalize_division,
    normalize_doc_type,
    normalize_legal_area,
    normalize_number,
    normalize_nrg,
    normalize_year,
)
from .models import Citation, Span
from .parser import CitationParser, ECLI_RE
from .structured_urls import merge_structured_with_general


log = logging.getLogger(__name__)

CITATION_EXTRACTOR_PROVIDER = "openai"
CITATION_EXTRACTOR_METHOD = "openai_structured"
CITATION_EXTRACTOR_MODEL = os.environ.get(
    "CITATION_EXTRACTOR_MODEL",
    "gpt-5.6-luna",
).strip()
CITATION_EXTRACTOR_REASONING_EFFORT = os.environ.get(
    "CITATION_EXTRACTOR_REASONING_EFFORT",
    "none",
).strip()
CITATION_EXTRACTOR_MAX_TOKENS = int(
    os.environ.get("CITATION_EXTRACTOR_MAX_OUTPUT_TOKENS", "4000")
)
CITATION_EXTRACTOR_RETRIES = int(
    os.environ.get("CITATION_EXTRACTOR_MAX_RETRIES", "3")
)
OPENAI_EXTRACTOR_SDK_MAX_RETRIES = int(
    os.environ.get("OPENAI_EXTRACTOR_SDK_MAX_RETRIES", "0")
)
OPENAI_EXTRACTOR_TIMEOUT_SECONDS = float(
    os.environ.get("OPENAI_EXTRACTOR_TIMEOUT_SECONDS", "180")
)

DIAGNOSTIC_EXCERPT_MAX_CHARS = int(
    os.environ.get("CITATION_EXTRACTOR_DIAGNOSTIC_EXCERPT_MAX_CHARS", "500")
)
TOOL_NAME = "extract_case_law_citations"

TRANSIENT_ERROR_CATEGORIES = {
    "transport_error",
    "sdk_timeout",
}
CORRECTABLE_FORMAT_ERROR_CATEGORIES = {
    "missing_choice",
    "missing_message",
    "missing_tool_calls",
    "wrong_tool_name",
    "missing_arguments",
    "arguments_invalid_type",
    "arguments_invalid_json",
    "payload_not_object",
    "citations_missing",
    "citations_not_list",
    "citation_item_invalid",
}

RAW_TEXT_CATEGORIES = {
    "raw_text_exact",
    "raw_text_normalized_match",
    "raw_text_recovered_by_unique_identifier",
    "raw_text_not_verbatim",
    "raw_text_identifier_not_in_source",
    "raw_text_ambiguous_in_source",
    "raw_text_missing",
}

CITATION_KINDS = {
    "case_law",
    "explicit_ecli",
    "outside_index_scope",
}
JURISDICTION_TYPES = set(SUPPORTED_NAMESPACES) | set(OUTSIDE_INDEX_NAMESPACES)
VALID_DOC_TYPES = {"SENT", "ORD", "DEC"}
VALID_LEGAL_AREAS = {"CIV", "PEN"}
VALID_VENUES = set(VENUE_CODES.values())
JUDICIAL_OUTSIDE_AUTHORITIES = ("CEDU", "CORTE EDU", "CGUE", "CORTE DI GIUSTIZIA")

EXTRACTION_TOOL: dict[str, Any] = {
    "name": TOOL_NAME,
    "description": (
        "Extract judicial decision citations from an Italian legal answer. "
        "Do not extract statutes, code articles, circulars, doctrine, books, "
        "or generic references without a judicial decision."
    ),
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "citations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "raw_text": {"type": "string"},
                        "citation_kind": {
                            "type": "string",
                            "enum": sorted(CITATION_KINDS),
                        },
                        "jurisdiction_type": {
                            "anyOf": [
                                {
                                    "type": "string",
                                    "enum": sorted(JURISDICTION_TYPES),
                                },
                                {"type": "null"},
                            ]
                        },
                        "court_name": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "court_code": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "venue_name": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "venue_code": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "doc_type": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "year": {
                            "anyOf": [
                                {"type": "integer"},
                                {"type": "string"},
                                {"type": "null"},
                            ]
                        },
                        "number": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "nrg": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "legal_area": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "sector": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "division": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "explicit_ecli": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    },
                    "required": [
                        "raw_text",
                        "citation_kind",
                        "jurisdiction_type",
                        "year",
                        "number",
                    ],
                },
            }
        },
        "required": ["citations"],
    },
}

OPENAI_EXTRACTION_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": EXTRACTION_TOOL["description"],
        "parameters": EXTRACTION_TOOL["input_schema"],
    },
}


def _strict_extraction_schema() -> dict[str, Any]:
    """Return the extractor schema in the strict Structured Outputs shape."""
    schema = json.loads(json.dumps(EXTRACTION_TOOL["input_schema"]))
    item_schema = schema["properties"]["citations"]["items"]
    item_schema["required"] = list(item_schema["properties"])
    return schema


OPENAI_EXTRACTION_TEXT_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "name": TOOL_NAME,
    "strict": True,
    "schema": _strict_extraction_schema(),
}

SYSTEM_PROMPT = """
You extract only judicial decision citations from Italian legal answers.

Rules:
- Extract case-law decisions/provvedimenti only.
- Do not extract statutes, code articles, doctrine, circulars, books, legal principles without a cited decision, or generic mentions of a court.
- Do not invent missing fields. Use null when a field is absent.
- Fill legal_area only when the raw citation explicitly says civil/civ./civile, pen./penale, sezione civile/penale, Sez. Lav., or Sezioni Unite civili/penali.
- Fill doc_type only when the raw citation explicitly says sentenza/sent. or ordinanza/ord.; do not infer SENT from generic pronuncia, decisione, provvedimento, or Cass.
- Fill sector/division only when Sezioni Unite, Sez. Lav., or a specific section appears in the raw citation.
- raw_text must be copied exactly from the input: no abbreviation, no normalization, no reconstruction, no translation, no merged facts from different spans.
- If no identifiable source span exists, omit that citation item.
- Include explicit_ecli only when that ECLI appears inside raw_text; otherwise use null.
- Return [] when no judicial decision citation is present.
- CEDU/CGUE judicial decisions are outside_index_scope, not Italian registry citations.

Correct raw_text example:
Input: "Example Court, order n. 12/2020"
raw_text: "Example Court, order n. 12/2020"

Incorrect raw_text examples:
- "Example Court order 12/2020" because punctuation was rewritten.
- "order n. 12/2020 [Example Court]" because separate spans were merged.
- "Example Court, decision n. 12/2020" because the wording was normalized.
""".strip()


class CitationExtractionError(RuntimeError):
    """Raised when LLM citation extraction cannot return a valid payload."""

    def __init__(
        self,
        message: str,
        *,
        category: str = "transport_error",
        retryable: bool | None = None,
        status_code: int | None = None,
        request_id: str | None = None,
        retry_reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.retryable = (
            category in TRANSIENT_ERROR_CATEGORIES if retryable is None else retryable
        )
        self.status_code = status_code
        self.request_id = request_id
        self.retry_reason = retry_reason


@dataclass(frozen=True)
class RawTextResolution:
    span: Span | None
    text: str | None
    category: str
    method: str
    discard_reason: str | None = None


class OpenAICitationExtractor:
    provider = CITATION_EXTRACTOR_PROVIDER

    def __init__(
        self,
        *,
        client: Any | None = None,
        model: str = CITATION_EXTRACTOR_MODEL,
        reasoning_effort: str = CITATION_EXTRACTOR_REASONING_EFFORT,
        max_tokens: int = CITATION_EXTRACTOR_MAX_TOKENS,
        max_retries: int = CITATION_EXTRACTOR_RETRIES,
        sdk_max_retries: int = OPENAI_EXTRACTOR_SDK_MAX_RETRIES,
        timeout_seconds: float = OPENAI_EXTRACTOR_TIMEOUT_SECONDS,
    ) -> None:
        self.client = client
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.sdk_max_retries = sdk_max_retries
        self.timeout_seconds = timeout_seconds
        self.last_usage: dict[str, int] = {}
        self.total_usage: dict[str, int] = {}
        self.last_diagnostics: dict[str, Any] = {}
        self.attempt_diagnostics: list[dict[str, Any]] = []

    def extract(self, text: str, *, task_id: str | None = None) -> list[Citation]:
        if not text.strip():
            return []

        self.last_diagnostics = {}
        self.attempt_diagnostics = []
        last_error: Exception | None = None
        corrective_message: str | None = None
        corrective_retry_used = False
        source_supported_citation_count = _supported_source_citation_count(text)
        for attempt in range(1, self.max_retries + 1):
            response: Any | None = None
            payload: Any = None
            validation_diagnostics: list[dict[str, Any]] = []
            started_at = time.perf_counter()
            started_at_iso = _utc_now_iso()
            try:
                response = self._create_response(text, corrective_message=corrective_message)
                payload = _extract_tool_payload(response)
                self.last_usage = _extract_usage(response)
                self._add_usage(self.last_usage)
                citations = propagate_coordinated_context(
                    text,
                    _validated_citations(
                        text,
                        payload,
                        extraction_method=CITATION_EXTRACTOR_METHOD,
                        diagnostics=validation_diagnostics,
                    ),
                )
                citations = deduplicate_citations(citations)
                category = "success" if citations else "empty_valid_result"
                self._record_diagnostics(
                    task_id=task_id,
                    attempt=attempt,
                    started_at_iso=started_at_iso,
                    duration_ms=latency_ms(started_at),
                    response=response,
                    payload=payload,
                    error=None,
                    error_category=category,
                    retry_reason=None,
                    validation_diagnostics=validation_diagnostics,
                    source_supported_citation_count=source_supported_citation_count,
                )
                return citations
            except Exception as exc:
                error = _coerce_extraction_error(exc)
                last_error = error
                retry_reason = _retry_reason(
                    error,
                    attempt=attempt,
                    max_retries=self.max_retries,
                    corrective_retry_used=corrective_retry_used,
                )
                self._record_diagnostics(
                    task_id=task_id,
                    attempt=attempt,
                    started_at_iso=started_at_iso,
                    duration_ms=latency_ms(started_at),
                    response=response,
                    payload=payload,
                    error=str(error),
                    error_category=error.category,
                    retry_reason=retry_reason,
                    exception=error,
                    validation_diagnostics=validation_diagnostics,
                    source_supported_citation_count=source_supported_citation_count,
                )
                if retry_reason is None:
                    break
                if error.category in CORRECTABLE_FORMAT_ERROR_CATEGORIES:
                    corrective_retry_used = True
                    corrective_message = _corrective_message(error.category)
                else:
                    corrective_message = None
                if attempt < self.max_retries:
                    time.sleep(0.5 * (2 ** (attempt - 1)))

        if isinstance(last_error, CitationExtractionError):
            raise CitationExtractionError(
                str(last_error),
                category=last_error.category,
                retryable=last_error.retryable,
                status_code=last_error.status_code,
                request_id=last_error.request_id,
                retry_reason=last_error.retry_reason,
            )
        raise CitationExtractionError(str(last_error) if last_error else "unknown error")

    def _record_diagnostics(
        self,
        *,
        task_id: str | None,
        attempt: int,
        started_at_iso: str,
        duration_ms: int,
        response: Any | None,
        payload: Any,
        error: str | None,
        error_category: str,
        retry_reason: str | None,
        exception: CitationExtractionError | None = None,
        validation_diagnostics: list[dict[str, Any]] | None = None,
        source_supported_citation_count: int | None = None,
    ) -> None:
        diagnostics = extractor_response_diagnostics(
            response=response,
            provider=CITATION_EXTRACTOR_PROVIDER,
            model=self.model,
            payload=payload,
            attempt=attempt,
            error=error,
            task_id=task_id,
            started_at_iso=started_at_iso,
            duration_ms=duration_ms,
            error_category=error_category,
            retry_reason=retry_reason,
            exception=exception,
            validation_diagnostics=validation_diagnostics or [],
            source_supported_citation_count=source_supported_citation_count,
        )
        self.last_diagnostics = diagnostics
        self.attempt_diagnostics.append(diagnostics)

    def _add_usage(self, usage: dict[str, Any]) -> None:
        for key, value in usage.items():
            if isinstance(value, int):
                self.total_usage[key] = self.total_usage.get(key, 0) + value

    def _client(self) -> Any:
        if self.client is None:
            from openai import OpenAI

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY non impostata in .env")
            self.client = OpenAI(
                api_key=api_key,
                max_retries=self.sdk_max_retries,
                timeout=self.timeout_seconds,
            )
        return self.client

    def _create_response(self, text: str, *, corrective_message: str | None = None) -> Any:
        user_content = f"Extract judicial decision citations from this text:\n\n{text}"
        if corrective_message:
            user_content = f"{corrective_message}\n\n{user_content}"
        client = self._client()
        responses = getattr(client, "responses", None)
        if responses is not None and hasattr(responses, "create"):
            return responses.create(
                model=self.model,
                instructions=SYSTEM_PROMPT,
                input=user_content,
                max_output_tokens=self.max_tokens,
                reasoning={"effort": self.reasoning_effort},
                text={"format": OPENAI_EXTRACTION_TEXT_FORMAT},
            )

        # Compatibility for injected OpenAI-compatible test clients. Production
        # uses the Responses API branch above.
        return client.chat.completions.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            reasoning_effort=self.reasoning_effort,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            tools=[OPENAI_EXTRACTION_TOOL],
            tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
        )


def _extract_tool_payload(response: Any) -> dict[str, Any]:
    responses_payload = _extract_openai_responses_payload(response)
    if responses_payload is not None:
        return responses_payload

    openai_payload = _extract_openai_tool_payload(response)
    if openai_payload is not None:
        return openai_payload

    content = _get_value(response, "content") or []
    wrong_tool_names: list[str] = []
    for block in content:
        block_type = _get_value(block, "type")
        block_name = _get_value(block, "name")
        if block_type == "tool_use" and block_name != TOOL_NAME:
            wrong_tool_names.append(str(block_name))
            continue
        if block_type == "tool_use" and block_name == TOOL_NAME:
            payload = _get_value(block, "input")
            if isinstance(payload, dict):
                return payload
            raise CitationExtractionError(
                "tool payload is not an object",
                category="payload_not_object",
                retryable=False,
            )
    if wrong_tool_names:
        raise CitationExtractionError(
            "wrong extraction tool name: " + ", ".join(wrong_tool_names),
            category="wrong_tool_name",
            retryable=False,
        )

    text_parts = [
        str(_get_value(block, "text"))
        for block in content
        if _get_value(block, "text") is not None
    ]
    if text_parts:
        payload = _strict_json_object_or_none("\n".join(text_parts))
        if payload is not None:
            return payload

    raise CitationExtractionError(
        "missing extraction tool payload",
        category="missing_tool_calls",
        retryable=False,
    )


def _extract_openai_responses_payload(response: Any) -> dict[str, Any] | None:
    """Extract strict JSON output from an OpenAI Responses API object."""
    output_present = _has_value(response, "output") or _has_value(response, "output_text")
    if not output_present:
        return None

    output_text = _get_value(response, "output_text")
    if isinstance(output_text, str) and output_text.strip():
        payload = _strict_json_object_or_none(output_text)
        if payload is not None:
            return payload

    wrong_function_names: list[str] = []
    text_parts: list[str] = []
    for item in _get_value(response, "output") or []:
        item_type = _get_value(item, "type")
        if item_type == "function_call":
            name = _get_value(item, "name")
            if name != TOOL_NAME:
                wrong_function_names.append(str(name))
                continue
            arguments = _get_value(item, "arguments")
            if arguments is None:
                raise CitationExtractionError(
                    "missing extraction tool arguments",
                    category="missing_arguments",
                    retryable=False,
                )
            return _payload_from_openai_arguments(arguments)
        for block in _get_value(item, "content") or []:
            text_value = _get_value(block, "text")
            if text_value is not None:
                text_parts.append(str(text_value))

    if text_parts:
        payload = _strict_json_object_or_none("\n".join(text_parts))
        if payload is not None:
            return payload
    if wrong_function_names:
        raise CitationExtractionError(
            "wrong extraction tool name: " + ", ".join(wrong_function_names),
            category="wrong_tool_name",
            retryable=False,
        )
    raise CitationExtractionError(
        "missing extraction structured output",
        category="missing_tool_calls",
        retryable=False,
    )


def _extract_openai_tool_payload(response: Any) -> dict[str, Any] | None:
    choices_present = _has_value(response, "choices")
    choices = _get_value(response, "choices") or []
    if not choices_present:
        return None
    if not choices:
        raise CitationExtractionError("missing response choice", category="missing_choice", retryable=False)

    saw_message = False
    wrong_tool_names: list[str] = []
    saw_tool_calls = False
    for choice in choices:
        message = _get_value(choice, "message")
        if message is None:
            continue
        saw_message = True
        for tool_call in _get_value(message, "tool_calls") or []:
            saw_tool_calls = True
            function = _get_value(tool_call, "function") or {}
            function_name = _get_value(function, "name")
            if function_name != TOOL_NAME:
                wrong_tool_names.append(str(function_name))
                continue
            arguments = _get_value(function, "arguments")
            if arguments is None:
                raise CitationExtractionError(
                    "missing extraction tool arguments",
                    category="missing_arguments",
                    retryable=False,
                )
            payload = _payload_from_openai_arguments(arguments)
            return payload

        content = _get_value(message, "content")
        if isinstance(content, str) and content.strip():
            payload = _strict_json_object_or_none(content)
            if payload is not None:
                return payload
    if not saw_message:
        raise CitationExtractionError("missing response message", category="missing_message", retryable=False)
    if saw_tool_calls:
        raise CitationExtractionError(
            "wrong extraction tool name: " + ", ".join(wrong_tool_names),
            category="wrong_tool_name",
            retryable=False,
        )
    raise CitationExtractionError(
        "missing extraction tool payload",
        category="missing_tool_calls",
        retryable=False,
    )


def _payload_from_openai_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            payload = _strict_json_object(arguments)
        except json.JSONDecodeError as exc:
            raise CitationExtractionError(
                "tool arguments are not valid JSON",
                category="arguments_invalid_json",
                retryable=False,
            ) from exc
        if isinstance(payload, dict):
            return payload
        raise CitationExtractionError(
            "tool payload is not an object",
            category="payload_not_object",
            retryable=False,
        )
    raise CitationExtractionError(
        "tool arguments have unsupported type",
        category="arguments_invalid_type",
        retryable=False,
    )


def _ensure_payload_shape(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise CitationExtractionError(
            "tool payload is not an object",
            category="payload_not_object",
            retryable=False,
        )
    if "citations" not in payload:
        raise CitationExtractionError(
            "payload.citations is missing",
            category="citations_missing",
            retryable=False,
        )
    if not isinstance(payload.get("citations"), list):
        raise CitationExtractionError(
            "payload.citations is not a list",
            category="citations_not_list",
            retryable=False,
        )


def _strip_outer_json_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(?P<body>.*?)\s*```", stripped, re.DOTALL | re.IGNORECASE)
    if not match:
        return stripped
    return match.group("body").strip()


def _strict_json_object(text: str) -> dict[str, Any]:
    parsed = json.loads(_strip_outer_json_fence(text))
    if not isinstance(parsed, dict):
        raise CitationExtractionError(
            "tool payload is not an object",
            category="payload_not_object",
            retryable=False,
        )
    return parsed


def _strict_json_object_or_none(text: str) -> dict[str, Any] | None:
    try:
        return _strict_json_object(text)
    except (json.JSONDecodeError, CitationExtractionError):
        return None


def _extract_usage(response: Any) -> dict[str, Any]:
    usage = _get_value(response, "usage")
    if usage is None:
        return {}

    result: dict[str, Any] = {}
    for key in (
        "input_tokens",
        "output_tokens",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "reasoning_tokens",
    ):
        value = _get_value(usage, key)
        if isinstance(value, int):
            result[key] = value
    for key in (
        "prompt_tokens_details",
        "completion_tokens_details",
        "output_tokens_details",
    ):
        value = _get_value(usage, key)
        details = _to_builtin(value)
        if isinstance(details, dict):
            result[key] = details
    return result


def latency_ms(started_at: float) -> int:
    return int(round((time.perf_counter() - started_at) * 1000))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _openai_message_diagnostics(response: Any | None) -> dict[str, Any]:
    tool_call_count = 0
    function_names: list[str | None] = []
    argument_type: str | None = None
    argument_length: int | None = None
    content_length = 0

    if response is not None:
        for choice in _get_value(response, "choices") or []:
            message = _get_value(choice, "message")
            if message is None:
                continue
            content = _get_value(message, "content")
            if isinstance(content, str):
                content_length += len(content)
            for tool_call in _get_value(message, "tool_calls") or []:
                tool_call_count += 1
                function = _get_value(tool_call, "function") or {}
                function_names.append(_get_value(function, "name"))
                arguments = _get_value(function, "arguments")
                if argument_type is None:
                    argument_type = _type_name(arguments)
                    argument_length = _value_length(arguments)

    return {
        "tool_call_count": tool_call_count,
        "tool_call_function_name": next(
            (name for name in function_names if name == TOOL_NAME),
            function_names[0] if function_names else None,
        ),
        "tool_call_function_names": [str(name) for name in function_names if name is not None],
        "function_arguments_type": argument_type,
        "function_arguments_length": argument_length,
        "message_content_length": content_length,
    }


def _response_status_code(
    response: Any | None,
    exception: CitationExtractionError | None = None,
) -> int | None:
    if exception is not None and exception.status_code is not None:
        return exception.status_code
    status = _get_value(response, "status_code") if response is not None else None
    return status if isinstance(status, int) else None


def _response_request_id(
    response: Any | None,
    exception: CitationExtractionError | None = None,
) -> str | None:
    if exception is not None and exception.request_id:
        return exception.request_id
    if response is None:
        return None
    for name in ("_request_id", "request_id", "id"):
        value = _get_value(response, name)
        if value:
            return str(value)
    return None


def _reasoning_tokens_from_usage(usage: dict[str, Any]) -> int | None:
    for key in ("completion_tokens_details", "output_tokens_details"):
        details = usage.get(key)
        if isinstance(details, dict):
            value = details.get("reasoning_tokens")
            if isinstance(value, int):
                return value
    value = usage.get("reasoning_tokens")
    return value if isinstance(value, int) else None


def _empty_result_category(
    *,
    payload_citation_count: int | None,
    validated_citation_count: int,
    source_supported_citation_count: int | None,
) -> str | None:
    if validated_citation_count > 0:
        return None
    if payload_citation_count == 0:
        if source_supported_citation_count == 0:
            return "empty_correct_no_supported_citations"
        return "empty_model_returned_no_items"
    if payload_citation_count and payload_citation_count > 0:
        return "empty_after_validation"
    return None


def _supported_source_citation_count(text: str) -> int:
    try:
        return len(
            [
                citation
                for citation in CitationParser().parse(text)
                if citation.suggested_namespace in SUPPORTED_NAMESPACES
                or _is_judicial_outside_scope_citation(citation)
                or citation.ecli
            ]
        )
    except Exception:
        return 0


def _is_judicial_outside_scope_citation(citation: Citation) -> bool:
    if not citation.outside_index_scope:
        return False
    authority = (citation.authority or "").upper()
    return any(token in authority for token in JUDICIAL_OUTSIDE_AUTHORITIES)


def extractor_response_diagnostics(
    *,
    response: Any | None,
    provider: str,
    model: str,
    payload: Any,
    attempt: int | None = None,
    error: str | None = None,
    task_id: str | None = None,
    started_at_iso: str | None = None,
    duration_ms: int | None = None,
    error_category: str | None = None,
    retry_reason: str | None = None,
    exception: CitationExtractionError | None = None,
    validation_diagnostics: list[dict[str, Any]] | None = None,
    source_supported_citation_count: int | None = None,
) -> dict[str, Any]:
    content = _get_value(response, "content") or [] if response is not None else []
    content_blocks = content if isinstance(content, list) else []
    citations_value = payload.get("citations") if isinstance(payload, dict) else None
    usage = _extract_usage(response)
    openai_diagnostics = _openai_message_diagnostics(response)
    finished_at_iso = _utc_now_iso()
    validation_diagnostics = validation_diagnostics or []
    payload_citation_count = len(citations_value) if isinstance(citations_value, list) else None
    validated_citation_count = sum(
        1 for item in validation_diagnostics if item.get("accepted")
    )
    discarded_citation_count = sum(
        1 for item in validation_diagnostics if item.get("accepted") is False
    )
    empty_result_category = _empty_result_category(
        payload_citation_count=payload_citation_count,
        validated_citation_count=validated_citation_count,
        source_supported_citation_count=source_supported_citation_count,
    )
    output_tokens = usage.get("output_tokens") or usage.get("completion_tokens")
    token_per_payload_citation = (
        (float(output_tokens) / payload_citation_count)
        if isinstance(output_tokens, int) and payload_citation_count
        else None
    )
    diagnostic: dict[str, Any] = {
        "task_id": task_id,
        "extractor_provider": provider,
        "extractor_model": model,
        "attempt": attempt,
        "attempt_started_at": started_at_iso,
        "attempt_finished_at": finished_at_iso,
        "duration_ms": duration_ms,
        "status_code": _response_status_code(response, exception),
        "request_id": _response_request_id(response, exception),
        "error_category": error_category,
        "retry_reason": retry_reason,
        "response_stop_reason": _response_stop_reason(response),
        "finish_reason": _response_stop_reason(response),
        "usage_input_tokens": usage.get("input_tokens") or usage.get("prompt_tokens"),
        "usage_output_tokens": usage.get("output_tokens") or usage.get("completion_tokens"),
        "usage_total_tokens": usage.get("total_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "completion_tokens_details": usage.get("completion_tokens_details") or {},
        "reasoning_tokens": _reasoning_tokens_from_usage(usage),
        "response_content_block_types": [
            str(_get_value(block, "type") or type(block).__name__) for block in content_blocks
        ]
        or _openai_response_block_types(response),
        **openai_diagnostics,
        "payload_type": _type_name(payload),
        "citations_value_type": _type_name(citations_value),
        "payload_citation_count": payload_citation_count,
        "validated_citation_count": validated_citation_count,
        "discarded_citation_count": discarded_citation_count,
        "source_supported_citation_count": source_supported_citation_count,
        "token_per_payload_citation": token_per_payload_citation,
        "empty_result_category": empty_result_category,
        "validation_diagnostics": validation_diagnostics,
        "raw_response_excerpt": _raw_response_excerpt(response, payload),
    }
    if error:
        diagnostic["validation_error"] = error
    return diagnostic


def _validated_citations(
    text: str,
    payload: dict[str, Any],
    *,
    extraction_method: str = CITATION_EXTRACTOR_METHOD,
    diagnostics: list[dict[str, Any]] | None = None,
) -> list[Citation]:
    _ensure_payload_shape(payload)
    raw_items = payload.get("citations")

    citations: list[Citation] = []
    for item in raw_items:
        if not isinstance(item, dict):
            raise CitationExtractionError(
                "citation item is not an object",
                category="citation_item_invalid",
                retryable=False,
            )
        citation = _validated_citation(
            text,
            item,
            extraction_method=extraction_method,
            diagnostics=diagnostics,
        )
        if citation is not None:
            citations.extend(_split_composite_citation(citation))
    return citations


def propagate_coordinated_context(text: str, citations: Iterable[Citation]) -> list[Citation]:
    """Conservatively inherit court metadata across coordinated short citations."""
    ordered = sorted(citations, key=lambda citation: citation.span)
    propagated: list[Citation] = []
    context: Citation | None = None

    for citation in ordered:
        if context is not None and _can_inherit_context(text, context, citation):
            citation = _inherit_context(context, citation)

        propagated.append(citation)
        if _starts_new_citation_context(citation):
            context = citation
        elif citation.context_inherited:
            context = citation
        elif context is not None and _breaks_context(text, context.span[1], citation.span[0]):
            context = None

    return propagated


def _starts_new_citation_context(citation: Citation) -> bool:
    return bool(
        citation.suggested_namespace
        and citation.court
        and (citation.legal_area or citation.division or citation.sector or citation.doc_type)
        and re.search(r"\b(?:Cass\.?|Cassazione|Corte\s+di\s+cassazione|SS\.?UU\.?)\b", citation.text, re.IGNORECASE)
    )


def _can_inherit_context(text: str, context: Citation, citation: Citation) -> bool:
    if not _is_short_coordinated_citation(citation):
        return False
    if _breaks_context(text, context.span[1], citation.span[0]):
        return False
    if _has_explicit_context_override(citation):
        return False
    return True


def _is_short_coordinated_citation(citation: Citation) -> bool:
    raw = citation.text.strip()
    return bool(
        re.match(r"^(?:e\s+)?(?:n\.|num\.|numero)\s*\d+", raw, re.IGNORECASE)
        and not re.search(r"\b(?:Cass\.?|Cassazione|Corte\s+di\s+cassazione|SS\.?UU\.?|Sez\.|Sezione)\b", raw, re.IGNORECASE)
    )


def _breaks_context(text: str, previous_end: int, current_start: int) -> bool:
    between = text[previous_end:current_start]
    if "\n\n" in between:
        return True
    if re.search(r"\.\s+", between):
        return True
    return not re.fullmatch(r"[\s,;:/()\-–—]*(?:e\s+|ed\s+|nonché\s+)?[\s,;:/()\-–—]*", between, re.IGNORECASE)


def _has_explicit_context_override(citation: Citation) -> bool:
    raw = _ascii_text(citation.text)
    if re.search(r"\bCASS(?:AZIONE)?\s+PEN\b", raw) or re.search(r"\bCASS(?:AZIONE)?\s+CIV\b", raw):
        return True
    if re.search(r"\bSS\s*UU\b", raw) or re.search(r"\bSEZ(?:IONE)?\b", raw):
        return True
    return False


def _inherit_context(context: Citation, citation: Citation) -> Citation:
    updates: dict[str, Any] = {}
    inherited_fields: list[str] = []
    metadata_context = dict(citation.metadata_context)

    for field_name in ("authority", "suggested_namespace", "court", "court_name", "jurisdiction_type"):
        if getattr(citation, field_name) is None and getattr(context, field_name) is not None:
            updates[field_name] = getattr(context, field_name)
            inherited_fields.append(field_name)

    for field_name in ("legal_area", "division", "sector", "doc_type"):
        if getattr(citation, field_name) is None and getattr(context, field_name) is not None:
            value = getattr(context, field_name)
            updates[field_name] = value
            inherited_fields.append(field_name)
            metadata_context[field_name] = {"value": value, "source": "inherited"}

    if not inherited_fields:
        return citation

    return replace(
        citation,
        **updates,
        context_inherited=True,
        inherited_fields=tuple(dict.fromkeys((*citation.inherited_fields, *inherited_fields))),
        inherited_from_span=context.span,
        metadata_context=metadata_context,
    )


def _append_validation_diagnostic(
    diagnostics: list[dict[str, Any]] | None,
    *,
    item: dict[str, Any],
    proposed_raw_text: str | None,
    raw_text_category: str,
    accepted: bool,
    recovered_span: Span | None = None,
    recovered_raw_text: str | None = None,
    recovery_method: str | None = None,
    discard_reason: str | None = None,
) -> None:
    if diagnostics is None:
        return
    diagnostics.append(
        {
            "accepted": accepted,
            "raw_text_category": raw_text_category,
            "proposed_raw_text_excerpt": (
                _truncate_excerpt(_sanitize_excerpt(proposed_raw_text))
                if proposed_raw_text
                else None
            ),
            "recovery_method": recovery_method,
            "recovered_span": list(recovered_span) if recovered_span else None,
            "recovered_raw_text_excerpt": (
                _truncate_excerpt(_sanitize_excerpt(recovered_raw_text))
                if recovered_raw_text
                else None
            ),
            "discard_reason": discard_reason,
            "number": normalize_number(item.get("number")),
            "year": normalize_year(item.get("year")),
            "jurisdiction_type": _clean_optional_string(item.get("jurisdiction_type")),
        }
    )


def _resolve_raw_text(text: str, raw_text: str, item: dict[str, Any]) -> RawTextResolution:
    span = _find_exact_span(text, raw_text)
    if span is not None:
        return RawTextResolution(
            span=span,
            text=text[span[0] : span[1]],
            category="raw_text_exact",
            method="exact",
        )

    typographic = _find_typographic_span(text, raw_text)
    if typographic.category == "raw_text_normalized_match":
        return typographic
    if typographic.category == "raw_text_ambiguous_in_source":
        return typographic

    identifier = _identifier_from_item_or_text(item, raw_text)
    if identifier is None:
        return RawTextResolution(
            span=None,
            text=None,
            category="raw_text_not_verbatim",
            method="identifier",
            discard_reason="missing_number_or_year",
        )

    number, year, namespace = identifier
    if not _identifier_occurs_in_source(text, number=number, year=year):
        return RawTextResolution(
            span=None,
            text=None,
            category="raw_text_identifier_not_in_source",
            method="identifier",
            discard_reason="number_or_year_absent",
        )

    candidates = [
        citation
        for citation in CitationParser().parse(text)
        if citation.number == number
        and citation.year == year
        and _namespace_compatible(namespace, citation)
    ]
    unique_spans = {
        span
        for citation in candidates
        for span in (citation.spans or (citation.span,))
    }
    if len(unique_spans) == 1:
        recovered_span = next(iter(unique_spans))
        return RawTextResolution(
            span=recovered_span,
            text=text[recovered_span[0] : recovered_span[1]],
            category="raw_text_recovered_by_unique_identifier",
            method="unique_identifier",
        )
    if len(unique_spans) > 1:
        return RawTextResolution(
            span=None,
            text=None,
            category="raw_text_ambiguous_in_source",
            method="unique_identifier",
            discard_reason="multiple_identifier_matches",
        )
    return RawTextResolution(
        span=None,
        text=None,
        category="raw_text_not_verbatim",
        method="unique_identifier",
        discard_reason="no_compatible_span",
    )


def _find_typographic_span(text: str, raw_text: str) -> RawTextResolution:
    normalized_text, index_map = _typographic_normalize_with_map(text)
    normalized_raw, _ = _typographic_normalize_with_map(raw_text)
    if not normalized_raw:
        return RawTextResolution(
            span=None,
            text=None,
            category="raw_text_not_verbatim",
            method="typographic",
            discard_reason="empty_normalized_raw_text",
        )
    starts = [match.start() for match in re.finditer(re.escape(normalized_raw), normalized_text)]
    if len(starts) > 1:
        return RawTextResolution(
            span=None,
            text=None,
            category="raw_text_ambiguous_in_source",
            method="typographic",
            discard_reason="multiple_typographic_matches",
        )
    if len(starts) == 1:
        start_norm = starts[0]
        end_norm = start_norm + len(normalized_raw) - 1
        start = index_map[start_norm]
        end = index_map[end_norm] + 1
        return RawTextResolution(
            span=(start, end),
            text=text[start:end],
            category="raw_text_normalized_match",
            method="typographic",
        )
    return RawTextResolution(
        span=None,
        text=None,
        category="raw_text_not_verbatim",
        method="typographic",
        discard_reason="no_typographic_match",
    )


def _typographic_normalize_with_map(value: str) -> tuple[str, list[int]]:
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
    }
    chars: list[str] = []
    index_map: list[int] = []
    previous_space = False
    for index, char in enumerate(value):
        normalized = replacements.get(char, char)
        for item in normalized:
            if item.isspace():
                if previous_space:
                    continue
                chars.append(" ")
                index_map.append(index)
                previous_space = True
            else:
                chars.append(item)
                index_map.append(index)
                previous_space = False
    normalized = "".join(chars).strip()
    if not normalized:
        return "", []
    leading_trim = len("".join(chars)) - len("".join(chars).lstrip())
    if leading_trim:
        index_map = index_map[leading_trim:]
    return normalized, index_map[: len(normalized)]


def _identifier_from_item_or_text(
    item: dict[str, Any],
    raw_text: str,
) -> tuple[str, int, str | None] | None:
    number = normalize_number(item.get("number"))
    year = normalize_year(item.get("year"))
    if number is None or year is None:
        parsed = _parse_number_year(raw_text)
        if parsed is not None:
            number = number or parsed[0]
            year = year or parsed[1]
    if number is None or year is None:
        return None
    namespace = _clean_optional_string(item.get("jurisdiction_type"))
    namespace = namespace.upper() if namespace else None
    return number, year, namespace


def _parse_number_year(value: str) -> tuple[str, int] | None:
    patterns = (
        r"(?:n\.|num\.|numero)\s*(?P<number>\d+)\s*(?:/|\s+del\s+)\s*(?P<year>\d{4})",
        r"(?P<number>\d+)\s*/\s*(?P<year>\d{4})",
    )
    for pattern in patterns:
        match = re.search(pattern, value, re.IGNORECASE)
        if not match:
            continue
        number = normalize_number(match.group("number"))
        year = normalize_year(match.group("year"))
        if number is not None and year is not None:
            return number, year
    return None


def _identifier_occurs_in_source(text: str, *, number: str, year: int) -> bool:
    number_pattern = re.compile(rf"(?<!\d)0*{re.escape(number)}(?!\d)")
    year_pattern = re.compile(rf"(?<!\d){year}(?!\d)")
    return bool(number_pattern.search(text) and year_pattern.search(text))


def _namespace_compatible(namespace: str | None, citation: Citation) -> bool:
    if namespace is None:
        return True
    observed = (citation.suggested_namespace or citation.jurisdiction_type or "").upper()
    if namespace in OUTSIDE_INDEX_NAMESPACES:
        return citation.outside_index_scope
    return observed == namespace


def _validated_citation(
    text: str,
    item: dict[str, Any],
    *,
    extraction_method: str,
    diagnostics: list[dict[str, Any]] | None = None,
) -> Citation | None:
    warnings: list[str] = []
    raw_text = _clean_optional_string(item.get("raw_text"))
    if not raw_text:
        _append_validation_diagnostic(
            diagnostics,
            item=item,
            proposed_raw_text=None,
            raw_text_category="raw_text_missing",
            accepted=False,
            discard_reason="raw_text_missing",
        )
        log.warning("Citation extractor discarded item without raw_text")
        return None

    raw_resolution = _resolve_raw_text(text, raw_text, item)
    if raw_resolution.span is None or raw_resolution.text is None:
        _append_validation_diagnostic(
            diagnostics,
            item=item,
            proposed_raw_text=raw_text,
            raw_text_category=raw_resolution.category,
            accepted=False,
            discard_reason=raw_resolution.discard_reason,
        )
        log.warning(
            "Citation extractor discarded raw_text (%s): %r",
            raw_resolution.category,
            raw_text,
        )
        return None
    if raw_resolution.category != "raw_text_exact":
        warnings.append(f"raw_text_recovery={raw_resolution.category}")
    raw_text = raw_resolution.text

    citation_kind = _clean_optional_string(item.get("citation_kind"))
    if citation_kind not in CITATION_KINDS:
        warnings.append(f"invalid citation_kind={citation_kind!r}")
        log.warning("Citation extractor discarded unsupported kind: %r", citation_kind)
        return None

    jurisdiction_type = _clean_optional_string(item.get("jurisdiction_type"))
    jurisdiction_type = jurisdiction_type.upper() if jurisdiction_type else None
    if jurisdiction_type not in JURISDICTION_TYPES:
        if jurisdiction_type is not None:
            warnings.append(f"invalid jurisdiction_type={jurisdiction_type!r}")
        jurisdiction_type = None

    explicit_ecli = _clean_optional_string(item.get("explicit_ecli"))
    ecli = None
    if explicit_ecli:
        explicit_ecli = explicit_ecli.upper()
        if explicit_ecli not in raw_text.upper():
            warnings.append("explicit_ecli_not_in_raw_text")
        elif not ECLI_RE.fullmatch(explicit_ecli):
            warnings.append("invalid_explicit_ecli")
        else:
            ecli = explicit_ecli

    namespace_from_ecli = infer_namespace_from_ecli(ecli) if ecli else None
    outside_index_scope = jurisdiction_type in OUTSIDE_INDEX_NAMESPACES
    namespace = namespace_from_ecli
    if namespace is None and jurisdiction_type in SUPPORTED_NAMESPACES:
        namespace = jurisdiction_type

    court_name = _clean_optional_string(item.get("court_name"))
    venue_name = _clean_optional_string(item.get("venue_name"))
    authority = court_name or venue_name
    derived_court, derived_venue = derive_authority_codes(
        " ".join(part for part in (court_name, venue_name) if part),
        namespace=namespace,
    )

    division = _explicit_division(raw_text, item.get("division"), warnings)
    sector = _explicit_cass_sector(raw_text, item.get("sector"), warnings)

    court_code = _validated_court_code(
        item.get("court_code"),
        namespace=namespace,
        derived=derived_court,
        warnings=warnings,
    )
    venue_code = _validated_venue_code(
        item.get("venue_code"),
        derived=derived_venue,
        warnings=warnings,
    )

    doc_type = _explicit_doc_type(raw_text, item.get("doc_type"), warnings)

    year = normalize_year(item.get("year"))
    if item.get("year") is not None and year is None:
        warnings.append(f"invalid_year={item.get('year')!r}")

    number = normalize_number(item.get("number"))
    if item.get("number") is not None and number is None:
        warnings.append(f"invalid_number={item.get('number')!r}")

    nrg = normalize_nrg(item.get("nrg"))
    if item.get("nrg") is not None and nrg is None:
        warnings.append(f"invalid_nrg={item.get('nrg')!r}")

    legal_area = _explicit_legal_area(raw_text, item.get("legal_area"), warnings)

    if namespace is None and not outside_index_scope and ecli is None:
        warnings.append("missing_supported_namespace")
        log.warning("Citation extractor discarded unsupported citation: %r", raw_text)
        _append_validation_diagnostic(
            diagnostics,
            item=item,
            proposed_raw_text=raw_text,
            raw_text_category=raw_resolution.category,
            accepted=False,
            recovered_span=raw_resolution.span,
            recovered_raw_text=raw_resolution.text,
            recovery_method=raw_resolution.method,
            discard_reason="unsupported_citation",
        )
        return None

    _append_validation_diagnostic(
        diagnostics,
        item=item,
        proposed_raw_text=raw_text,
        raw_text_category=raw_resolution.category,
        accepted=True,
        recovered_span=raw_resolution.span,
        recovered_raw_text=raw_resolution.text,
        recovery_method=raw_resolution.method,
    )

    return Citation(
        text=raw_resolution.text,
        span=raw_resolution.span,
        ecli=ecli,
        citation_kind=citation_kind,
        authority=authority,
        suggested_namespace=namespace,
        number=number,
        year=year,
        legal_area=legal_area,
        court=court_code,
        court_name=court_name,
        venue=venue_code,
        venue_name=venue_name,
        doc_type=doc_type,
        jurisdiction_type=jurisdiction_type or namespace,
        nrg=nrg,
        sector=sector,
        division=division,
        outside_index_scope=outside_index_scope,
        extraction_method=extraction_method,
        extraction_warnings=tuple(warnings),
    )


def _validated_court_code(
    value: Any,
    *,
    namespace: str | None,
    derived: str | None,
    warnings: list[str],
) -> str | None:
    code = normalize_code(value)
    if code is None:
        return derived

    if namespace in {"CASS", "COST", "CONT", "ABF", "COVIP"}:
        if namespace == "CASS" and code in {"CASS", "CASSSU", "CASSLAV"}:
            return "CASS"
        if code == namespace:
            return code
        warnings.append(f"untrusted_court_code={code!r}")
        return derived or namespace

    if namespace == "MER" and _is_valid_venue_scoped_code(code, {"TR", "CA"}):
        return code
    if namespace == "ADM" and (code == "CDS" or _is_valid_venue_scoped_code(code, {"TAR"})):
        return code
    if namespace == "TAX" and _is_valid_venue_scoped_code(code, {"CG1", "CG2"}):
        return code

    warnings.append(f"untrusted_court_code={code!r}")
    return derived


def _validated_venue_code(
    value: Any,
    *,
    derived: str | None,
    warnings: list[str],
) -> str | None:
    code = normalize_code(value)
    if code is None:
        return derived
    if code in VALID_VENUES or code in {"CASS", "COST", "CONT", "ABF", "COVIP"}:
        return code
    warnings.append(f"untrusted_venue_code={code!r}")
    return derived


def _is_valid_venue_scoped_code(code: str, prefixes: set[str]) -> bool:
    for prefix in prefixes:
        if code.startswith(prefix) and code[len(prefix) :] in VALID_VENUES:
            return True
    return False


def _explicit_doc_type(raw_text: str, value: Any, warnings: list[str]) -> str | None:
    explicit = _doc_type_from_text(raw_text)
    provided = normalize_doc_type(value)
    if value is not None and provided is None:
        warnings.append(f"invalid_doc_type={value!r}")
    if provided is not None and explicit is None:
        warnings.append("doc_type_not_explicit")
        return None
    if provided is not None and explicit is not None and provided != explicit:
        warnings.append(f"doc_type_conflicts_with_raw_text={value!r}")
    return explicit


def _doc_type_from_text(raw_text: str) -> str | None:
    text = _ascii_text(raw_text)
    if re.search(r"\bORD(?:INANZA|INANZE)?\b", text) or re.search(r"\bORD\b", text):
        return "ORD"
    if re.search(r"\bSENT(?:ENZA|ENZE)?\b", text) or re.search(r"\bSENT\b", text):
        return "SENT"
    if re.search(r"\bDECR(?:ETO|ETI)?\b", text) or re.search(r"\bDEC\b", text):
        return "DEC"
    return None


def _explicit_legal_area(raw_text: str, value: Any, warnings: list[str]) -> str | None:
    explicit = _legal_area_from_text(raw_text)
    provided = normalize_legal_area(value)
    if value is not None and provided is None:
        warnings.append(f"invalid_legal_area={value!r}")
    if provided is not None and explicit is None:
        warnings.append("legal_area_not_explicit")
        return None
    if provided is not None and explicit is not None and provided != explicit:
        warnings.append(f"legal_area_conflicts_with_raw_text={value!r}")
    return explicit


def _legal_area_from_text(raw_text: str) -> str | None:
    text = _ascii_text(raw_text)
    if (
        re.search(r"\bCASS(?:AZIONE)?\s+CIV(?:ILE|ILI)?\b", text)
        or re.search(r"\bCASS\s+CIV\b", text)
        or re.search(r"\bSEZ(?:IONE|IONI)?(?:\s+[A-Z0-9]+){0,4}\s+CIV(?:ILE|ILI)?\b", text)
        or re.search(r"\bSEZ(?:IONE)?\s+LAV(?:ORO)?\b", text)
    ):
        return "CIV"
    if (
        re.search(r"\bCASS(?:AZIONE)?\s+PEN(?:ALE|ALI)?\b", text)
        or re.search(r"\bCASS\s+PEN\b", text)
        or re.search(r"\bSEZ(?:IONE|IONI)?(?:\s+[A-Z0-9]+){0,4}\s+PEN(?:ALE|ALI)?\b", text)
    ):
        return "PEN"
    return None


def _explicit_cass_sector(raw_text: str, value: Any, warnings: list[str]) -> str | None:
    explicit = normalize_cass_sector(raw_text)
    provided = normalize_cass_sector(value)
    if value is not None and provided is None:
        warnings.append(f"invalid_sector={value!r}")
    if provided is not None and explicit is None:
        warnings.append("sector_not_explicit")
        return None
    if provided is not None and explicit is not None and provided != explicit:
        warnings.append(f"sector_conflicts_with_raw_text={value!r}")
    return explicit


SECTION_TEXT_RE = re.compile(
    r"\b(?:SEZ\.?|SEZIONE)\s+(?P<section>[A-Z0-9IVXLCDM]+(?:[-\s][A-Z0-9IVXLCDM]+)?|[A-Z]+(?:\s+[A-Z]+)?)",
    re.IGNORECASE,
)


def _explicit_division(raw_text: str, value: Any, warnings: list[str]) -> str | None:
    match = SECTION_TEXT_RE.search(raw_text)
    explicit = normalize_division(match.group("section")) if match else None
    provided = normalize_division(value)
    if value is not None and provided is None:
        warnings.append(f"invalid_division={value!r}")
    if provided is not None and explicit is None:
        warnings.append("division_not_explicit")
        return None
    if provided is not None and explicit is not None and provided != explicit:
        warnings.append(f"division_conflicts_with_raw_text={value!r}")
    return explicit


def _ascii_text(value: str) -> str:
    import unicodedata

    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = re.sub(r"[^A-Za-z0-9]+", " ", ascii_text)
    return re.sub(r"\s+", " ", ascii_text).strip().upper()


def _find_exact_span(text: str, raw_text: str) -> Span | None:
    start = text.find(raw_text)
    if start < 0:
        return None
    return (start, start + len(raw_text))


COMPOSITE_NUMBER_YEAR_RE = re.compile(
    r"(?:n\.|num\.|numero)\s*(?P<number>\d+)\s*(?:/|\s+del\s+)\s*(?P<year>\d{4})",
    re.IGNORECASE,
)


def _split_composite_citation(citation: Citation) -> list[Citation]:
    matches = list(COMPOSITE_NUMBER_YEAR_RE.finditer(citation.text))
    if len(matches) < 2:
        return [citation]

    split: list[Citation] = []
    for match in matches:
        start = citation.span[0] + match.start()
        end = citation.span[0] + match.end()
        split.append(
            replace(
                citation,
                text=match.group(0),
                span=(start, end),
                spans=((start, end),),
                number=normalize_number(match.group("number")),
                year=normalize_year(match.group("year")),
                mentions=(match.group(0),),
            )
        )
    return split


def deduplicate_citations(citations: Iterable[Citation]) -> list[Citation]:
    by_key: dict[tuple[object, ...], Citation] = {}
    order: list[tuple[object, ...]] = []

    for citation in sorted(citations, key=lambda item: item.span):
        key = citation_key(citation)
        if key not in by_key:
            by_key[key] = citation
            order.append(key)
            continue
        by_key[key] = merge_citations(by_key[key], citation)

    return [by_key[key] for key in order]


def merge_citations(first: Citation, second: Citation) -> Citation:
    method = first.extraction_method
    if second.extraction_method and second.extraction_method != method:
        method = "merged"
    elif method is None:
        method = second.extraction_method

    warnings = tuple(dict.fromkeys((*first.extraction_warnings, *second.extraction_warnings)))
    spans = tuple(dict.fromkeys((*first.spans, *second.spans)))

    return replace(
        first,
        ecli=first.ecli or second.ecli,
        citation_kind=first.citation_kind or second.citation_kind,
        authority=first.authority or second.authority,
        suggested_namespace=first.suggested_namespace or second.suggested_namespace,
        number=first.number or second.number,
        year=first.year if first.year is not None else second.year,
        legal_area=first.legal_area or second.legal_area,
        court=first.court or second.court,
        court_name=first.court_name or second.court_name,
        venue=first.venue or second.venue,
        venue_name=first.venue_name or second.venue_name,
        doc_type=first.doc_type or second.doc_type,
        jurisdiction_type=first.jurisdiction_type or second.jurisdiction_type,
        nrg=first.nrg or second.nrg,
        sector=first.sector or second.sector,
        division=first.division or second.division,
        outside_index_scope=first.outside_index_scope or second.outside_index_scope,
        extraction_method=method,
        extraction_warnings=warnings,
        mentions=tuple(dict.fromkeys((*first.mentions, *second.mentions))),
        spans=spans,
        context_inherited=first.context_inherited or second.context_inherited,
        inherited_fields=tuple(dict.fromkeys((*first.inherited_fields, *second.inherited_fields))),
        inherited_from_span=first.inherited_from_span or second.inherited_from_span,
        metadata_context={**second.metadata_context, **first.metadata_context},
    )


def citation_key(citation: Citation) -> tuple[object, ...]:
    if citation.ecli:
        return ("ecli", citation.ecli.upper())
    if citation.outside_index_scope:
        return (
            "outside",
            citation.jurisdiction_type,
            citation.number,
            citation.year,
            citation.text.lower(),
        )
    return (
        citation.jurisdiction_type or citation.suggested_namespace,
        citation.court,
        citation.venue,
        citation.number,
        citation.year,
    )


def merge_extraction_sources(*citation_groups: Iterable[Citation]) -> list[Citation]:
    citations = [citation for group in citation_groups for citation in group]
    return deduplicate_citations(merge_structured_with_general(citations))


def _clean_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _get_value(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _has_value(obj: Any, key: str) -> bool:
    if isinstance(obj, dict):
        return key in obj
    return hasattr(obj, key)


def _value_length(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        return len(value)
    try:
        return len(json.dumps(_to_builtin(value), ensure_ascii=False))
    except Exception:
        return len(repr(value))


def _coerce_extraction_error(exc: Exception) -> CitationExtractionError:
    if isinstance(exc, CitationExtractionError):
        return exc

    status_code = _exception_status_code(exc)
    request_id = _exception_request_id(exc)
    class_name = exc.__class__.__name__.lower()
    if "timeout" in class_name:
        return CitationExtractionError(
            str(exc),
            category="sdk_timeout",
            retryable=True,
            status_code=status_code,
            request_id=request_id,
            retry_reason="timeout",
        )
    if status_code == 429 or (status_code is not None and status_code >= 500):
        return CitationExtractionError(
            str(exc),
            category="transport_error",
            retryable=True,
            status_code=status_code,
            request_id=request_id,
            retry_reason=f"http_{status_code}",
        )
    if "connection" in class_name or "apierror" in class_name or "api_error" in class_name:
        return CitationExtractionError(
            str(exc),
            category="transport_error",
            retryable=True,
            status_code=status_code,
            request_id=request_id,
            retry_reason="connection_error",
        )
    return CitationExtractionError(
        str(exc),
        category="transport_error",
        retryable=False,
        status_code=status_code,
        request_id=request_id,
    )


def _exception_status_code(exc: Exception) -> int | None:
    value = getattr(exc, "status_code", None)
    if isinstance(value, int):
        return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def _exception_request_id(exc: Exception) -> str | None:
    for name in ("request_id", "_request_id"):
        value = getattr(exc, name, None)
        if value:
            return str(value)
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        for name in ("x-request-id", "request-id"):
            value = headers.get(name)
            if value:
                return str(value)
    return None


def _retry_reason(
    error: CitationExtractionError,
    *,
    attempt: int,
    max_retries: int,
    corrective_retry_used: bool,
) -> str | None:
    if attempt >= max_retries:
        return None
    if error.category in TRANSIENT_ERROR_CATEGORIES and error.retryable:
        return error.retry_reason or error.category
    if error.category in CORRECTABLE_FORMAT_ERROR_CATEGORIES and not corrective_retry_used:
        return f"corrective_{error.category}"
    return None


def _corrective_message(category: str) -> str:
    return (
        "Previous extraction response was invalid "
        f"({category}). Return exactly one tool call named "
        f"{TOOL_NAME} with arguments as a JSON object containing "
        'a "citations" array. Do not return prose.'
    )


def _response_stop_reason(response: Any | None) -> Any:
    if response is None:
        return None
    stop_reason = _get_value(response, "stop_reason")
    if stop_reason is not None:
        return stop_reason
    choices = _get_value(response, "choices") or []
    if choices:
        return _get_value(choices[0], "finish_reason")
    status = _get_value(response, "status")
    if status is not None:
        return status
    return None


def _openai_response_block_types(response: Any | None) -> list[str]:
    if response is None:
        return []
    types: list[str] = []
    for choice in _get_value(response, "choices") or []:
        message = _get_value(choice, "message")
        if message is None:
            continue
        if _get_value(message, "tool_calls"):
            types.append("tool_call")
        if _get_value(message, "content"):
            types.append("message_content")
    for item in _get_value(response, "output") or []:
        item_type = _get_value(item, "type")
        if item_type:
            types.append(str(item_type))
        for block in _get_value(item, "content") or []:
            block_type = _get_value(block, "type")
            if block_type:
                types.append(str(block_type))
    return types


def _type_name(value: Any) -> str:
    if value is None:
        return "NoneType"
    return type(value).__name__


def _raw_response_excerpt(response: Any | None, payload: Any) -> str | None:
    if response is None:
        return None
    parts: list[str] = []
    for choice in _get_value(response, "choices") or []:
        message = _get_value(choice, "message")
        if message is None:
            continue
        for tool_call in _get_value(message, "tool_calls") or []:
            function = _get_value(tool_call, "function") or {}
            parts.append(
                _safe_json_excerpt(
                    {
                        "name": _get_value(function, "name"),
                        "arguments": _get_value(function, "arguments"),
                    }
                )
            )
        content_value = _get_value(message, "content")
        if content_value:
            parts.append(str(content_value))
    content = _get_value(response, "content") or []
    if isinstance(content, list):
        for block in content:
            block_type = _get_value(block, "type")
            if block_type == "tool_use":
                parts.append(_safe_json_excerpt(_get_value(block, "input")))
            elif _get_value(block, "text") is not None:
                parts.append(str(_get_value(block, "text")))
            else:
                parts.append(_safe_json_excerpt(_to_builtin(block)))
    output_text = _get_value(response, "output_text")
    if output_text:
        parts.append(str(output_text))
    for item in _get_value(response, "output") or []:
        if _get_value(item, "type") == "function_call":
            parts.append(
                _safe_json_excerpt(
                    {
                        "name": _get_value(item, "name"),
                        "arguments": _get_value(item, "arguments"),
                    }
                )
            )
        for block in _get_value(item, "content") or []:
            text_value = _get_value(block, "text")
            if text_value:
                parts.append(str(text_value))
    if not parts and payload is not None:
        parts.append(_safe_json_excerpt(payload))
    excerpt = _sanitize_excerpt("\n".join(part for part in parts if part))
    if not excerpt:
        return None
    if len(excerpt) > DIAGNOSTIC_EXCERPT_MAX_CHARS:
        return excerpt[:DIAGNOSTIC_EXCERPT_MAX_CHARS] + "...[truncated]"
    return excerpt


# Legacy import aliases. Runtime calls OpenAI/Luna through Responses API.


def _safe_json_excerpt(value: Any) -> str:
    try:
        return json.dumps(_to_builtin(value), ensure_ascii=False, sort_keys=True)
    except Exception:
        return repr(value)


def _sanitize_excerpt(value: str) -> str:
    sanitized = re.sub(r"sk-[A-Za-z0-9_-]{8,}", "[redacted]", value)
    sanitized = re.sub(r"sk_ant_[A-Za-z0-9_-]{8,}", "[redacted]", sanitized)
    sanitized = re.sub(r"sk-ant-[A-Za-z0-9_-]{8,}", "[redacted]", sanitized)
    sanitized = re.sub(r"AIza[0-9A-Za-z_-]{20,}", "[redacted]", sanitized)
    return sanitized


def _truncate_excerpt(value: str, max_chars: int = DIAGNOSTIC_EXCERPT_MAX_CHARS) -> str:
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "...[truncated]"


def _to_builtin(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return _to_builtin(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return _to_builtin(vars(value))
    return repr(value)
