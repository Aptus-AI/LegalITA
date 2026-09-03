from __future__ import annotations

import inspect
import os
import re
from typing import Any

from .llm_extractor import (
    CitationExtractionError,
    OpenAICitationExtractor,
    merge_extraction_sources,
)
from .models import Citation, RegistryCandidate, ResolutionResult
from .parser import CitationParser
from .local_resolver import LocalCitationResolver
from .structured_urls import extract_structured_url_citations


FINAL_CITATION_STATUSES = (
    "resolved_local_registry_exact",
    "resolved_local_registry_incomplete",
    "resolved_local_registry_metadata_mismatch",
    "ambiguous_local_registry",
    "not_found_in_index",
    "suspected_fabricated",
    "confirmed_fabricated",
    "outside_index_scope",
    "resolver_error",
    "citation_extraction_error",
)

OUTSIDE_AUTHORITIES = ("CEDU", "CORTE EDU", "CGUE", "CORTE DI GIUSTIZIA")
FAST_PATH_ENV = "CITATION_EXTRACTOR_FAST_PATH"
LOOSE_CITATION_MARKER_RE = re.compile(
    r"\b(?:Cass\.?|Cassazione|Corte\s+cost\.?|Consiglio\s+di\s+Stato|"
    r"Tribunale|T\.?\s*A\.?\s*R\.?|sentenza|ordinanza|decreto)\b",
    re.IGNORECASE,
)


class CitationExistenceService:
    def __init__(
        self,
        parser: CitationParser | None = None,
        resolver: Any | None = None,
        extractor: Any | None = None,
    ) -> None:
        self._closed = False
        self.parser = parser or CitationParser()
        if resolver is None:
            raise ValueError("CitationExistenceService richiede un resolver locale")
        self.resolver = resolver
        self.extractor = extractor or OpenAICitationExtractor()

    def check_text(self, text: str, *, task_id: str | None = None) -> dict[str, Any]:
        structured_citations = extract_structured_url_citations(text)
        regex_citations = self._explicit_regex_citations(text)
        extraction_error: str | None = None
        extractor_skipped = False

        if self._can_skip_llm_extractor(text, structured_citations, regex_citations):
            llm_citations = []
            extractor_skipped = True
            if hasattr(self.extractor, "last_diagnostics"):
                self.extractor.last_diagnostics = {
                    "task_id": task_id,
                    "extractor_provider": getattr(self.extractor, "provider", "openai"),
                    "extractor_model": getattr(self.extractor, "model", "unknown"),
                    "error_category": "fast_path_skipped",
                    "fast_path_skipped_llm": True,
                    "structured_citation_count": len(structured_citations),
                    "regex_citation_count": len(regex_citations),
                }
            if hasattr(self.extractor, "attempt_diagnostics"):
                self.extractor.attempt_diagnostics = []
        else:
            try:
                extract = self.extractor.extract
                if "task_id" in inspect.signature(extract).parameters:
                    llm_citations = extract(text, task_id=task_id)
                else:
                    llm_citations = extract(text)
            except CitationExtractionError as exc:
                llm_citations = []
                extraction_error = str(exc)
            except Exception as exc:
                llm_citations = []
                extraction_error = str(exc)

        citations = merge_extraction_sources(
            structured_citations,
            regex_citations,
            llm_citations,
        )
        local_registry_results = self.resolver.resolve_all(citations)
        results = [self._finalize_result(result) for result in local_registry_results]
        warnings = _collect_extraction_warnings(citations)

        counts = count_statuses(results)
        if extraction_error:
            counts["citation_extraction_error"] = counts.get("citation_extraction_error", 0) + 1

        return {
            "results": results,
            "counts": counts,
            "summary": summarize_results(results, self),
            "citation_extraction_error": extraction_error,
            "citation_extraction_diagnostics": getattr(
                self.extractor, "last_diagnostics", {}
            ),
            "citation_extraction_attempt_diagnostics": getattr(
                self.extractor, "attempt_diagnostics", []
            ),
            "citation_extractor_skipped": extractor_skipped,
            "structured_citation_count": len(structured_citations),
            "regex_citation_count": len(regex_citations),
            "llm_citation_count": len(llm_citations),
            "extraction_warnings": warnings,
        }

    def get_registry_info(self) -> dict[str, str | None]:
        if hasattr(self.resolver, "get_registry_info"):
            return self.resolver.get_registry_info()
        return {
            "built_at": None,
            "index_name": None,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self.resolver, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "CitationExistenceService":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # Destructors must never obscure the original exception or fail
            # during interpreter shutdown; normal callers use close()/with.
            pass

    def _explicit_regex_citations(self, text: str) -> list[Citation]:
        citations = self.parser._parse_ecli(text)
        citations.extend(
            citation
            for citation in self.parser._parse_outside_scope(text)
            if _is_judicial_outside_scope(citation)
        )
        return citations

    def _can_skip_llm_extractor(
        self,
        text: str,
        structured_citations: list[Citation],
        regex_citations: list[Citation],
    ) -> bool:
        if os.environ.get(FAST_PATH_ENV, "").strip().lower() not in {"1", "true", "yes"}:
            return False
        citations = [*structured_citations, *regex_citations]
        if not citations:
            return False
        if not all(_is_complete_fast_path_citation(citation) for citation in citations):
            return False
        return not _has_uncovered_citation_markers(text, citations)

    def _finalize_result(self, result: ResolutionResult) -> dict[str, Any]:
        serialized = serialize_result(result)
        local_registry_status = local_registry_public_status(result)
        normalized_key = normalized_citation_key(
            result.citation,
            requested_ecli_candidates=result.requested_ecli_candidates,
        )
        final_candidate_count = _final_candidate_count(result)
        final_status = local_registry_status

        if result.status == "not_found" and not result.existence_confirmed:
            final_status = "not_found_in_index"

        if result.status == "insufficient_data":
            final_status = "not_found_in_index"

        citation_accuracy = result.citation_accuracy
        metadata_mismatches = dict(result.metadata_mismatches)

        existence_confirmed = result.existence_confirmed or result.status in {"resolved", "ambiguous"}
        existence_source = (
            result.existence_source
            if result.existence_source != "none" or not existence_confirmed
            else "local_registry"
        )
        identity_status = result.identity_status
        if identity_status == "unverified" and result.status == "resolved":
            identity_status = "exact"
        elif identity_status == "unverified" and result.status == "ambiguous":
            identity_status = "ambiguous"

        serialized.update(
            {
                "status": final_status,
                "existence_status": final_status,
                "final_status": final_status,
                "existence_confirmed": existence_confirmed,
                "existence_source": existence_source,
                "identity_status": identity_status,
                "normalized_key": normalized_key,
                "citation_accuracy": citation_accuracy,
                "metadata_mismatches": metadata_mismatches,
                "local_registry_status": local_registry_status,
                "local_registry_matched_ecli": result.matched_ecli,
                "local_registry_candidate_count": final_candidate_count,
                "local_registry_metadata_mismatches": result.metadata_mismatches,
                "not_found_in_index": result.status == "not_found" and not result.existence_confirmed,
                "citation_hard_fail": final_status == "confirmed_fabricated",
            }
        )
        return serialized


def _is_judicial_outside_scope(citation: Citation) -> bool:
    authority = (citation.authority or "").upper()
    return any(token in authority for token in OUTSIDE_AUTHORITIES)


def normalized_citation_key(
    citation: Citation,
    *,
    requested_ecli_candidates: tuple[str, ...] = (),
) -> str:
    if requested_ecli_candidates:
        return "|".join(sorted(str(item).upper() for item in requested_ecli_candidates))
    if citation.ecli:
        return str(citation.ecli).upper()
    parts = [
        citation.suggested_namespace or citation.jurisdiction_type or "",
        citation.authority or "",
        citation.number or "",
        str(citation.year or ""),
        citation.legal_area or "",
    ]
    key = ":".join(str(part).strip().upper() for part in parts if str(part).strip())
    return key or citation.text.strip().upper()


def _is_complete_fast_path_citation(citation: Citation) -> bool:
    if citation.ecli:
        return True
    if citation.outside_index_scope:
        return bool(citation.authority)
    return bool(
        (citation.suggested_namespace or citation.jurisdiction_type)
        and citation.number
        and citation.year
    )


def _has_uncovered_citation_markers(text: str, citations: list[Citation]) -> bool:
    covered = [False] * len(text)
    for citation in citations:
        for start, end in citation.spans:
            for index in range(max(0, start), min(len(text), end)):
                covered[index] = True
    uncovered_chars = [
        " " if covered[index] else char
        for index, char in enumerate(text)
    ]
    return bool(LOOSE_CITATION_MARKER_RE.search("".join(uncovered_chars)))


def _collect_extraction_warnings(citations: list[Citation]) -> list[str]:
    warnings: list[str] = []
    for citation in citations:
        warnings.extend(citation.extraction_warnings)
    return list(dict.fromkeys(warnings))


def local_registry_public_status(result: ResolutionResult) -> str:
    if result.status == "resolved":
        if result.citation_accuracy == "metadata_mismatch":
            return "resolved_local_registry_metadata_mismatch"
        if result.citation_accuracy == "incomplete":
            return "resolved_local_registry_incomplete"
        return "resolved_local_registry_exact"
    if result.status == "ambiguous":
        return "ambiguous_local_registry"
    if result.status == "not_found":
        return "not_found_in_index"
    if result.status == "outside_index_scope":
        return "outside_index_scope"
    if result.status == "resolver_error":
        return "resolver_error"
    if result.status == "insufficient_data":
        return "not_found_in_index"
    return "resolver_error"


def count_statuses(results: list[dict[str, Any]]) -> dict[str, int]:
    counts = {status: 0 for status in FINAL_CITATION_STATUSES}
    for result in results:
        status = str(result.get("final_status") or result.get("status") or "resolver_error")
        counts[status] = counts.get(status, 0) + 1
    return counts


def summarize_results(results: list[dict[str, Any]], service: CitationExistenceService) -> dict[str, Any]:
    counts = count_statuses(results)
    return {
        "citations_total": len(results),
        "unique_citations": len({result.get("normalized_key") for result in results if result.get("normalized_key")}),
        "resolved_by_local_registry": sum(
            counts.get(status, 0)
            for status in (
                "resolved_local_registry_exact",
                "resolved_local_registry_incomplete",
                "resolved_local_registry_metadata_mismatch",
            )
        ),
        "ambiguous_local_registry": counts.get("ambiguous_local_registry", 0),
        "not_found_in_index": sum(1 for result in results if result.get("not_found_in_index")),
        "suspected_fabricated": counts.get("suspected_fabricated", 0),
        "confirmed_fabricated": counts.get("confirmed_fabricated", 0),
        "resolver_errors": counts.get("resolver_error", 0),
        "responses_with_confirmed_fabrication_hard_fail": (
            1 if counts.get("confirmed_fabricated", 0) else 0
        ),
    }


def serialize_span(span: tuple[int, int]) -> list[int]:
    return [span[0], span[1]]


def serialize_citation(citation: Citation) -> dict[str, Any]:
    return {
        "text": citation.text,
        "span": serialize_span(citation.span),
        "spans": [serialize_span(span) for span in citation.spans],
        "ecli": citation.ecli,
        "citation_kind": citation.citation_kind,
        "authority": citation.authority,
        "suggested_namespace": citation.suggested_namespace,
        "number": citation.number,
        "year": citation.year,
        "section": citation.section,
        "legal_area": citation.legal_area,
        "court": citation.court,
        "court_name": citation.court_name,
        "venue": citation.venue,
        "venue_name": citation.venue_name,
        "doc_type": citation.doc_type,
        "jurisdiction_type": citation.jurisdiction_type,
        "nrg": citation.nrg,
        "sector": citation.sector,
        "division": citation.division,
        "outside_index_scope": citation.outside_index_scope,
        "extraction_method": citation.extraction_method,
        "extraction_warnings": list(citation.extraction_warnings),
        "mentions": list(citation.mentions),
        "context_inherited": citation.context_inherited,
        "inherited_fields": list(citation.inherited_fields),
        "inherited_from_span": (
            serialize_span(citation.inherited_from_span)
            if citation.inherited_from_span
            else None
        ),
        "metadata_context": citation.metadata_context,
    }


def serialize_candidate(candidate: RegistryCandidate) -> dict[str, Any]:
    return {
        "namespace": candidate.namespace,
        "ecli": candidate.ecli,
        "confidence": candidate.confidence,
        "year": candidate.year,
        "citation_number": candidate.citation_number,
        "legal_area": candidate.legal_area,
        "resolution_method": candidate.resolution_method,
        "matched_metadata": candidate.matched_metadata,
        "candidate_count": candidate.candidate_count,
    }


def serialize_result(result: ResolutionResult) -> dict[str, Any]:
    final_candidate_count = _final_candidate_count(result)
    return {
        "status": result.status,
        "confidence": result.confidence,
        "error": result.error,
        "resolution_method": result.resolution_method,
        "matched_ecli": result.matched_ecli,
        "matched_metadata": result.matched_metadata,
        "candidate_count": final_candidate_count,
        "attempted_ecli": result.attempted_ecli,
        "attempted_prefix": result.attempted_prefix,
        "existence_status": result.existence_status or result.status,
        "citation_accuracy": result.citation_accuracy,
        "metadata_mismatches": result.metadata_mismatches,
        "existence_confirmed": result.existence_confirmed,
        "existence_source": result.existence_source,
        "identity_status": result.identity_status,
        "requested_ecli_candidates": list(result.requested_ecli_candidates),
        "matched_requested_ecli": list(result.matched_requested_ecli),
        "homonymous_ecli_candidates": list(result.homonymous_ecli_candidates),
        "homonymous_matches": list(result.homonymous_matches),
        "ecli_source": result.ecli_source,
        "raw_candidate_count": result.raw_candidate_count,
        "compatible_candidate_count": result.compatible_candidate_count,
        "final_candidate_count": final_candidate_count,
        "citation": serialize_citation(result.citation),
        "candidates": [serialize_candidate(candidate) for candidate in result.candidates],
    }


def _final_candidate_count(result: ResolutionResult) -> int:
    if result.final_candidate_count:
        return result.final_candidate_count
    return int(result.candidate_count or 0)
