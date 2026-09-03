from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Iterable

from .ecli import infer_namespace_from_ecli, normalize_code, normalize_number, normalize_year
from .models import Citation, Span
from .parser import parse_ecli_fields


URL_EXTRACTION_METHOD = "deterministic_ecli_url"
_URL_PREFIX_RE = r"https?://[^\s<>()\]]*/"

MARKDOWN_URL_RE = re.compile(
    r"\[(?P<anchor>[^\]]+)\]\("
    rf"(?P<url>{_URL_PREFIX_RE}(?P<ecli>ECLI_[A-Za-z0-9_.-]+))"
    r"\)",
    re.IGNORECASE,
)

RAW_URL_RE = re.compile(
    rf"(?P<url>{_URL_PREFIX_RE}(?P<ecli>ECLI_[A-Za-z0-9_.-]+))",
    re.IGNORECASE,
)

TRAILING_URL_PUNCTUATION = ".,;:)]}"


def extract_structured_url_citations(model_output: str) -> list[Citation]:
    """Extract exact ECLI identities from case-law URLs."""
    if "ECLI_" not in model_output:
        return []

    citations: list[Citation] = []
    consumed_url_spans: list[Span] = []

    for match in MARKDOWN_URL_RE.finditer(model_output):
        url, ecli_raw, url_span = _clean_url_match(match)
        citation = _citation_from_url(
            text=match.group(0),
            span=match.span(),
            url=url,
            url_span=url_span,
            ecli_raw=ecli_raw,
            anchor_text=match.group("anchor"),
        )
        if citation is not None:
            citations.append(citation)
            consumed_url_spans.append(url_span)

    for match in RAW_URL_RE.finditer(model_output):
        url, ecli_raw, url_span = _clean_url_match(match)
        if any(_span_contains(existing, url_span[0]) for existing in consumed_url_spans):
            continue
        citation = _citation_from_url(
            text=url,
            span=url_span,
            url=url,
            url_span=url_span,
            ecli_raw=ecli_raw,
            anchor_text=None,
        )
        if citation is not None:
            citations.append(citation)

    return _dedupe_structured(citations)


def _clean_url_match(match: re.Match[str]) -> tuple[str, str, Span]:
    url = match.group("url")
    ecli_raw = match.group("ecli")
    stripped = 0
    while url and url[-1] in TRAILING_URL_PUNCTUATION:
        url = url[:-1]
        ecli_raw = ecli_raw[:-1]
        stripped += 1
    start, end = match.span("url")
    return url, ecli_raw, (start, end - stripped)


def _citation_from_url(
    *,
    text: str,
    span: Span,
    url: str,
    url_span: Span,
    ecli_raw: str,
    anchor_text: str | None,
) -> Citation | None:
    ecli = normalize_url_ecli(ecli_raw)
    if ecli is None:
        return None

    number, year, legal_area = parse_ecli_fields(ecli)
    namespace = infer_namespace_from_ecli(ecli)
    court_code = normalize_code(ecli.split(":")[2] if len(ecli.split(":")) >= 3 else None)
    return Citation(
        text=text,
        span=span,
        ecli=ecli,
        citation_kind="explicit_ecli",
        authority=anchor_text,
        suggested_namespace=namespace,
        number=number,
        year=year,
        legal_area=legal_area,
        court=court_code if namespace in {"CASS", "COST", "CONT", "ABF", "COVIP"} else None,
        jurisdiction_type=namespace,
        extraction_method=URL_EXTRACTION_METHOD,
        metadata_context={
            "case_law_url": {
                "url": url,
                "span": list(url_span),
                "anchor_text": anchor_text,
                "raw_ecli": ecli_raw,
            },
            "identity": {
                "status": "explicit",
                "source": URL_EXTRACTION_METHOD,
            },
        },
    )


def normalize_url_ecli(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value).strip().upper()
    match = re.search(r"ECLI[_:]IT[_:][A-Z0-9]+[_:]\d{4}[_:][A-Z0-9_.-]+", raw)
    token = match.group(0) if match else raw
    token = token.strip(TRAILING_URL_PUNCTUATION).replace("_", ":")
    parts = token.split(":")
    if len(parts) < 5 or parts[0] != "ECLI" or parts[1] != "IT":
        return None
    year = normalize_year(parts[3])
    if year is None:
        return None
    return ":".join([parts[0], parts[1], parts[2], str(year), "".join(parts[4:])])


def has_supported_case_law_url(text: str) -> bool:
    return bool(extract_structured_url_citations(text))


def structured_eclis_in_text(text: str) -> set[str]:
    return {citation.ecli for citation in extract_structured_url_citations(text) if citation.ecli}


def merge_structured_with_general(citations: Iterable[Citation]) -> list[Citation]:
    """Deduplicate with deterministic URL ECLI identities taking priority."""
    ordered = sorted(citations, key=lambda citation: citation.span)
    by_ecli: dict[str, Citation] = {}
    others: list[Citation] = []

    for citation in ordered:
        if citation.ecli:
            key = citation.ecli.upper()
            existing = by_ecli.get(key)
            by_ecli[key] = _merge_citation(existing, citation) if existing else citation
        else:
            others.append(citation)

    remaining: list[Citation] = []
    for citation in others:
        key = _matching_structured_ecli_key(citation, by_ecli)
        if key is None:
            remaining.append(citation)
            continue
        by_ecli[key] = _merge_citation(by_ecli[key], citation, keep_identity_from_first=True)

    combined = [*by_ecli.values(), *remaining]
    return sorted(combined, key=lambda citation: citation.span)


def _matching_structured_ecli_key(citation: Citation, by_ecli: dict[str, Citation]) -> str | None:
    namespace = (citation.suggested_namespace or citation.jurisdiction_type or citation.court or "").upper()
    if namespace != "CASS":
        return None
    year = normalize_year(citation.year)
    number = normalize_number(citation.number)
    if year is None or number is None:
        return None
    area = (citation.legal_area or "").upper()
    for key, structured in by_ecli.items():
        if structured.extraction_method != URL_EXTRACTION_METHOD:
            continue
        if structured.suggested_namespace != "CASS":
            continue
        if structured.year != year or structured.number != number:
            continue
        if area and structured.legal_area and area != structured.legal_area:
            continue
        return key
    return None


def _dedupe_structured(citations: list[Citation]) -> list[Citation]:
    by_ecli: dict[str, Citation] = {}
    order: list[str] = []
    for citation in citations:
        if not citation.ecli:
            continue
        key = citation.ecli.upper()
        if key not in by_ecli:
            by_ecli[key] = citation
            order.append(key)
            continue
        by_ecli[key] = _merge_citation(by_ecli[key], citation)
    return [by_ecli[key] for key in order]


def _merge_citation(
    first: Citation | None,
    second: Citation,
    *,
    keep_identity_from_first: bool = False,
) -> Citation:
    if first is None:
        return second

    metadata_context = dict(second.metadata_context)
    metadata_context.update(first.metadata_context)
    source_methods = [
        method for method in (first.extraction_method, second.extraction_method) if method
    ]
    if source_methods:
        metadata_context["source_methods"] = {"values": list(dict.fromkeys(source_methods))}
    inherited_fields = tuple(dict.fromkeys((*first.inherited_fields, *second.inherited_fields)))
    extraction_method = first.extraction_method
    if second.extraction_method and second.extraction_method != extraction_method:
        extraction_method = first.extraction_method if keep_identity_from_first else "merged"

    updates = {
        "spans": tuple(dict.fromkeys((*first.spans, *second.spans))),
        "mentions": tuple(dict.fromkeys((*first.mentions, *second.mentions))),
        "extraction_warnings": tuple(
            dict.fromkeys((*first.extraction_warnings, *second.extraction_warnings))
        ),
        "metadata_context": metadata_context,
        "inherited_fields": inherited_fields,
        "extraction_method": extraction_method,
    }
    if not keep_identity_from_first:
        updates.update(
            {
                "ecli": first.ecli or second.ecli,
                "citation_kind": first.citation_kind or second.citation_kind,
                "authority": first.authority or second.authority,
                "suggested_namespace": first.suggested_namespace or second.suggested_namespace,
                "number": first.number or second.number,
                "year": first.year if first.year is not None else second.year,
                "legal_area": first.legal_area or second.legal_area,
                "court": first.court or second.court,
                "court_name": first.court_name or second.court_name,
                "venue": first.venue or second.venue,
                "venue_name": first.venue_name or second.venue_name,
                "doc_type": first.doc_type or second.doc_type,
                "jurisdiction_type": first.jurisdiction_type or second.jurisdiction_type,
                "nrg": first.nrg or second.nrg,
                "sector": first.sector or second.sector,
                "division": first.division or second.division,
                "outside_index_scope": first.outside_index_scope or second.outside_index_scope,
            }
        )
    return replace(first, **updates)


def _span_contains(span: Span, position: int) -> bool:
    return span[0] <= position < span[1]
