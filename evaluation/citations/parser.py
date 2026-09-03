from __future__ import annotations

import re
from dataclasses import replace
from typing import Iterable

from .ecli import derive_authority_codes, normalize_cass_sector, normalize_division, normalize_doc_type
from .models import Citation, Span


ECLI_RE = re.compile(
    r"\b(?P<ecli>ECLI:IT:[A-Z0-9]+:\d{4}:[A-Z0-9.]+)\b",
    re.IGNORECASE,
)

NUMBER_YEAR_RE = (
    r"(?:(?P<doc_type_word>sentenza|sent\.?|ordinanza|ord\.?|decreto|dec\.?)\s*)?"
    r"(?:n\.|num\.|numero)\s*(?P<number>\d+)\s*(?:/|\s+del\s+)\s*(?P<year>\d{4})"
)
SECTION_RE = r"(?:,\s*)?(?:Sez\.|Sezione)\s*(?P<section>[A-Za-z0-9IVXLCDM\- ]+?)\s*,?\s*"


CITATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "CASS",
        re.compile(
            rf"""
            (?P<authority>CASSSU|CASSLAV|Cass\.|Cassazione|Corte\s+di\s+cassazione)
            \s*
            (?:(?P<area>civ\.|civile|pen\.|penale)\s*)?
            (?:,\s*)?
            (?:{SECTION_RE})?
            {NUMBER_YEAR_RE}
            """,
            re.IGNORECASE | re.VERBOSE,
        ),
    ),
    (
        "MER",
        re.compile(
            rf"""
            (?P<authority>
                Tribunale\s+di\s+[^,;\n]+?
                |Corte\s+d['’]Appello\s+di\s+[^,;\n]+?
                |Corte\s+di\s+Appello\s+di\s+[^,;\n]+?
            )
            \s*,?\s*
            {NUMBER_YEAR_RE}
            """,
            re.IGNORECASE | re.VERBOSE,
        ),
    ),
    (
        "ADM",
        re.compile(
            rf"""
            (?P<authority>
                Cons\.\s*Stato
                |Consiglio\s+di\s+Stato
                |T\.?\s*A\.?\s*R\.?(?:\s+[^,;\n]+?)?
            )
            \s*,?\s*
            {NUMBER_YEAR_RE}
            """,
            re.IGNORECASE | re.VERBOSE,
        ),
    ),
    (
        "COST",
        re.compile(
            rf"""
            (?P<authority>Corte\s+cost\.|Corte\s+costituzionale)
            \s*,?\s*
            {NUMBER_YEAR_RE}
            """,
            re.IGNORECASE | re.VERBOSE,
        ),
    ),
    (
        "CONT",
        re.compile(
            rf"""
            (?P<authority>Corte\s+dei\s+conti(?:\s+[^,;\n]+?)?)
            \s*,?\s*
            {NUMBER_YEAR_RE}
            """,
            re.IGNORECASE | re.VERBOSE,
        ),
    ),
    (
        "TAX",
        re.compile(
            rf"""
            (?P<authority>
                Corte\s+di\s+giustizia\s+tributaria(?:\s+[^,;\n]+?)?
                |Commissione\s+tributaria(?:\s+[^,;\n]+?)?
            )
            \s*,?\s*
            {NUMBER_YEAR_RE}
            """,
            re.IGNORECASE | re.VERBOSE,
        ),
    ),
)


OUTSIDE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?P<authority>CEDU(?=\W|$)|Corte\s+EDU(?=\W|$)|Corte\s+europea\s+dei\s+diritti\s+dell['’]uomo)"
        r"(?:[^.;\n]{0,80})?",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?P<authority>CGUE|Corte\s+di\s+giustizia\s+(?:UE|dell['’]Unione\s+europea))"
        r"(?:[^.;\n]{0,80})?",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?P<authority>art\.|articolo)\s+(?P<number>\d+[a-z]*)\s+"
        r"(?P<code>c\.c\.|c\.p\.|c\.p\.c\.|c\.p\.p\.|Cost\.|Costituzione)(?=\W|$)",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?P<authority>dottrina)\b", re.IGNORECASE),
)


def normalize_number(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.lstrip("0")
    return stripped or "0"


def infer_namespace_from_ecli(ecli: str) -> str | None:
    parts = ecli.upper().split(":", 4)
    if len(parts) != 5:
        return None

    court_code = parts[2]
    if court_code == "CASS":
        return "CASS"
    if court_code == "COST":
        return "COST"
    if court_code == "CONT":
        return "CONT"
    if court_code == "ABF":
        return "ABF"
    if court_code == "COVIP":
        return "COVIP"
    if court_code == "CDS" or court_code.startswith("TAR"):
        return "ADM"
    if court_code.startswith("CG1") or court_code.startswith("CG2"):
        return "TAX"
    if court_code.startswith("CA") or court_code.startswith("TR"):
        return "MER"
    return None


def parse_ecli_fields(ecli: str) -> tuple[str | None, int | None, str | None]:
    parts = ecli.upper().split(":", 4)
    if len(parts) != 5:
        return None, None, None

    year = int(parts[3]) if parts[3].isdigit() else None
    tail = parts[4]
    match = re.match(r"^(?P<number>\d+)", tail)
    number = normalize_number(match.group("number")) if match else None

    legal_area = None
    area_match = re.fullmatch(r"\d+(CIV|PEN)", tail)
    if area_match:
        legal_area = area_match.group(1)

    return number, year, legal_area


def normalize_legal_area(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.lower().rstrip(".")
    if normalized in {"civ", "civile"}:
        return "CIV"
    if normalized in {"pen", "penale"}:
        return "PEN"
    return None


def citation_key(citation: Citation) -> tuple[object, ...]:
    if citation.ecli:
        return ("ecli", citation.ecli.upper())
    if citation.outside_index_scope:
        return (
            "outside",
            (citation.authority or "").lower(),
            citation.number,
            citation.year,
            citation.text.lower(),
        )
    return (
        citation.suggested_namespace,
        citation.number,
        citation.year,
        citation.legal_area,
        (citation.section or "").upper(),
        (citation.authority or "").lower(),
    )


def merge_duplicates(citations: Iterable[Citation]) -> list[Citation]:
    by_key: dict[tuple[object, ...], Citation] = {}
    order: list[tuple[object, ...]] = []

    for citation in sorted(citations, key=lambda item: item.span):
        key = citation_key(citation)
        if key not in by_key:
            by_key[key] = citation
            order.append(key)
            continue

        existing = by_key[key]
        by_key[key] = replace(existing, spans=(*existing.spans, citation.span))

    return [by_key[key] for key in order]


class CitationParser:
    def parse(self, text: str) -> list[Citation]:
        citations: list[Citation] = []

        citations.extend(self._parse_ecli(text))
        citations.extend(self._parse_court_citations(text))
        citations.extend(self._parse_outside_scope(text))

        return merge_duplicates(citations)

    def _parse_ecli(self, text: str) -> list[Citation]:
        citations: list[Citation] = []
        for match in ECLI_RE.finditer(text):
            ecli = match.group("ecli").upper()
            number, year, legal_area = parse_ecli_fields(ecli)
            citations.append(
                Citation(
                    text=match.group(0),
                    span=match.span(),
                    ecli=ecli,
                    citation_kind="ecli",
                    suggested_namespace=infer_namespace_from_ecli(ecli),
                    number=number,
                    year=year,
                    legal_area=legal_area,
                    jurisdiction_type=infer_namespace_from_ecli(ecli),
                    extraction_method="explicit_regex",
                )
            )
        return citations

    def _parse_court_citations(self, text: str) -> list[Citation]:
        citations: list[Citation] = []
        for namespace, pattern in CITATION_PATTERNS:
            for match in pattern.finditer(text):
                groupdict = match.groupdict()
                authority = groupdict.get("authority")
                court, venue = derive_authority_codes(authority, namespace=namespace)
                section = groupdict.get("section")
                authority_context = " ".join(
                    part for part in (authority, section, groupdict.get("area")) if part
                )
                citations.append(
                    Citation(
                        text=match.group(0).strip(),
                        span=match.span(),
                        citation_kind="case_law",
                        authority=authority,
                        suggested_namespace=namespace,
                        number=normalize_number(groupdict.get("number")),
                        year=int(groupdict["year"]) if groupdict.get("year") else None,
                        section=section,
                        legal_area=normalize_legal_area(groupdict.get("area")),
                        court="CASS" if namespace == "CASS" else court,
                        venue=venue,
                        doc_type=normalize_doc_type(groupdict.get("doc_type_word")),
                        jurisdiction_type=namespace,
                        sector=normalize_cass_sector(authority_context) if namespace == "CASS" else None,
                        division=normalize_division(section),
                        extraction_method="explicit_regex",
                    )
                )
        return citations

    def _parse_outside_scope(self, text: str) -> list[Citation]:
        citations: list[Citation] = []
        for pattern in OUTSIDE_PATTERNS:
            for match in pattern.finditer(text):
                groupdict = match.groupdict()
                number = normalize_number(groupdict.get("number"))
                citations.append(
                    Citation(
                        text=match.group(0).strip(),
                        span=match.span(),
                        citation_kind="outside_index_scope",
                        authority=groupdict.get("authority"),
                        number=number,
                        outside_index_scope=True,
                        extraction_method="explicit_regex",
                    )
                )
        return citations


def parse_citations(text: str) -> list[Citation]:
    return CitationParser().parse(text)
