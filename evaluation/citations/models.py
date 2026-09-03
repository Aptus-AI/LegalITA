from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Span = tuple[int, int]

ResolutionStatus = Literal[
    "resolved",
    "not_found",
    "ambiguous",
    "insufficient_data",
    "outside_index_scope",
    "resolver_error",
]

CitationAccuracy = Literal[
    "exact",
    "metadata_mismatch",
    "incomplete",
    "unknown",
]

ExistenceSource = Literal[
    "local_registry",
    "none",
]

IdentityStatus = Literal[
    "exact",
    "ambiguous",
    "metadata_conflict",
    "unverified",
]

@dataclass(frozen=True)
class Citation:
    text: str
    span: Span
    ecli: str | None = None
    citation_kind: str | None = None
    authority: str | None = None
    suggested_namespace: str | None = None
    number: str | None = None
    year: int | None = None
    section: str | None = None
    legal_area: str | None = None
    court: str | None = None
    court_name: str | None = None
    venue: str | None = None
    venue_name: str | None = None
    doc_type: str | None = None
    jurisdiction_type: str | None = None
    nrg: str | None = None
    sector: str | None = None
    division: str | None = None
    outside_index_scope: bool = False
    extraction_method: str | None = None
    extraction_warnings: tuple[str, ...] = ()
    mentions: tuple[str, ...] = ()
    spans: tuple[Span, ...] = field(default_factory=tuple)
    context_inherited: bool = False
    inherited_fields: tuple[str, ...] = ()
    inherited_from_span: Span | None = None
    metadata_context: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.spans:
            object.__setattr__(self, "spans", (self.span,))
        if not self.mentions:
            object.__setattr__(self, "mentions", (self.text,))


@dataclass(frozen=True)
class RegistryCandidate:
    namespace: str
    ecli: str
    confidence: str
    year: int | None = None
    citation_number: str | None = None
    legal_area: str | None = None
    resolution_method: str | None = None
    matched_metadata: dict[str, Any] | None = None
    candidate_count: int | None = None


@dataclass(frozen=True)
class ResolutionResult:
    citation: Citation
    status: ResolutionStatus
    candidates: tuple[RegistryCandidate, ...] = ()
    confidence: str | None = None
    error: str | None = None
    resolution_method: str | None = None
    matched_ecli: str | None = None
    matched_metadata: dict[str, Any] | None = None
    candidate_count: int | None = None
    attempted_ecli: str | None = None
    attempted_prefix: str | None = None
    existence_status: ResolutionStatus | None = None
    citation_accuracy: CitationAccuracy = "unknown"
    metadata_mismatches: dict[str, dict[str, Any]] = field(default_factory=dict)
    existence_confirmed: bool = False
    existence_source: ExistenceSource = "none"
    identity_status: IdentityStatus = "unverified"
    requested_ecli_candidates: tuple[str, ...] = ()
    matched_requested_ecli: tuple[str, ...] = ()
    homonymous_ecli_candidates: tuple[str, ...] = ()
    homonymous_matches: tuple[str, ...] = ()
    ecli_source: str | None = None
    raw_candidate_count: int = 0
    compatible_candidate_count: int = 0
    final_candidate_count: int = 0
