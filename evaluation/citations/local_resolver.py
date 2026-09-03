from __future__ import annotations

from typing import Any, Iterable, Protocol

from .ecli import (
    build_ecli_prefix,
    build_exact_ecli,
    cass_ecli_base,
    cass_ecli_candidates,
    derive_authority_codes,
    namespace_for_citation,
    normalize_cass_sector,
    normalize_code,
    normalize_division,
    normalize_doc_type,
    normalize_legal_area,
    normalize_metadata_number,
    normalize_number,
    normalize_year,
)
from .models import Citation, RegistryCandidate, ResolutionResult


ESSENTIAL_METADATA_FIELDS = (
    "jurisdictionType",
    "court",
    "year",
    "number",
    "venue",
    "docType",
    "legalArea",
    "nrg",
    "sector",
    "division",
    "section",
)

DEFAULT_MAX_PREFIX_CANDIDATES = 25


class RegistryIndexLike(Protocol):
    def fetch(
        self,
        *,
        ids: list[str] | tuple[str, ...],
        namespace: str = "",
        timeout: float | None = None,
    ) -> Any:
        ...

    def list(
        self,
        *,
        prefix: str | None = None,
        limit: int | None = None,
        namespace: str = "",
        timeout: float | None = None,
    ) -> Iterable[Any]:
        ...


class LocalCitationResolver:
    def __init__(
        self,
        *,
        index: RegistryIndexLike,
        index_name: str | None = None,
        max_prefix_candidates: int = DEFAULT_MAX_PREFIX_CANDIDATES,
    ) -> None:
        self.index_name = index_name or getattr(index, "index_name", None)
        self.max_prefix_candidates = max_prefix_candidates
        self._index = index

        if self.max_prefix_candidates <= 0:
            raise ValueError("max_prefix_candidates must be positive")

    def get_registry_info(self) -> dict[str, str | None]:
        built_at = getattr(self._index, "built_at", None)
        return {
            "built_at": built_at,
            "index_name": self.index_name,
        }

    def close(self) -> None:
        close = getattr(self._index, "close", None)
        if callable(close):
            close()

    def resolve_all(self, citations: list[Citation]) -> list[ResolutionResult]:
        return [self.resolve(citation) for citation in citations]

    def resolve(self, citation: Citation) -> ResolutionResult:
        if citation.outside_index_scope:
            return ResolutionResult(
                citation=citation,
                status="outside_index_scope",
                existence_status="outside_index_scope",
                citation_accuracy="unknown",
            )

        try:
            return self._resolve(citation)
        except Exception as exc:
            return ResolutionResult(
                citation=citation,
                status="resolver_error",
                existence_status="resolver_error",
                citation_accuracy="unknown",
                error=str(exc),
            )

    def _resolve(self, citation: Citation) -> ResolutionResult:
        namespace = namespace_for_citation(citation)
        if namespace is None:
            return ResolutionResult(
                citation=citation,
                status="insufficient_data",
                existence_status="insufficient_data",
                citation_accuracy="incomplete",
            )

        if namespace == "COVIP" and not citation.ecli:
            return ResolutionResult(
                citation=citation,
                status="insufficient_data",
                existence_status="insufficient_data",
                citation_accuracy="incomplete",
                identity_status="unverified",
            )

        if citation.ecli:
            return self._resolve_exact(citation, namespace, citation.ecli.upper(), "exact_ecli")

        components = self._components(citation, namespace)
        if components["year"] is None or components["number"] is None:
            return ResolutionResult(
                citation=citation,
                status="insufficient_data",
                existence_status="insufficient_data",
                citation_accuracy="incomplete",
                identity_status="unverified",
            )

        if namespace == "CASS":
            return self._resolve_cass_number_year(citation, components)

        exact_ecli = build_exact_ecli(
            namespace,
            components["year"],
            components["number"],
            legal_area=components["legalArea"],
            court=components["court"],
            venue=components["venue"],
            doc_type=components["docType"],
            nrg=components["nrg"],
        )
        if exact_ecli:
            return self._resolve_exact(citation, namespace, exact_ecli, "constructed_ecli")

        prefix = build_ecli_prefix(
            namespace,
            components["year"],
            components["number"],
            legal_area=components["legalArea"],
            court=components["court"],
            venue=components["venue"],
            doc_type=components["docType"],
            nrg=components["nrg"],
        )
        if prefix is None:
            return ResolutionResult(
                citation=citation,
                status="insufficient_data",
                existence_status="insufficient_data",
                citation_accuracy="incomplete",
                identity_status="unverified",
            )

        return self._resolve_prefix(citation, namespace, prefix, components)

    def _resolve_cass_number_year(
        self,
        citation: Citation,
        components: dict[str, Any],
    ) -> ResolutionResult:
        number = components.get("ecli_number") or normalize_number(components["number"])
        base = cass_ecli_base(components["year"], number)
        requested, homonymous = cass_ecli_candidates(
            components["year"],
            number,
            legal_area=components["legalArea"],
        )
        ids = [*requested, *homonymous]
        records = self._fetch_records("CASS", ids)
        raw_candidate_count = len(records)
        homonymous_matches = tuple(ecli for ecli in homonymous if ecli in records)

        candidates: list[tuple[RegistryCandidate, dict[str, Any], dict[str, dict[str, Any]], set[str]]] = []
        for ecli in requested:
            record = records.get(ecli)
            if record is None:
                continue
            metadata = self._metadata(record)
            mismatches, incomplete = self._metadata_diagnostics(metadata, components)
            candidates.append(
                (
                    self._candidate(
                        namespace="CASS",
                        ecli=ecli,
                        method="cass_exact_ecli_candidates",
                        metadata=metadata,
                        candidate_count=len(requested),
                    ),
                    metadata,
                    mismatches,
                    incomplete,
                )
            )

        if not candidates:
            homonymous_candidates = tuple(
                self._candidate(
                    namespace="CASS",
                    ecli=ecli,
                    method="cass_exact_ecli_candidates_homonymous",
                    metadata=self._metadata(records[ecli]),
                    candidate_count=len(homonymous_matches),
                )
                for ecli in homonymous_matches
            )
            metadata_mismatches: dict[str, dict[str, Any]] = {}
            for ecli in homonymous_matches:
                mismatches, _ = self._metadata_diagnostics(self._metadata(records[ecli]), components)
                metadata_mismatches.update(mismatches)
            return ResolutionResult(
                citation=citation,
                status="not_found",
                existence_status="not_found",
                citation_accuracy="metadata_mismatch" if homonymous_matches else "unknown",
                metadata_mismatches=metadata_mismatches,
                candidates=homonymous_candidates,
                resolution_method="cass_exact_ecli_candidates",
                candidate_count=0,
                attempted_prefix=base,
                existence_confirmed=False,
                existence_source="none",
                identity_status="metadata_conflict" if homonymous_matches else "unverified",
                requested_ecli_candidates=tuple(requested),
                matched_requested_ecli=(),
                homonymous_ecli_candidates=tuple(homonymous),
                homonymous_matches=homonymous_matches,
                ecli_source="synthetic",
                raw_candidate_count=raw_candidate_count,
                compatible_candidate_count=0,
                final_candidate_count=0,
            )

        exact = [item for item in candidates if not item[2] and not item[3]]
        coherent = [item for item in candidates if not item[2]]
        if len(exact) == 1:
            return self._resolved_from_diagnostic(
                citation,
                exact[0],
                base,
                "exact",
                requested=requested,
                homonymous=homonymous,
                homonymous_matches=homonymous_matches,
                raw_candidate_count=raw_candidate_count,
                compatible_candidate_count=len(candidates),
            )
        if len(coherent) == 1:
            return self._resolved_from_diagnostic(
                citation,
                coherent[0],
                base,
                "incomplete",
                requested=requested,
                homonymous=homonymous,
                homonymous_matches=homonymous_matches,
                raw_candidate_count=raw_candidate_count,
                compatible_candidate_count=len(candidates),
            )
        if len(candidates) == 1:
            candidate, _, mismatches, incomplete = candidates[0]
            accuracy = "metadata_mismatch" if mismatches else "incomplete"
            return ResolutionResult(
                citation=citation,
                status="resolved",
                existence_status="resolved",
                citation_accuracy=accuracy,
                metadata_mismatches=mismatches,
                candidates=(candidate,),
                confidence="high",
                resolution_method="cass_exact_ecli_candidates",
                matched_ecli=candidate.ecli,
                matched_metadata=candidate.matched_metadata,
                candidate_count=1,
                attempted_prefix=base,
                existence_confirmed=True,
                existence_source="local_registry",
                identity_status="exact",
                requested_ecli_candidates=tuple(requested),
                matched_requested_ecli=(candidate.ecli,),
                homonymous_ecli_candidates=tuple(homonymous),
                homonymous_matches=homonymous_matches,
                ecli_source="synthetic",
                raw_candidate_count=raw_candidate_count,
                compatible_candidate_count=1,
                final_candidate_count=1,
            )

        candidate_tuple = tuple(item[0] for item in candidates)
        all_mismatches: dict[str, dict[str, Any]] = {}
        for _, _, mismatches, _ in candidates:
            all_mismatches.update(mismatches)
        return ResolutionResult(
            citation=citation,
            status="ambiguous",
            existence_status="ambiguous",
            citation_accuracy="metadata_mismatch" if all_mismatches else "incomplete",
            metadata_mismatches=all_mismatches,
            candidates=candidate_tuple,
            resolution_method="cass_exact_ecli_candidates",
            candidate_count=len(candidate_tuple),
            attempted_prefix=base,
            existence_confirmed=True,
            existence_source="local_registry",
            identity_status="ambiguous",
            requested_ecli_candidates=tuple(requested),
            matched_requested_ecli=tuple(candidate.ecli for candidate in candidate_tuple),
            homonymous_ecli_candidates=tuple(homonymous),
            homonymous_matches=homonymous_matches,
            ecli_source="synthetic",
            raw_candidate_count=raw_candidate_count,
            compatible_candidate_count=len(candidate_tuple),
            final_candidate_count=len(candidate_tuple),
        )

    def _resolved_from_diagnostic(
        self,
        citation: Citation,
        diagnostic: tuple[RegistryCandidate, dict[str, Any], dict[str, dict[str, Any]], set[str]],
        prefix: str,
        accuracy: str,
        *,
        requested: list[str],
        homonymous: list[str],
        homonymous_matches: tuple[str, ...],
        raw_candidate_count: int,
        compatible_candidate_count: int,
    ) -> ResolutionResult:
        candidate, _, mismatches, _ = diagnostic
        return ResolutionResult(
            citation=citation,
            status="resolved",
            existence_status="resolved",
            citation_accuracy=accuracy,  # type: ignore[arg-type]
            metadata_mismatches=mismatches,
            candidates=(candidate,),
            confidence="high",
            resolution_method="cass_exact_ecli_candidates",
            matched_ecli=candidate.ecli,
            matched_metadata=candidate.matched_metadata,
            candidate_count=1,
            attempted_prefix=prefix,
            existence_confirmed=True,
            existence_source="local_registry",
            identity_status="exact",
            requested_ecli_candidates=tuple(requested),
            matched_requested_ecli=(candidate.ecli,),
            homonymous_ecli_candidates=tuple(homonymous),
            homonymous_matches=homonymous_matches,
            ecli_source="synthetic",
            raw_candidate_count=raw_candidate_count,
            compatible_candidate_count=compatible_candidate_count,
            final_candidate_count=1,
        )

    def _resolve_exact(
        self,
        citation: Citation,
        namespace: str,
        ecli: str,
        method: str,
    ) -> ResolutionResult:
        records = self._fetch_records(namespace, [ecli])
        record = records.get(ecli)
        if record is None:
            ecli_source = "explicit" if method == "exact_ecli" else "synthetic"
            return ResolutionResult(
                citation=citation,
                status="not_found",
                existence_status="not_found",
                citation_accuracy="unknown",
                resolution_method=method,
                candidate_count=0,
                attempted_ecli=ecli,
                existence_confirmed=False,
                existence_source="none",
                identity_status="exact" if ecli_source == "explicit" else "unverified",
                requested_ecli_candidates=(ecli,),
                ecli_source=ecli_source,
                raw_candidate_count=0,
                compatible_candidate_count=0,
                final_candidate_count=0,
            )

        canonical_ecli = str(self._get_value(record, "canonical_id") or ecli)
        metadata = self._metadata(record)
        components = self._components(citation, namespace)
        mismatches, incomplete = self._metadata_diagnostics(metadata, components)
        accuracy = "metadata_mismatch" if mismatches else ("incomplete" if incomplete else "exact")
        candidate = self._candidate(
            namespace=namespace,
            ecli=canonical_ecli,
            method=method,
            metadata=metadata,
            candidate_count=1,
        )
        return ResolutionResult(
            citation=citation,
            status="resolved",
            existence_status="resolved",
            citation_accuracy=accuracy,
            metadata_mismatches=mismatches,
            candidates=(candidate,),
            confidence="high",
            resolution_method=method,
            matched_ecli=canonical_ecli,
            matched_metadata=self._essential_metadata(metadata),
            candidate_count=1,
            attempted_ecli=ecli,
            existence_confirmed=True,
            existence_source="local_registry",
            identity_status="exact",
            requested_ecli_candidates=(ecli,),
            matched_requested_ecli=(canonical_ecli,),
            ecli_source="explicit" if method == "exact_ecli" else "synthetic",
            raw_candidate_count=1,
            compatible_candidate_count=1,
            final_candidate_count=1,
        )

    def _resolve_prefix(
        self,
        citation: Citation,
        namespace: str,
        prefix: str,
        components: dict[str, Any],
    ) -> ResolutionResult:
        ids = self._list_ids(namespace, prefix)
        if not ids:
            return ResolutionResult(
                citation=citation,
                status="not_found",
                existence_status="not_found",
                citation_accuracy="unknown",
                resolution_method="prefix_metadata",
                candidate_count=0,
                attempted_prefix=prefix,
                existence_confirmed=False,
                existence_source="none",
                identity_status="unverified",
                raw_candidate_count=0,
                compatible_candidate_count=0,
                final_candidate_count=0,
            )

        records = self._fetch_records(namespace, ids)
        candidates: list[RegistryCandidate] = []
        for ecli, record in records.items():
            metadata = self._metadata(record)
            if not self._metadata_matches(metadata, components):
                continue
            candidates.append(
                self._candidate(
                    namespace=namespace,
                    ecli=ecli,
                    method="prefix_metadata",
                    metadata=metadata,
                    candidate_count=len(ids),
                )
            )

        if not candidates:
            return ResolutionResult(
                citation=citation,
                status="not_found",
                existence_status="not_found",
                citation_accuracy="unknown",
                resolution_method="prefix_metadata",
                candidate_count=0,
                attempted_prefix=prefix,
                existence_confirmed=False,
                existence_source="none",
                identity_status="unverified",
                raw_candidate_count=len(records),
                compatible_candidate_count=0,
                final_candidate_count=0,
            )

        candidate_tuple = tuple(candidates)
        if len(candidate_tuple) > 1:
            return ResolutionResult(
                citation=citation,
                status="ambiguous",
                existence_status="ambiguous",
                citation_accuracy="unknown",
                candidates=candidate_tuple,
                resolution_method="prefix_metadata",
                candidate_count=len(candidate_tuple),
                attempted_prefix=prefix,
                existence_confirmed=True,
                existence_source="local_registry",
                identity_status="ambiguous",
                raw_candidate_count=len(records),
                compatible_candidate_count=len(candidate_tuple),
                final_candidate_count=len(candidate_tuple),
            )

        candidate = candidate_tuple[0]
        return ResolutionResult(
            citation=citation,
            status="resolved",
            existence_status="resolved",
            citation_accuracy="exact",
            candidates=candidate_tuple,
            confidence="high",
            resolution_method="prefix_metadata",
            matched_ecli=candidate.ecli,
            matched_metadata=candidate.matched_metadata,
            candidate_count=1,
            attempted_prefix=prefix,
            existence_confirmed=True,
            existence_source="local_registry",
            identity_status="exact",
            raw_candidate_count=len(records),
            compatible_candidate_count=1,
            final_candidate_count=1,
        )

    def _fetch_records(self, namespace: str, ids: list[str]) -> dict[str, Any]:
        if not ids:
            return {}
        response = self._index.fetch(ids=ids, namespace=namespace)
        vectors = self._get_value(response, "vectors") or {}
        if isinstance(vectors, dict):
            return {str(key): value for key, value in vectors.items()}
        return {}

    def _list_ids(self, namespace: str, prefix: str) -> list[str]:
        ids: list[str] = []
        pages = self._index.list(
            namespace=namespace,
            prefix=prefix,
            limit=self.max_prefix_candidates,
        )
        for page in pages:
            for item in self._page_items(page):
                item_id = self._extract_id(item)
                if not item_id:
                    continue
                ids.append(item_id)
                if len(ids) >= self.max_prefix_candidates:
                    return ids
        return ids

    def _components(self, citation: Citation, namespace: str) -> dict[str, Any]:
        authority_text = citation.authority or citation.court_name or citation.venue_name
        derived_court, derived_venue = derive_authority_codes(
            authority_text,
            namespace=namespace,
        )
        court = normalize_code(citation.court) or derived_court
        venue = normalize_code(citation.venue) or derived_venue
        legal_area = normalize_legal_area(citation.legal_area)
        doc_type = normalize_doc_type(citation.doc_type)
        jurisdiction_type = normalize_code(citation.jurisdiction_type) or namespace
        sector = normalize_cass_sector(citation.sector)
        division = normalize_division(citation.division or citation.section)

        if namespace in {"CASS", "COST", "CONT", "ABF", "COVIP"} and court is None:
            court = namespace

        return {
            "jurisdictionType": jurisdiction_type,
            "court": court,
            "year": normalize_year(citation.year),
            "number": normalize_metadata_number(citation.number),
            "ecli_number": normalize_number(citation.number),
            "venue": venue,
            "docType": doc_type,
            "legalArea": legal_area,
            "nrg": citation.nrg,
            "sector": sector,
            "division": division,
        }

    def _metadata_identity_matches(self, metadata: dict[str, Any], components: dict[str, Any]) -> bool:
        return (
            normalize_year(metadata.get("year")) == components.get("year")
            and normalize_metadata_number(metadata.get("number")) == components.get("number")
        )

    def _metadata_diagnostics(
        self,
        metadata: dict[str, Any],
        components: dict[str, Any],
    ) -> tuple[dict[str, dict[str, Any]], set[str]]:
        mismatches: dict[str, dict[str, Any]] = {}
        incomplete: set[str] = set()

        checks = {
            "legalArea": components.get("legalArea"),
            "docType": components.get("docType"),
            "sector": components.get("sector"),
            "division": components.get("division"),
        }
        for key, expected in checks.items():
            if expected is None:
                continue
            observed = self._normalized_metadata_value(metadata, key)
            if observed is None:
                if key == "division":
                    observed_sector = normalize_cass_sector(metadata.get("sector"))
                    if observed_sector == "SU" and expected != "SU":
                        mismatches[key] = {"expected": expected, "observed": observed_sector}
                        continue
                if key == "sector" and expected == "L":
                    observed_area = normalize_legal_area(metadata.get("legalArea"))
                    if observed_area == "PEN":
                        mismatches[key] = {"expected": expected, "observed": observed_area}
                        continue
                incomplete.add(key)
                continue
            if observed != expected:
                mismatches[key] = {"expected": expected, "observed": observed}

        return mismatches, incomplete

    def _normalized_metadata_value(self, metadata: dict[str, Any], key: str) -> Any:
        if key == "legalArea":
            return normalize_legal_area(metadata.get("legalArea"))
        if key == "docType":
            return normalize_doc_type(metadata.get("docType"))
        if key == "sector":
            return normalize_cass_sector(metadata.get("sector"))
        if key == "division":
            return normalize_division(metadata.get("division") or metadata.get("section"))
        return metadata.get(key)

    def _metadata_matches(self, metadata: dict[str, Any], components: dict[str, Any]) -> bool:
        checks = {
            "jurisdictionType": components.get("jurisdictionType"),
            "court": components.get("court"),
            "year": components.get("year"),
            "number": components.get("number"),
            "venue": components.get("venue"),
            "docType": components.get("docType"),
            "legalArea": components.get("legalArea"),
        }
        for key, expected in checks.items():
            if expected is None:
                continue
            if key not in metadata:
                return False
            observed = metadata[key]
            if key == "year":
                observed = normalize_year(observed)
            elif key == "number":
                observed = normalize_metadata_number(observed)
            elif key == "docType":
                observed = normalize_doc_type(observed)
            elif key == "legalArea":
                observed = normalize_legal_area(observed)
            else:
                observed = normalize_code(observed)
            if observed != expected:
                return False
        return True

    def _candidate(
        self,
        namespace: str,
        ecli: str,
        method: str,
        metadata: dict[str, Any],
        candidate_count: int,
    ) -> RegistryCandidate:
        return RegistryCandidate(
            namespace=namespace,
            ecli=ecli,
            confidence="high",
            year=normalize_year(metadata.get("year")),
            citation_number=normalize_number(metadata.get("number")),
            legal_area=normalize_legal_area(metadata.get("legalArea")),
            resolution_method=method,
            matched_metadata=self._essential_metadata(metadata),
            candidate_count=candidate_count,
        )

    @staticmethod
    def _essential_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            key: metadata[key]
            for key in ESSENTIAL_METADATA_FIELDS
            if key in metadata
        }

    @staticmethod
    def _metadata(record: Any) -> dict[str, Any]:
        metadata = LocalCitationResolver._get_value(record, "metadata")
        if metadata is None:
            metadata = LocalCitationResolver._get_value(record, "fields")
        return metadata if isinstance(metadata, dict) else {}

    @staticmethod
    def _get_value(obj: Any, key: str) -> Any:
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    @staticmethod
    def _page_items(page: Any) -> Iterable[Any]:
        vectors = LocalCitationResolver._get_value(page, "vectors")
        if vectors is not None:
            return vectors
        ids = LocalCitationResolver._get_value(page, "ids")
        if ids is not None:
            return ids
        if isinstance(page, (list, tuple)):
            return page
        return ()

    @staticmethod
    def _extract_id(item: Any) -> str | None:
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            value = item.get("id") or item.get("_id")
            return str(value) if value else None
        value = getattr(item, "id", None) or getattr(item, "_id", None)
        return str(value) if value else None
