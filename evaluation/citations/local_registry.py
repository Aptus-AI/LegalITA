"""Read-only SQLite registry used by the public citation-grounding pipeline.

The registry bundle contains citation identifiers and a small set of identity
metadata.  It is opened in SQLite read-only mode and never makes network calls.
Rulings published after the bundle's ``built_at`` timestamp are necessarily
unknown, so reports always record the registry version that produced them.

Standard library only.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable, Iterator

DEFAULT_REGISTRY_PATH = Path("data") / "citation_pool" / "registry" / "ecli_registry_v1.sqlite"
CASS_YEAR_COVERAGE_MIN_ROWS = 1000


class LocalRegistryIndex:
    """Read-only lookup index backed by the local registry snapshot."""

    def __init__(
        self,
        sqlite_path: str | Path = DEFAULT_REGISTRY_PATH,
        namespaces: Iterable[str] | None = None,
    ) -> None:
        self.path = Path(sqlite_path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._con = sqlite3.connect(
            f"file:{self.path}?mode=ro",
            uri=True,
            check_same_thread=False,
        )
        meta = dict(self._con.execute("SELECT key, value FROM meta").fetchall())
        available = {ns for ns in (meta.get("namespaces") or "").split(",") if ns}
        self._namespaces = set(namespaces) if namespaces is not None else available
        self.built_at: str | None = meta.get("built_at")
        self.index_name = f"local-registry:{self.path.name}@{(self.built_at or '')[:10]}"

    # -- Lookup API ----------------------------------------------------------
    def fetch(
        self,
        *,
        ids: list[str] | tuple[str, ...],
        namespace: str = "",
        timeout: float | None = None,
    ) -> dict[str, Any]:
        ids = [str(identifier) for identifier in ids]
        if namespace not in self._namespaces or not ids:
            return {"vectors": {}}
        # merit-court ids: URL slugs separate number and progressive with '-',
        # the index uses '.'; answer under the id the caller asked for.
        alias: dict[str, str] = {}
        for identifier in list(ids):
            if "-" in identifier.split(":")[-1]:
                head, _, tail = identifier.rpartition(":")
                canonical = f"{head}:{tail.replace('-', '.', 1)}"
                alias[canonical] = identifier
                ids.append(canonical)
        placeholders = ",".join("?" * len(ids))
        rows = self._con.execute(
            "SELECT ecli, court, year, number, legal_area, doc_type, sector, publication_date "
            f"FROM registry WHERE namespace = ? AND ecli IN ({placeholders})",
            [namespace, *ids],
        ).fetchall()
        vectors: dict[str, dict[str, Any]] = {}
        for row in rows:
            key = alias.get(row[0], row[0])
            vectors[key] = {
                "id": key,
                "metadata": self._metadata(*row[1:]),
                "canonical_id": row[0],
            }
        return {"vectors": vectors}

    def list(
        self,
        *,
        prefix: str | None = "",
        limit: int | None = 100,
        namespace: str = "",
        timeout: float | None = None,
    ) -> Iterator[list[str]]:
        if namespace not in self._namespaces:
            return iter(())
        return self._pages(namespace, prefix or "", limit or 100)

    def describe_index_stats(self) -> dict[str, Any]:
        counts = dict(
            self._con.execute(
                "SELECT namespace, COUNT(*) FROM registry GROUP BY namespace"
            ).fetchall()
        )
        return {
            "total_vector_count": sum(counts.values()),
            "namespaces": {ns: {"vector_count": n} for ns, n in counts.items()},
            "built_at": self.built_at,
        }

    def get_registry_info(self) -> dict[str, str | None]:
        return {"built_at": self.built_at, "index_name": self.index_name}

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> "LocalRegistryIndex":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def coverage(self) -> dict[str, Any]:
        """Namespaces present and the CASS years with substantial coverage.

        A ``not_found`` for a namespace absent from the snapshot, or for a CASS
        year the snapshot barely covers, says nothing about the citation itself.
        """
        namespaces = {
            namespace
            for (namespace,) in self._con.execute("SELECT DISTINCT namespace FROM registry")
        }
        courts = {court for (court,) in self._con.execute("SELECT DISTINCT court FROM registry")}
        cass_years = {
            int(year)
            for year, rows in self._con.execute(
                "SELECT year, COUNT(*) FROM registry WHERE namespace = 'CASS' GROUP BY year"
            )
            if rows >= CASS_YEAR_COVERAGE_MIN_ROWS
        }
        return {
            "namespaces": namespaces,
            "courts": courts,
            "cass_years": cass_years,
            "snapshot_year": int(self.built_at[:4]) if self.built_at else None,
            "built_at": self.built_at,
            "index_name": self.index_name,
        }

    # -- helpers -------------------------------------------------------------
    def _pages(self, namespace: str, prefix: str, limit: int) -> Iterator[list[str]]:
        page: list[str] = []
        cursor = self._con.execute(
            "SELECT ecli FROM registry WHERE namespace = ? AND ecli >= ? AND ecli < ? ORDER BY ecli",
            (namespace, prefix, prefix + "\uffff"),
        )
        for (ecli,) in cursor:
            page.append(ecli)
            if len(page) >= limit:
                yield page
                page = []
        if page:
            yield page

    @staticmethod
    def _metadata(
        court: str,
        year: int,
        number: int,
        area: str | None,
        doc_type: str | None = None,
        sector: str | None = None,
        publication_date: str | None = None,
    ) -> dict[str, str]:
        metadata = {
            "jurisdiction": "IT",
            "jurisdictionType": court,
            "court": court,
            "venue": court,
            "year": str(year),
            "number": f"{number:08d}",
        }
        if area:
            metadata["legalArea"] = area
        if doc_type:
            metadata["docType"] = doc_type
        if sector:
            metadata["sector"] = sector
        if publication_date:
            metadata["publicationDate"] = publication_date
        return metadata
