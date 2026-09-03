"""Public, local-only citation grounding for LegalITA answers."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from dotenv import load_dotenv

from evaluation.citations.ecli import normalize_ecli
from evaluation.citations.llm_extractor import CITATION_EXTRACTOR_MODEL, OpenAICitationExtractor
from evaluation.citations.local_registry import DEFAULT_REGISTRY_PATH, LocalRegistryIndex
from evaluation.citations.local_resolver import LocalCitationResolver
from evaluation.citations.service import CitationExistenceService


BENCHMARK_TASK_COUNT = 67
DEFAULT_PROFILES_DIR = Path("data/citation_pool/question_profiles")
DEFAULT_RESULTS_ROOT = Path("results/grounding-offline")
MERIT_COURT_RE = re.compile(
    r"Trib|Corte d.?[Aa]ppello|App\.|TAR|Consiglio di Stato|Giudice di [Pp]ace|"
    r"Corte dei [Cc]onti|CGT|Comm"
)


@dataclass(frozen=True)
class ResponseRecord:
    task_id: str
    response: str
    model: str | None = None


@dataclass(frozen=True)
class QuestionProfile:
    task_id: str
    path: Path
    question: str
    resolving: frozenset[str]
    retrieved: frozenset[str]
    tiers: Mapping[str, tuple[str, ...]]


class QuestionProfiles:
    """Load the public per-question citation profiles from a bundle."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        self._cache: dict[str, QuestionProfile] = {}

    def load(self, task_id: str) -> QuestionProfile:
        if task_id in self._cache:
            return self._cache[task_id]
        path = self.root / f"{task_id.replace('/', '_')}.json"
        if not path.exists():
            raise FileNotFoundError(f"Question profile mancante per {task_id}: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        resolving: set[str] = set()
        retrieved: set[str] = set()
        tiers: dict[str, tuple[str, ...]] = {}
        for ruling in payload.get("resolving_rulings") or []:
            ecli = normalize_ecli(ruling.get("ecli"))
            if not ecli:
                continue
            resolving.add(ecli)
            retrieved.add(ecli)
            tiers[ecli] = tuple(ruling.get("evidence_tiers") or ())
        for key in ("retrieved_marginal", "retrieved_irrelevant", "retrieved_unassessed"):
            for ruling in payload.get(key) or []:
                ecli = normalize_ecli(ruling.get("ecli"))
                if ecli:
                    retrieved.add(ecli)
        profile = QuestionProfile(
            task_id=task_id,
            path=path,
            question=str(payload.get("question") or "").strip(),
            resolving=frozenset(resolving),
            retrieved=frozenset(retrieved),
            tiers=tiers,
        )
        self._cache[task_id] = profile
        return profile

    def all(self) -> list[QuestionProfile]:
        profiles: list[QuestionProfile] = []
        for path in sorted(self.root.glob("*.json")):
            if path.name == "index.json":
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            task_id = str(payload.get("task_id") or "").strip()
            if task_id:
                profiles.append(self.load(task_id))
        return profiles

    def describe(self) -> dict[str, Any]:
        path = self.root / "index.json"
        index = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        return {
            "kind": "question_profiles",
            "root": str(self.root),
            "version": index.get("schema_version"),
            "generated_at": index.get("generated_at"),
        }


def run_grounding(
    records: Iterable[ResponseRecord],
    *,
    registry_path: Path,
    profiles_dir: Path,
    out_dir: Path,
    n_tasks: int = BENCHMARK_TASK_COUNT,
    extractor_model: str = CITATION_EXTRACTOR_MODEL,
    extractor_timeout_seconds: float = 120.0,
    extractor_max_retries: int = 2,
) -> dict[str, Any]:
    profiles = QuestionProfiles(profiles_dir)
    task_reports: list[dict[str, Any]] = []
    with LocalRegistryIndex(registry_path) as index:
        coverage = index.coverage()
        resolver = LocalCitationResolver(index=index)
        extractor = OpenAICitationExtractor(
            model=extractor_model,
            timeout_seconds=extractor_timeout_seconds,
            max_retries=extractor_max_retries,
        )
        service = CitationExistenceService(resolver=resolver, extractor=extractor)
        for record in records:
            profile = profiles.load(record.task_id)
            checked = service.check_text(record.response, task_id=record.task_id)
            citations = [
                _ground_citation(item, profile=profile, coverage=coverage)
                for item in checked.get("results") or []
            ]
            task_reports.append(
                {
                    "task_id": record.task_id,
                    "model": record.model,
                    "profile_path": str(profile.path),
                    "citation_extraction_error": checked.get("citation_extraction_error"),
                    "citation_extractor_skipped": checked.get("citation_extractor_skipped"),
                    "citation_summary": _citation_summary(citations),
                    "citations": citations,
                }
            )
        registry_info = index.get_registry_info()

    summary = summarize_grounding(task_reports, n_tasks=n_tasks)
    summary["gog_backend"] = "local"
    summary["registry_built_at"] = registry_info.get("built_at")
    payload = {
        "schema_version": "legalita-citation-grounding-local-1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backend": "local",
        "registry_info": registry_info,
        "profile_source": profiles.describe(),
        "task_count": len(task_reports),
        "summary": summary,
        "tasks": task_reports,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(out_dir / "citation_grounding_v3.json", json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    _atomic_write(out_dir / "citation_grounding_v3.md", render_markdown(payload))
    return payload


def _ground_citation(
    result: dict[str, Any],
    *,
    profile: QuestionProfile,
    coverage: Mapping[str, Any],
) -> dict[str, Any]:
    item = dict(result)
    status = str(item.get("final_status") or item.get("status") or "resolver_error")
    matched = normalize_ecli(item.get("matched_ecli"))
    relation = "unresolved"
    if status in {"resolved_local_registry_exact", "resolved_local_registry_incomplete"} and matched:
        if matched in profile.resolving:
            relation = "issue_aligned"
        elif matched in profile.retrieved:
            relation = "retrieved_only"
        else:
            relation = "outside_profile"
    elif status == "not_found_in_index":
        note, unverifiable = _coverage_note(item, coverage)
        if note:
            item["offline_coverage_note"] = note
        relation = "unresolved" if unverifiable else "fabricated_or_not_found"
        if unverifiable:
            item["final_status"] = "not_verifiable_offline"

    item["gold_v3_relation"] = relation
    item["issue_profile_match"] = [matched] if relation == "issue_aligned" and matched else []
    item["offline_tiers"] = list(profile.tiers.get(matched, ())) if matched else []
    return item


def _coverage_note(result: Mapping[str, Any], coverage: Mapping[str, Any]) -> tuple[str | None, bool]:
    candidates = [
        ecli
        for value in result.get("requested_ecli_candidates") or ()
        if (ecli := normalize_ecli(value))
    ]
    citation = result.get("citation") or {}
    explicit = normalize_ecli(citation.get("ecli")) if isinstance(citation, dict) else None
    if explicit:
        candidates.append(explicit)
    if candidates:
        namespaces = {ecli.split(":")[2] for ecli in candidates if ecli.count(":") >= 4}
        available = set(coverage.get("namespaces") or ()) | set(coverage.get("courts") or ())
        if namespaces and not (namespaces & available):
            return (f"giurisdizione {sorted(namespaces)} non presente nel registry offline", True)
        cass_years = set(coverage.get("cass_years") or ())
        cass = [ecli for ecli in candidates if ecli.startswith("ECLI:IT:CASS:")]
        if cass and cass_years:
            years = {int(ecli.split(":")[3]) for ecli in cass}
            if not (years & cass_years):
                return (f"copertura CASS insufficiente per l'anno {sorted(years)}", False)
    text = str(citation.get("text") or "") if isinstance(citation, dict) else ""
    if MERIT_COURT_RE.search(text):
        return ("citazione di merito senza ECLI: non verificabile offline", True)
    return (None, False)


def _citation_summary(citations: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    names = (
        "issue_aligned",
        "retrieved_only",
        "outside_profile",
        "unresolved",
        "fabricated_or_not_found",
    )
    counts = {name: 0 for name in names}
    items = list(citations)
    for item in items:
        relation = str(item.get("gold_v3_relation") or "unresolved")
        counts[relation] = counts.get(relation, 0) + 1
    return {"citations_total": len(items), **counts}


def summarize_grounding(tasks: Iterable[Mapping[str, Any]], *, n_tasks: int) -> dict[str, Any]:
    task_items = list(tasks)
    gog_by_task: dict[str, float] = {}
    coverage_by_task: dict[str, int] = {}
    totals = {"citations_total": 0, "issue_aligned": 0}
    for task in task_items:
        task_id = str(task.get("task_id") or "")
        summary = task.get("citation_summary") or {}
        cited = int(summary.get("citations_total") or 0)
        aligned = int(summary.get("issue_aligned") or 0)
        gog_by_task[task_id] = aligned / cited if cited else 0.0
        coverage_by_task[task_id] = int(aligned > 0)
        totals["citations_total"] += cited
        totals["issue_aligned"] += aligned
    denominator = n_tasks if n_tasks > 0 else len(task_items)
    return {
        "gog": sum(gog_by_task.values()) / denominator if denominator else 0.0,
        "coverage": sum(coverage_by_task.values()) / denominator if denominator else 0.0,
        "gog_by_task": gog_by_task,
        "coverage_by_task": coverage_by_task,
        "gog_tasks_total": denominator,
        "tasks_reported": len(task_items),
        **totals,
    }


def records_from_results(path: Path) -> list[ResponseRecord]:
    source = _results_file(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Formato risultati non valido: {source}")
    records = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        task_id = str(item.get("task_id") or "").strip()
        response = item.get("response") or item.get("model_output")
        if task_id and isinstance(response, str) and response.strip():
            records.append(ResponseRecord(task_id, response.strip(), item.get("model")))
    if not records:
        raise ValueError(f"Nessuna risposta caricabile da {source}")
    return records


def records_from_csv(
    path: Path,
    *,
    profiles: QuestionProfiles,
    task_id_column: str | None,
    question_column: str | None,
    answer_column: str | None,
    model: str | None,
) -> list[ResponseRecord]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(8192)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        rows = list(csv.DictReader(handle, dialect=dialect))
    if not rows:
        raise ValueError(f"CSV vuoto: {path}")
    columns = list(rows[0])
    task_col = _column(columns, task_id_column, ("task_id", "Task ID", "id", "ID"), required=False)
    question_col = _column(columns, question_column, ("Domanda", "Question", "question", "query", "Query"), required=task_col is None)
    answer_col = _column(columns, answer_column, ("Risposta", "Risposte", "Answer", "answer", "response", "Response"), required=True)
    by_question = {_normalize_question(item.question): item.task_id for item in profiles.all() if item.question}
    records: list[ResponseRecord] = []
    for row_number, row in enumerate(rows, start=2):
        response = str(row.get(answer_col or "") or "").strip()
        if not response:
            continue
        task_id = str(row.get(task_col or "") or "").strip() if task_col else ""
        if not task_id and question_col:
            task_id = by_question.get(_normalize_question(str(row.get(question_col) or "")), "")
        if not task_id:
            raise ValueError(f"Task non riconosciuto alla riga {row_number}")
        records.append(ResponseRecord(task_id, response, model))
    return records


def _column(columns: list[str], requested: str | None, candidates: tuple[str, ...], *, required: bool) -> str | None:
    if requested:
        if requested not in columns:
            raise ValueError(f"Colonna {requested!r} assente; disponibili: {columns}")
        return requested
    lowered = {column.casefold(): column for column in columns}
    for candidate in candidates:
        if candidate.casefold() in lowered:
            return lowered[candidate.casefold()]
    if required:
        raise ValueError(f"Colonna richiesta non trovata; disponibili: {columns}")
    return None


def _normalize_question(value: str) -> str:
    return " ".join(value.casefold().split())


def _results_file(path: Path) -> Path:
    if path.is_file():
        return path
    for name in ("scores.json", "outputs.json"):
        candidate = path / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"scores.json/outputs.json non trovato in {path}")


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def render_markdown(payload: Mapping[str, Any]) -> str:
    summary = payload.get("summary") or {}
    lines = [
        "# Citation Grounding",
        "",
        f"- GOG: `{summary.get('gog', 0):.4f}`",
        f"- Coverage: `{summary.get('coverage', 0):.4f}`",
        f"- Backend: `local`",
        "",
        "| Task | Citazioni | Allineate | GOG | Coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for task in payload.get("tasks") or []:
        task_summary = task.get("citation_summary") or {}
        total = int(task_summary.get("citations_total") or 0)
        aligned = int(task_summary.get("issue_aligned") or 0)
        gog = aligned / total if total else 0.0
        lines.append(f"| `{task.get('task_id')}` | {total} | {aligned} | {gog:.3f} | {int(aligned > 0)} |")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Citation grounding locale di una run LegalITA.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path)
    source.add_argument("--results", type=Path)
    parser.add_argument("--backend", choices=["local"], default="local")
    parser.add_argument("--citation-registry", type=Path, default=None)
    parser.add_argument("--question-profiles", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--task-id-column", default=None)
    parser.add_argument("--question-column", default=None)
    parser.add_argument("--answer-column", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument(
        "--n-tasks",
        type=int,
        default=None,
        help=(
            "Denominatore di GOG e Coverage. Default: numero di task distinti presenti "
            "nell'input (dopo --task-ids). Usare 67 per confrontare con il benchmark "
            "completo: i task assenti contano zero."
        ),
    )
    parser.add_argument("--extractor-model", default=CITATION_EXTRACTOR_MODEL)
    parser.add_argument("--extractor-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--extractor-max-retries", type=int, default=2)
    parser.add_argument("--fast-path", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    if args.fast_path:
        os.environ["CITATION_EXTRACTOR_FAST_PATH"] = "1"
    registry, profiles_dir = resolve_bundle_paths(args.citation_registry, args.question_profiles)
    require_bundle(registry, profiles_dir)
    profiles = QuestionProfiles(profiles_dir)
    records = (
        records_from_csv(
            args.csv,
            profiles=profiles,
            task_id_column=args.task_id_column,
            question_column=args.question_column,
            answer_column=args.answer_column,
            model=args.model,
        )
        if args.csv
        else records_from_results(args.results)
    )
    if args.task_ids:
        wanted = set(args.task_ids)
        records = [record for record in records if record.task_id in wanted]
    if not records:
        raise SystemExit("Nessuna risposta da valutare")
    n_tasks = args.n_tasks if args.n_tasks is not None else len({record.task_id for record in records})
    label_source = args.results or args.csv
    label = (args.model or (label_source.stem if label_source else "run")).replace(" ", "_")
    out_dir = args.out_dir or _available_out_dir(DEFAULT_RESULTS_ROOT / label)
    payload = run_grounding(
        records,
        registry_path=registry,
        profiles_dir=profiles_dir,
        out_dir=out_dir,
        n_tasks=n_tasks,
        extractor_model=args.extractor_model,
        extractor_timeout_seconds=args.extractor_timeout_seconds,
        extractor_max_retries=args.extractor_max_retries,
    )
    _print_report(payload, out_dir)
    return 0


def resolve_bundle_paths(
    registry: Path | None = None,
    profiles_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Registry and profiles paths: explicit argument, then environment, then default."""
    registry = registry or Path(os.environ.get("LEGALITA_CITATION_REGISTRY_PATH", DEFAULT_REGISTRY_PATH))
    profiles_dir = profiles_dir or Path(os.environ.get("LEGALITA_QUESTION_PROFILES_DIR", DEFAULT_PROFILES_DIR))
    return Path(registry), Path(profiles_dir)


def require_bundle(registry: Path, profiles_dir: Path, *, hint: str | None = None) -> None:
    """Fail early, with guidance, when the separately distributed bundle is missing."""
    missing: list[str] = []
    if not registry.is_file():
        missing.append(
            f"Registry locale non trovato: {registry}\n"
            "  (opzione --citation-registry o variabile LEGALITA_CITATION_REGISTRY_PATH)"
        )
    if not profiles_dir.is_dir():
        missing.append(
            f"Question profiles non trovati: {profiles_dir}\n"
            "  (opzione --question-profiles o variabile LEGALITA_QUESTION_PROFILES_DIR)"
        )
    if missing:
        raise SystemExit(
            "\n".join(missing)
            + "\n\nIl bundle registry + question profiles e' distribuito separatamente dal codice:"
            "\nestrarlo in data/citation_pool/ come descritto in docs/CITATION_GROUNDING.md."
            + (f"\n{hint}" if hint else "")
        )


def _available_out_dir(preferred: Path) -> Path:
    if not preferred.exists():
        return preferred
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return preferred.with_name(f"{preferred.name}-{stamp}")


def _print_report(payload: Mapping[str, Any], out_dir: Path) -> None:
    summary = payload.get("summary") or {}
    registry = payload.get("registry_info") or {}
    built_at = str(registry.get("built_at") or "unknown")[:10]
    print(
        f"GOG={float(summary.get('gog') or 0) * 100:.1f}%  "
        f"Coverage={float(summary.get('coverage') or 0) * 100:.1f}%  "
        f"Tasks={int(summary.get('gog_tasks_total') or 0)}  "
        f"backend=local  registry={built_at}"
    )
    print(f"results={out_dir.resolve()}")


if __name__ == "__main__":
    raise SystemExit(main())
