"""
Scoring dei task del benchmark.
Adattato da Harvey LAB (MIT) per il benchmark legale italiano.

Per ogni task:
1. Chiama il judge per ogni criterio
2. Calcola all-pass (1.0 solo se tutti i criteri required passano)
3. Registra required e bonus criterion pass rate come diagnostica
4. Restituisce TaskScore completo

Uso tipico:
    from evaluation.scoring import score_task

    task_score = score_task(
        task        = task,
        model_output = "risposta del modello...",
        model        = "gpt-4o",
        judge        = judge,
    )
"""

from __future__ import annotations

import json
import logging
import inspect
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from datetime import datetime, timezone

try:
    from evaluation.grounding_metrics import enrich_grounding_summary
except ImportError:
    enrich_grounding_summary = None
from evaluation.judge import Judge
from legal_ita.schemas import BenchmarkTask, ConsensusResult, CriterionResult, JudgeVote, TaskScore
from legal_ita.modeling.usage import aggregate_model_call_metrics

log = logging.getLogger(__name__)

PUBLIC_CITATION_STATUSES = (
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

MAX_CITATION_CONTEXT_CANDIDATES = 5


class CitationGroundingError(RuntimeError):
    """Errore non recuperabile del citation grounding in modalita strict."""


def public_citation_status(status: str) -> str:
    if status == "resolved":
        return "resolved_local_registry_exact"
    if status == "not_found":
        return "not_found_in_index"
    if status == "insufficient_data":
        return "not_found_in_index"
    if status == "ambiguous":
        return "ambiguous_local_registry"
    if status in PUBLIC_CITATION_STATUSES:
        return status
    return "resolver_error"


@dataclass(frozen=True)
class CitationExistenceScore:
    citation_score: float | None = None
    citation_coverage_score: float | None = None
    citation_relevance_score: float | None = None
    citation_fabrication_rate: float | None = None
    citation_verdict: str = "not_applicable"
    citation_scoring_status: str = "not_applicable"
    citation_gold_count: int = 0
    citation_required_count: int = 0
    citation_required_matched_count: int = 0
    citation_required_missing_count: int = 0
    citation_required_unresolved_count: int = 0
    citation_acceptable_count: int = 0
    citation_acceptable_matched_count: int = 0
    citations_extracted_count: int = 0
    citations_matched_gold_count: int = 0
    citations_relevant_count: int = 0
    citations_outside_gold_count: int = 0
    citations_fabricated_count: int = 0
    citations_unresolved_count: int = 0
    citation_evaluation_error: str | None = None
    citation_extraction_status: str = "not_applicable"
    citation_extraction_error_count: int = 0
    citation_failure_reasons: list[str] = field(default_factory=list)
    citation_unresolved_reasons: list[str] = field(default_factory=list)
    citation_coverage: dict[str, Any] = field(default_factory=dict)
    citation_relevance: dict[str, Any] = field(default_factory=dict)
    citation_existence: dict[str, Any] = field(default_factory=dict)
    citation_results: list[dict[str, Any]] = field(default_factory=list)
    citation_score_bounds: dict[str, float | None] = field(default_factory=dict)
    citation_coverage_bounds: dict[str, float | None] = field(default_factory=dict)
    citation_relevance_bounds: dict[str, float | None] = field(default_factory=dict)


def is_citation_scoring_applicable(task: Any) -> bool:
    task_type = task.get("task_type") if isinstance(task, Mapping) else getattr(task, "task_type", None)
    task_id = task.get("task_id") if isinstance(task, Mapping) else getattr(task, "task_id", None)
    if task_type == "bullshit":
        return False
    if task_type:
        return True
    return not str(task_id or "").replace("\\", "/").startswith("bullshit/")


def evaluate_citation_existence(
    *,
    citation_results: list[dict[str, Any]],
    citation_counts: dict[str, int] | None = None,
    citation_grounding_enabled: bool = True,
) -> CitationExistenceScore:
    if not citation_grounding_enabled:
        return CitationExistenceScore(
            citation_results=[dict(item) for item in citation_results],
            citation_extraction_status="disabled",
        )

    counts = dict(citation_counts or {})
    extracted = len(citation_results)
    fabricated = int(counts.get("confirmed_fabricated", 0) or 0)
    unresolved_statuses = {"not_found_in_index", "resolver_error", "citation_extraction_error"}
    unresolved = sum(int(counts.get(status, 0) or 0) for status in unresolved_statuses)
    confirmed = sum(
        int(counts.get(status, 0) or 0)
        for status in (
            "resolved_local_registry_exact",
            "resolved_local_registry_incomplete",
            "resolved_local_registry_metadata_mismatch",
            "ambiguous_local_registry",
        )
    )
    status = "not_cited" if extracted == 0 else "complete"
    verdict = "nc" if extracted == 0 else "not_applicable"
    return CitationExistenceScore(
        citation_verdict=verdict,
        citation_scoring_status=status,
        citations_extracted_count=extracted,
        citations_fabricated_count=fabricated,
        citations_unresolved_count=unresolved,
        citation_extraction_error_count=int(counts.get("citation_extraction_error", 0) or 0),
        citation_unresolved_reasons=[status for status in sorted(unresolved_statuses) if counts.get(status)],
        citation_existence={
            "confirmed_count": confirmed,
            "unresolved_count": unresolved,
            "fabricated_count": fabricated,
            "local_registry_only": True,
        },
        citation_results=[dict(item) for item in citation_results],
    )


def map_citation_report(report: dict) -> tuple[list[dict], dict[str, int]]:
    counts = {status: 0 for status in PUBLIC_CITATION_STATUSES}
    mapped_results: list[dict] = []

    for result in report.get("results", []):
        mapped = dict(result)
        raw_existence_status = str(
            result.get("final_status")
            or result.get("existence_status")
            or result.get("status", "")
        )
        mapped["status"] = public_citation_status(raw_existence_status)
        mapped["final_status"] = mapped["status"]
        counts[mapped["status"]] += 1
        mapped_results.append(mapped)

    report_counts = report.get("counts") or {}
    if isinstance(report_counts, dict):
        for status in PUBLIC_CITATION_STATUSES:
            try:
                counts[status] = max(counts[status], int(report_counts.get(status, 0) or 0))
            except (TypeError, ValueError):
                continue

    return mapped_results, counts


def _candidate_eclis_for_context(
    candidates: Any,
    *,
    limit: int = MAX_CITATION_CONTEXT_CANDIDATES,
) -> list[str]:
    eclis: list[str] = []
    for candidate in candidates or []:
        if isinstance(candidate, Mapping):
            ecli = candidate.get("ecli") or candidate.get("matched_ecli")
        else:
            ecli = getattr(candidate, "ecli", None) or getattr(candidate, "matched_ecli", None)
        if not ecli:
            continue
        ecli_text = str(ecli)
        if ecli_text not in eclis:
            eclis.append(ecli_text)
        if len(eclis) >= limit:
            break
    return eclis


def format_citation_context(citation_results: list[dict], citation_counts: dict[str, int]) -> str:
    if not citation_results:
        return json.dumps(
            {
                "summary": "Nessuna citazione completa estratta dalla risposta.",
                "counts": citation_counts,
                "citations": [],
            },
            ensure_ascii=False,
            indent=2,
        )

    citations = []
    for result in citation_results:
        citation = result.get("citation", {})
        citations.append(
            {
                "status": result.get("status"),
                "existence_status": result.get("existence_status"),
                "final_status": result.get("final_status"),
                "existence_confirmed": result.get("existence_confirmed"),
                "existence_source": result.get("existence_source"),
                "identity_status": result.get("identity_status"),
                "local_registry_status": result.get("local_registry_status"),
                "citation_accuracy": result.get("citation_accuracy"),
                "metadata_mismatches": result.get("metadata_mismatches"),
                "requested_ecli_candidates": result.get("requested_ecli_candidates"),
                "matched_requested_ecli": result.get("matched_requested_ecli"),
                "homonymous_matches": result.get("homonymous_matches"),
                "raw_candidate_count": result.get("raw_candidate_count"),
                "compatible_candidate_count": result.get("compatible_candidate_count"),
                "final_candidate_count": result.get("final_candidate_count"),
                "text": citation.get("text"),
                "mentions": citation.get("mentions"),
                "ecli": citation.get("ecli"),
                "authority": citation.get("authority"),
                "namespace": citation.get("suggested_namespace"),
                "number": citation.get("number"),
                "year": citation.get("year"),
                "candidate_eclis": _candidate_eclis_for_context(result.get("candidates", [])),
                "error": result.get("error"),
            }
        )

    return json.dumps(
        {
            "instructions": (
                "Usa questi stati come verifica di esistenza delle fonti; "
                "non verificarle autonomamente."
            ),
            "counts": citation_counts,
            "citations": citations,
        },
        ensure_ascii=False,
        indent=2,
    )


def criterion_result_from_judge_output(
    *,
    criterion_id: str,
    criterion_title: str,
    scoring_type: str,
    category: str,
    judge_output: Any,
) -> CriterionResult:
    """Converte output legacy o multi-judge nel modello pubblico CriterionResult."""
    verdict = "unresolved"
    reasoning = "Errore tecnico del judge: verdetto non disponibile."
    consensus_method: str | None = None
    supporting_judges: list[str] = []
    tie_breaker_used = False
    judge_votes: list[JudgeVote] = []

    if isinstance(judge_output, ConsensusResult):
        verdict = judge_output.verdict
        reasoning = judge_output.reasoning
        consensus_method = judge_output.consensus_method
        supporting_judges = list(judge_output.supporting_judges)
        tie_breaker_used = judge_output.tie_breaker_used
        judge_votes = list(judge_output.judge_votes)
    elif isinstance(judge_output, JudgeVote):
        judge_votes = [judge_output]
        if judge_output.status == "ok" and judge_output.verdict in ("pass", "fail"):
            verdict = judge_output.verdict
            reasoning = judge_output.reasoning or ""
            supporting_judges = [judge_output.judge_id]
        else:
            reasoning = "Errore tecnico del judge: verdetto non disponibile."
    elif isinstance(judge_output, Mapping):
        raw_verdict = str(judge_output.get("verdict", "")).lower().strip()
        if raw_verdict in ("pass", "fail", "unresolved"):
            verdict = raw_verdict
            reasoning = str(judge_output.get("reasoning") or "")
        consensus_method = judge_output.get("consensus_method")  # type: ignore[assignment]
        supporting_judges = list(judge_output.get("supporting_judges") or [])
        tie_breaker_used = bool(judge_output.get("tie_breaker_used") or False)
        for item in judge_output.get("judge_votes") or []:
            try:
                judge_votes.append(
                    item if isinstance(item, JudgeVote) else JudgeVote.model_validate(item)
                )
            except Exception:
                log.debug("JudgeVote legacy non valido ignorato: %r", item)

    return CriterionResult(
        id=criterion_id,
        title=criterion_title,
        verdict=verdict,
        reasoning=reasoning,
        scoring_type=scoring_type,
        category=category,
        consensus_method=consensus_method,
        supporting_judges=supporting_judges,
        tie_breaker_used=tie_breaker_used,
        judge_votes=judge_votes,
    )


def summarize_consensus_diagnostics(criteria_results: list[CriterionResult]) -> dict[str, Any]:
    """Metriche diagnostiche del consenso multi-judge, senza toccare lo score primario."""
    total_criteria = len(criteria_results)
    method_counts: Counter[str] = Counter(
        result.consensus_method or "single" for result in criteria_results
    )
    c_calls = sum(
        1
        for result in criteria_results
        if any(vote.judge_id == "C" for vote in result.judge_votes)
    )
    individual_valid: dict[str, Counter[str]] = {
        "A": Counter(),
        "B": Counter(),
        "C": Counter(),
    }
    combination_counts: Counter[str] = Counter()
    ab_pairs: list[tuple[str, str]] = []

    for result in criteria_results:
        by_judge = {vote.judge_id: vote for vote in result.judge_votes}
        labels: list[str] = []
        for judge_id in ("A", "B", "C"):
            vote = by_judge.get(judge_id)
            labels.append(f"{judge_id}:{_vote_label(vote)}")
            if vote and vote.status == "ok" and vote.verdict in ("pass", "fail"):
                individual_valid[judge_id][vote.verdict] += 1
        if by_judge:
            combination_counts["|".join(labels)] += 1

        vote_a = by_judge.get("A")
        vote_b = by_judge.get("B")
        if (
            vote_a
            and vote_b
            and vote_a.status == "ok"
            and vote_b.status == "ok"
            and vote_a.verdict in ("pass", "fail")
            and vote_b.verdict in ("pass", "fail")
        ):
            ab_pairs.append((vote_a.verdict, vote_b.verdict))

    return {
        "criteria_evaluated": total_criteria,
        "initial_agreement_rate": _rate(method_counts["initial_agreement"], total_criteria),
        "tie_breaker_rate": _rate(method_counts["tie_breaker"], total_criteria),
        "recovery_rate": _rate(method_counts["recovery_agreement"], total_criteria),
        "unresolved_rate": _rate(method_counts["unresolved"], total_criteria),
        "judge_c_evaluated_criteria": c_calls,
        "judge_call_counts": {
            judge_id: sum(
                1
                for result in criteria_results
                for vote in result.judge_votes
                if vote.judge_id == judge_id
            )
            for judge_id in ("A", "B", "C")
        },
        "individual_pass_rates": {
            judge_id: _rate(counts["pass"], counts["pass"] + counts["fail"])
            for judge_id, counts in individual_valid.items()
        },
        "vote_combination_matrix": dict(combination_counts),
        "cohen_kappa_ab": _cohen_kappa(ab_pairs),
    }


def summarize_batch_scores(scores: list[TaskScore]) -> dict[str, Any]:
    """Aggregati batch con esclusione dei task incompleti dai denominatori primari."""
    complete_scores = [score for score in scores if score.reasoning_scoring_status == "complete"]
    n_total = len(scores)
    n_complete = len(complete_scores)
    n_incomplete = n_total - n_complete
    n_allpass = sum(1 for score in complete_scores if score.reasoning_all_pass is True)
    total_required = sum(score.n_required for score in complete_scores)
    total_required_passed = sum(score.n_required_passed for score in complete_scores)
    total_bonus_valid = sum(score.n_bonus - score.n_bonus_unresolved for score in complete_scores)
    total_bonus_passed = sum(score.n_bonus_passed for score in complete_scores)
    total_criteria = sum(score.n_criteria for score in scores)
    total_unresolved = sum(score.n_unresolved for score in scores)
    criteria_results = [
        result
        for score in scores
        for result in score.criteria_results
    ]
    method_counts: Counter[str] = Counter(
        result.consensus_method or "single" for result in criteria_results
    )
    judge_call_counts = {
        judge_id: sum(
            1
            for result in criteria_results
            for vote in result.judge_votes
            if vote.judge_id == judge_id
        )
        for judge_id in ("A", "B", "C")
    }
    citation_applicable = [
        score
        for score in scores
        if score.citation_scoring_status != "not_applicable"
        and score.citation_verdict != "not_applicable"
    ]
    citation_complete = [
        score for score in citation_applicable if score.citation_scoring_status == "complete"
    ]
    citation_pass = sum(1 for score in citation_complete if score.citation_verdict == "pass")
    citation_fail = sum(1 for score in citation_complete if score.citation_verdict == "fail")
    citation_score_values = [
        float(score.citation_score)
        for score in citation_complete
        if score.citation_score is not None
    ]
    citation_coverage_values = [
        float(score.citation_coverage_score)
        for score in citation_complete
        if score.citation_coverage_score is not None
    ]
    citation_relevance_values = [
        float(score.citation_relevance_score)
        for score in citation_complete
        if score.citation_relevance_score is not None
    ]
    citation_nc = sum(1 for score in citation_applicable if score.citation_verdict == "nc")
    citation_unresolved = sum(
        1 for score in citation_applicable if score.citation_verdict == "unresolved"
    )
    citation_not_applicable = n_total - len(citation_applicable)
    citation_required_total = sum(score.citation_required_count for score in citation_applicable)
    citation_required_matched = sum(
        score.citation_required_matched_count for score in citation_applicable
    )
    citations_evaluable = sum(
        int((score.citation_relevance or {}).get("evaluated_count") or 0)
        for score in citation_applicable
    )
    citations_relevant = sum(score.citations_relevant_count for score in citation_applicable)
    citations_fabricated = sum(score.citations_fabricated_count for score in citation_applicable)
    existence_unresolved = sum(
        int((score.citation_existence or {}).get("unresolved_count") or 0)
        for score in citation_applicable
    )
    citation_existence_unresolved_tasks = sum(
        1
        for score in citation_applicable
        if (score.citation_existence or {}).get("verdict") == "unresolved"
    )

    summary = {
        "n_tasks": n_total,
        "n_complete": n_complete,
        "n_incomplete": n_incomplete,
        "n_allpass": n_allpass,
        "allpass_rate": _rate(n_allpass, n_complete),
        "reasoning_allpass_rate": _rate(n_allpass, n_complete),
        "required_criterion_pass_rate": _rate(total_required_passed, total_required),
        "bonus_pass_rate": (
            _rate(total_bonus_passed, total_bonus_valid)
            if total_bonus_valid > 0
            else None
        ),
        "unresolved_rate": _rate(total_unresolved, total_criteria),
        "n_unresolved": total_unresolved,
        "criteria_evaluated": total_criteria,
        "judge_call_counts": judge_call_counts,
        "initial_agreements": method_counts["initial_agreement"],
        "tie_breakers": method_counts["tie_breaker"],
        "recoveries": method_counts["recovery_agreement"],
        "unresolved_criteria": method_counts["unresolved"],
        "judge_c_evaluated_criteria": judge_call_counts["C"],
        "citation_complete_tasks": len(citation_complete),
        "citation_incomplete_tasks": citation_unresolved,
        "citation_existence_unresolved_tasks": citation_existence_unresolved_tasks,
        "citation_not_cited_tasks": citation_nc,
        "citation_not_applicable_tasks": citation_not_applicable,
        "citation_pass_tasks": citation_pass,
        "citation_fail_tasks": citation_fail,
        "citation_perfect_tasks": sum(1 for value in citation_score_values if value == 1.0),
        "citation_perfect_rate": _rate(
            sum(1 for value in citation_score_values if value == 1.0),
            len(citation_score_values),
        ),
        "citation_mean_score": _mean(citation_score_values),
        "citation_mean_coverage_score": _mean(citation_coverage_values),
        "citation_mean_relevance_score": _mean(citation_relevance_values),
        "citation_all_pass_rate": _rate(citation_pass, citation_pass + citation_fail),
        "citation_pass_rate": _rate(citation_pass, len(citation_complete)),
        "citation_fail_rate": _rate(citation_fail, len(citation_complete)),
        "citation_nc_rate": _rate(citation_nc, len(citation_applicable)),
        "citation_unresolved_rate": _rate(citation_unresolved, len(citation_applicable)),
        "citation_complete_coverage": _rate(len(citation_complete), len(citation_applicable)),
        "citation_scoring_coverage": _rate(len(citation_complete), len(citation_applicable)),
        "citation_required_total": citation_required_total,
        "citation_required_matched_total": citation_required_matched,
        "citation_required_missing_total": sum(
            score.citation_required_missing_count for score in citation_applicable
        ),
        "citation_required_unresolved_total": sum(
            score.citation_required_unresolved_count for score in citation_applicable
        ),
        "required_citation_coverage_rate": _rate(
            citation_required_matched, citation_required_total
        ),
        "citation_minimum_complete_tasks": sum(
            1
            for score in citation_applicable
            if (score.citation_coverage or {}).get("verdict") == "pass"
        ),
        "citation_minimum_incomplete_tasks": sum(
            1
            for score in citation_applicable
            if (score.citation_coverage or {}).get("verdict") == "fail"
        ),
        "citation_relevant_rate": _rate(citations_relevant, citations_evaluable),
        "citation_outside_gold_tasks": sum(
            1 for score in citation_applicable if score.citations_outside_gold_count > 0
        ),
        "citation_fabrication_rate": _rate(citations_fabricated, citations_evaluable + citations_fabricated),
        "confirmed_fabrication_rate": _rate(citations_fabricated, citations_evaluable + citations_fabricated),
        "unresolved_verification_rate": _rate(existence_unresolved, citations_evaluable),
        "tasks_with_citations": sum(
            1 for score in citation_applicable if score.citations_extracted_count > 0
        ),
        "citation_gold_total": sum(score.citation_gold_count for score in citation_applicable),
        "citations_extracted_total": sum(
            score.citations_extracted_count for score in citation_applicable
        ),
        "citations_matched_gold_total": sum(
            score.citations_matched_gold_count for score in citation_applicable
        ),
        "citations_relevant_total": citations_relevant,
        "citations_evaluable_total": citations_evaluable,
        "citations_outside_gold_total": sum(
            score.citations_outside_gold_count for score in citation_applicable
        ),
        "citations_fabricated_total": citations_fabricated,
        "citations_unresolved_total": sum(
            score.citations_unresolved_count for score in citation_applicable
        ),
        "citation_extraction_error_total": sum(
            score.citation_extraction_error_count for score in citation_applicable
        ),
        "citation_extraction_error_tasks": sum(
            1 for score in citation_applicable if score.citation_extraction_status == "error"
        ),
        "model_call": aggregate_model_call_metrics(scores),
        "by_macro_area": _summarize_by_macro_area(scores),
    }
    if enrich_grounding_summary is None:
        return summary
    return enrich_grounding_summary(summary, scores)


def _vote_label(vote: JudgeVote | None) -> str:
    if vote is None:
        return "not_called"
    if vote.status != "ok":
        return "error"
    return str(vote.verdict)


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _summarize_by_macro_area(scores: list[TaskScore]) -> dict[str, dict[str, Any]]:
    areas: dict[str, list[TaskScore]] = {}
    for score in scores:
        area = score.task_id.split("/", 1)[0] if "/" in score.task_id else "unknown"
        areas.setdefault(area, []).append(score)

    out: dict[str, dict[str, Any]] = {}
    for area, items in areas.items():
        reasoning_complete = [item for item in items if item.reasoning_scoring_status == "complete"]
        citation_applicable = [
            item
            for item in items
            if item.citation_scoring_status != "not_applicable"
            and item.citation_verdict != "not_applicable"
        ]
        citation_complete = [
            item for item in citation_applicable if item.citation_scoring_status == "complete"
        ]
        out[area] = {
            "n_tasks": len(items),
            "n_citation_applicable_tasks": len(citation_applicable),
            "reasoning_allpass_rate": _rate(
                sum(1 for item in reasoning_complete if item.reasoning_all_pass is True),
                len(reasoning_complete),
            ),
            "reasoning_criterion_pass_rate": _rate(
                sum(item.n_required_passed for item in reasoning_complete),
                sum(item.n_required for item in reasoning_complete),
            ),
            "citation_pass_rate": _rate(
                sum(1 for item in citation_complete if item.citation_verdict == "pass"),
                len(citation_complete),
            ),
            "citation_complete_coverage": _rate(len(citation_complete), len(citation_applicable)),
            "citation_nc_rate": _rate(
                sum(1 for item in citation_applicable if item.citation_verdict == "nc"),
                len(citation_applicable),
            ),
            "citation_unresolved_rate": _rate(
                sum(1 for item in citation_applicable if item.citation_verdict == "unresolved"),
                len(citation_applicable),
            ),
            "required_citation_coverage_rate": _rate(
                sum(item.citation_required_matched_count for item in citation_applicable),
                sum(item.citation_required_count for item in citation_applicable),
            ),
        }
    return out


def _cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    n = len(pairs)
    if n == 0:
        return None

    observed = sum(1 for a, b in pairs if a == b) / n
    a_counts = Counter(a for a, _ in pairs)
    b_counts = Counter(b for _, b in pairs)
    expected = sum(
        (a_counts[label] / n) * (b_counts[label] / n)
        for label in ("pass", "fail")
    )
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def _log_criterion_consensus(
    task_id: str,
    criterion_id: str,
    criterion_result: CriterionResult,
) -> None:
    if not criterion_result.judge_votes:
        log.info(
            "%s / %s - verdict=%s",
            task_id,
            criterion_id,
            criterion_result.verdict.upper(),
        )
        return

    by_judge = {vote.judge_id: vote for vote in criterion_result.judge_votes}
    labels = []
    for judge_id in ("A", "B", "C"):
        vote = by_judge.get(judge_id)
        if vote is not None:
            labels.append(f"{judge_id}={_vote_label(vote).upper()}")
    method = criterion_result.consensus_method or "single"
    log.info(
        "%s / %s - %s - %s",
        task_id,
        criterion_id,
        " ".join(labels),
        method,
    )


def score_task(
    task: BenchmarkTask,
    model_output: str,
    model: str,
    judge: Judge | None = None,
    citation_service: Any | None = None,
    citation_grounding_enabled: bool = False,
    strict_citation_grounding: bool = False,
) -> TaskScore:
    """
    Valuta la risposta di un modello su un singolo task.

    Args:
        task:         Il BenchmarkTask con query e criteri.
        model_output: La risposta del modello valutato.
        model:        Nome del modello (es. "gpt-4o").
        judge:        Istanza di Judge. Se None ne crea una.

    Returns:
        TaskScore completo con verdetti per ogni criterio.
    """
    if judge is None:
        judge = Judge()

    citation_results: list[dict] = []
    citation_counts = {status: 0 for status in PUBLIC_CITATION_STATUSES}
    citation_hard_fail = False
    citation_registry_built_at: str | None = None
    citation_registry_index_name: str | None = None
    citation_extraction_diagnostics: dict[str, Any] = {}
    citation_extraction_attempt_diagnostics: list[dict[str, Any]] = []
    evaluation_error: str | None = None
    citation_scoring_applicable = is_citation_scoring_applicable(task)

    if citation_grounding_enabled and citation_scoring_applicable:
        if citation_service is None:
            raise RuntimeError(
                "Per integrare il grounding nello scoring serve un citation_service esplicito. "
                "Il comando pubblico legalita-grounding viene eseguito separatamente."
            )
        registry_info = citation_service.get_registry_info()
        citation_registry_built_at = registry_info.get("built_at")
        citation_registry_index_name = registry_info.get("index_name")
        check_text = citation_service.check_text
        if "task_id" in inspect.signature(check_text).parameters:
            citation_report = check_text(model_output, task_id=task.task_id)
        else:
            citation_report = check_text(model_output)
        citation_results, citation_counts = map_citation_report(citation_report)
        citation_extraction_diagnostics = dict(
            citation_report.get("citation_extraction_diagnostics") or {}
        )
        citation_extraction_attempt_diagnostics = list(
            citation_report.get("citation_extraction_attempt_diagnostics") or []
        )
        citation_hard_fail = citation_counts.get("confirmed_fabricated", 0) > 0

        extraction_error = citation_report.get("citation_extraction_error")
        if extraction_error:
            citation_counts["citation_extraction_error"] = max(
                citation_counts.get("citation_extraction_error", 0),
                1,
            )
            evaluation_error = f"Citation extraction error: {extraction_error}"

        if citation_counts.get("resolver_error", 0) > 0:
            errors = [
                str(result.get("error"))
                for result in citation_results
                if result.get("status") == "resolver_error" and result.get("error")
            ]
            resolver_error = (
                "Citation grounding resolver_error"
                + (": " + " | ".join(errors) if errors else "")
            )
            evaluation_error = (
                f"{evaluation_error} | {resolver_error}"
                if evaluation_error
                else resolver_error
            )
            if strict_citation_grounding:
                raise CitationGroundingError(evaluation_error)

    citation_context = (
        format_citation_context(citation_results, citation_counts)
        if citation_grounding_enabled and citation_scoring_applicable
        else None
    )
    citation_gold_score = evaluate_citation_existence(
        citation_results=citation_results,
        citation_counts=citation_counts,
        citation_grounding_enabled=citation_grounding_enabled,
    )
    citation_results = citation_gold_score.citation_results

    criteria_results: list[CriterionResult] = []
    n_passed = 0
    n_unresolved = 0
    n_required = 0
    n_required_passed = 0
    n_required_unresolved = 0
    n_bonus = 0
    n_bonus_passed = 0
    n_bonus_unresolved = 0

    for criterion in task.criteria:
        result = judge.evaluate(
            task_description=task.query,
            agent_output=model_output,
            criterion_title=criterion.title,
            match_criteria=criterion.match_criteria,
            citation_context=citation_context,
        )
        criterion_result = criterion_result_from_judge_output(
            criterion_id=criterion.id,
            criterion_title=criterion.title,
            scoring_type=criterion.scoring_type,
            category=criterion.category,
            judge_output=result,
        )
        verdict = criterion_result.verdict

        if verdict == "pass":
            n_passed += 1
        elif verdict == "unresolved":
            n_unresolved += 1

        if criterion.scoring_type == "required":
            n_required += 1
            if verdict == "pass":
                n_required_passed += 1
            elif verdict == "unresolved":
                n_required_unresolved += 1
        elif criterion.scoring_type == "bonus":
            n_bonus += 1
            if verdict == "pass":
                n_bonus_passed += 1
            elif verdict == "unresolved":
                n_bonus_unresolved += 1

        criteria_results.append(criterion_result)
        _log_criterion_consensus(task.task_id, criterion.id, criterion_result)

    n_criteria = len(task.criteria)
    scoring_status = "incomplete" if n_required_unresolved > 0 else "complete"
    if scoring_status == "incomplete":
        content_all_pass = None
        reasoning_all_pass = None
        reasoning_score = None
        required_pass_rate = None
    else:
        content_all_pass = n_required > 0 and n_required_passed == n_required
        reasoning_all_pass = content_all_pass
        reasoning_score = 1.0 if reasoning_all_pass else 0.0
        required_pass_rate = n_required_passed / n_required if n_required > 0 else 0.0
    all_pass = reasoning_all_pass
    score = reasoning_score
    valid_bonus = n_bonus - n_bonus_unresolved
    bonus_pass_rate = n_bonus_passed / valid_bonus if valid_bonus > 0 else None
    unresolved_rate = n_unresolved / n_criteria if n_criteria > 0 else 0.0

    reasoning_parts = [
        f"Reasoning: {n_required_passed}/{n_required} criteri required superati.",
    ]
    if n_bonus > 0:
        reasoning_parts.append(f"Bonus: {n_bonus_passed}/{n_bonus} ottenuti.")
    if scoring_status == "incomplete":
        reasoning_parts.append(
            f"VALUTAZIONE INCOMPLETA: {n_required_unresolved} required unresolved."
        )
    elif n_required == 0:
        reasoning_parts.append("NESSUN CRITERIO REQUIRED. FAIL.")
    elif content_all_pass:
        reasoning_parts.append("ALL PASS.")
    else:
        reasoning_parts.append(
            f"FALLITI {n_required - n_required_passed} REQUIRED."
        )
    if n_unresolved:
        reasoning_parts.append(f"UNRESOLVED: {n_unresolved}/{n_criteria} criteri.")
    summary_parts = [" ".join(reasoning_parts)]
    if citation_gold_score.citation_scoring_status != "not_applicable":
        summary_parts.append(_citation_summary(citation_gold_score, citation_hard_fail, citation_counts))
    if evaluation_error:
        summary_parts.append("ATTENZIONE: citation grounding non affidabile.")
    summary = " ".join(summary_parts)

    log.info(f"{task.task_id} [{model}] → {summary}")

    return TaskScore(
        task_id=task.task_id,
        model=model,
        model_output=model_output,
        citation_results=citation_results,
        citation_counts=citation_counts,
        citation_hard_fail=citation_hard_fail,
        citation_registry_built_at=citation_registry_built_at,
        citation_registry_index_name=citation_registry_index_name,
        citation_score=citation_gold_score.citation_score,
        citation_coverage_score=citation_gold_score.citation_coverage_score,
        citation_relevance_score=citation_gold_score.citation_relevance_score,
        citation_fabrication_rate=citation_gold_score.citation_fabrication_rate,
        citation_score_bounds=citation_gold_score.citation_score_bounds,
        citation_coverage_bounds=citation_gold_score.citation_coverage_bounds,
        citation_relevance_bounds=citation_gold_score.citation_relevance_bounds,
        citation_verdict=citation_gold_score.citation_verdict,  # type: ignore[arg-type]
        citation_scoring_status=citation_gold_score.citation_scoring_status,  # type: ignore[arg-type]
        citation_gold_count=citation_gold_score.citation_gold_count,
        citation_required_count=citation_gold_score.citation_required_count,
        citation_required_matched_count=citation_gold_score.citation_required_matched_count,
        citation_required_missing_count=citation_gold_score.citation_required_missing_count,
        citation_required_unresolved_count=citation_gold_score.citation_required_unresolved_count,
        citation_acceptable_count=citation_gold_score.citation_acceptable_count,
        citation_acceptable_matched_count=citation_gold_score.citation_acceptable_matched_count,
        citations_extracted_count=citation_gold_score.citations_extracted_count,
        citations_matched_gold_count=citation_gold_score.citations_matched_gold_count,
        citations_relevant_count=citation_gold_score.citations_relevant_count,
        citations_outside_gold_count=citation_gold_score.citations_outside_gold_count,
        citations_fabricated_count=citation_gold_score.citations_fabricated_count,
        citations_unresolved_count=citation_gold_score.citations_unresolved_count,
        citation_evaluation_error=citation_gold_score.citation_evaluation_error,
        citation_extraction_status=citation_gold_score.citation_extraction_status,
        citation_extraction_error_count=citation_gold_score.citation_extraction_error_count,
        citation_extraction_diagnostics=citation_extraction_diagnostics,
        citation_extraction_attempt_diagnostics=citation_extraction_attempt_diagnostics,
        citation_failure_reasons=citation_gold_score.citation_failure_reasons,
        citation_unresolved_reasons=citation_gold_score.citation_unresolved_reasons,
        citation_coverage=citation_gold_score.citation_coverage,
        citation_relevance=citation_gold_score.citation_relevance,
        citation_existence=citation_gold_score.citation_existence,
        evaluation_error=evaluation_error,
        score=score,
        all_pass=all_pass,
        reasoning_score=reasoning_score,
        reasoning_all_pass=reasoning_all_pass,
        reasoning_scoring_status=scoring_status,  # type: ignore[arg-type]
        content_all_pass=content_all_pass,
        scoring_status=scoring_status,
        n_criteria=n_criteria,
        n_passed=n_passed,
        n_unresolved=n_unresolved,
        n_required=n_required,
        n_required_passed=n_required_passed,
        n_required_unresolved=n_required_unresolved,
        n_bonus=n_bonus,
        n_bonus_passed=n_bonus_passed,
        n_bonus_unresolved=n_bonus_unresolved,
        required_pass_rate=required_pass_rate,
        bonus_pass_rate=bonus_pass_rate,
        unresolved_rate=unresolved_rate,
        summary=summary,
        criteria_results=criteria_results,
        judge_model=getattr(judge, "model", "unknown"),
        judge_strategy=getattr(judge, "strategy", "single"),
        judge_models=getattr(
            judge,
            "judge_models",
            {"A": getattr(judge, "model", "unknown")},
        ),
        judge_diagnostics=summarize_consensus_diagnostics(criteria_results),
        scored_at=datetime.now(timezone.utc).isoformat(),
    )


def _citation_summary(citation_gold_score: Any, citation_hard_fail: bool, citation_counts: dict[str, int]) -> str:
    if citation_gold_score.citation_verdict == "pass":
        text = _citation_score_summary(citation_gold_score)
    elif citation_gold_score.citation_verdict == "fail":
        text = _citation_score_summary(citation_gold_score)
    elif citation_gold_score.citation_verdict == "nc":
        text = "Citazioni: NC, nessuna sentenza citata."
    elif citation_gold_score.citation_verdict == "not_applicable":
        text = ""
    else:
        text = "Citazioni: UNRESOLVED, la verifica incompleta impedisce il calcolo dello score."
    if citation_hard_fail:
        text += (
            " CITATION HARD FAIL: "
            f"{citation_counts.get('confirmed_fabricated', 0)} fonte/i confirmed_fabricated."
        )
    if citation_gold_score.citation_evaluation_error:
        text += f" Diagnostica: {citation_gold_score.citation_evaluation_error}."
    return text


def _citation_score_summary(citation_gold_score: Any) -> str:
    score = _pct(getattr(citation_gold_score, "citation_score", None))
    coverage = getattr(citation_gold_score, "citation_coverage", {}) or {}
    relevance = getattr(citation_gold_score, "citation_relevance", {}) or {}
    existence = getattr(citation_gold_score, "citation_existence", {}) or {}
    fabricated = getattr(citation_gold_score, "citations_fabricated_count", 0) or 0
    outside = getattr(citation_gold_score, "citations_outside_gold_count", 0) or 0
    parts = [
        f"Citazioni: score {score}",
        (
            "coverage required "
            f"{coverage.get('required_matched', 0)}/{coverage.get('required_total', 0)} "
            f"({_pct(coverage.get('coverage_rate'))})"
        ),
        (
            "relevance "
            f"{relevance.get('relevant_count', 0)}/{relevance.get('evaluated_count', 0)} "
            f"({_pct(relevance.get('relevance_rate'))})"
        ),
    ]
    if fabricated:
        parts.append(f"{fabricated} pronuncia/e fabbricata/e")
    else:
        parts.append("nessuna fonte fabbricata")
    if outside:
        parts.append(f"{outside} pronuncia/e fuori gold")
    existence_unresolved = int(existence.get("unresolved_count") or 0)
    if existence_unresolved:
        confirmed = int(existence.get("confirmed_count") or 0)
        parts.append(
            f"{confirmed} pronuncia/e confermata/e e {existence_unresolved} con esistenza non verificata"
        )
    return "; ".join(parts) + "."


def _pct(value: Any) -> str:
    if value is None:
        return "n.d."
    try:
        return f"{float(value) * 100:.1f}%".replace(".", ",")
    except (TypeError, ValueError):
        return "n.d."


def score_batch(
    tasks: list[BenchmarkTask],
    outputs: dict[str, str],
    model: str,
    judge: Judge | None = None,
    citation_service: Any | None = None,
    citation_grounding_enabled: bool = False,
    strict_citation_grounding: bool = False,
) -> list[TaskScore]:
    """
    Valuta le risposte di un modello su una lista di task.

    Args:
        tasks:   Lista di BenchmarkTask.
        outputs: Dict {task_id: risposta_modello}.
        model:   Nome del modello valutato.
        judge:   Istanza di Judge condivisa. Se None ne crea una.

    Returns:
        Lista di TaskScore, uno per task.
        I task senza output in outputs vengono saltati con warning.
    """
    if judge is None:
        judge = Judge()

    scores: list[TaskScore] = []
    n_skip = 0

    for task in tasks:
        output = outputs.get(task.task_id)
        if output is None:
            log.warning(f"Nessun output per {task.task_id} — saltato")
            n_skip += 1
            continue

        score = score_task(
            task,
            output,
            model,
            judge,
            citation_service=citation_service,
            citation_grounding_enabled=citation_grounding_enabled,
            strict_citation_grounding=strict_citation_grounding,
        )
        scores.append(score)

    # riepilogo
    summary = summarize_batch_scores(scores)

    log.info("=" * 50)
    log.info(f"Modello:                      {model}")
    log.info(
        "Task valutati:                %d (%d completi, %d incompleti, %d saltati)",
        summary["n_tasks"],
        summary["n_complete"],
        summary["n_incomplete"],
        n_skip,
    )
    log.info(f"Reasoning all-pass rate:      {summary['reasoning_allpass_rate']:.1%}")
    log.info(
        "Required criterion pass rate: %.1f%%",
        summary["required_criterion_pass_rate"] * 100,
    )
    if summary["bonus_pass_rate"] is not None:
        log.info(f"Bonus pass rate:              {summary['bonus_pass_rate']:.1%}")
    log.info(f"Unresolved rate:              {summary['unresolved_rate']:.1%}")
    log.info(f"Criteri unresolved:           {summary['n_unresolved']}")
    log.info(
        "Citation perfect rate:        %.1f%% (%d/%d completi)",
        summary["citation_perfect_rate"] * 100,
        summary["citation_perfect_tasks"],
        summary["citation_complete_tasks"],
    )
    if summary["citation_mean_score"] is not None:
        log.info("Citation mean score:         %.1f%%", summary["citation_mean_score"] * 100)
    if summary["citation_mean_coverage_score"] is not None:
        log.info(
            "Citation mean coverage:      %.1f%%",
            summary["citation_mean_coverage_score"] * 100,
        )
    if summary["citation_mean_relevance_score"] is not None:
        log.info(
            "Citation mean relevance:     %.1f%%",
            summary["citation_mean_relevance_score"] * 100,
        )
    log.info(f"Citation fail rate:           {summary['citation_fail_rate']:.1%}")
    log.info(f"Citation NC rate:             {summary['citation_nc_rate']:.1%}")
    log.info(f"Citation unresolved rate:     {summary['citation_unresolved_rate']:.1%}")
    log.info(
        "Citation coverage:            %d/%d task con score completo",
        summary["citation_complete_tasks"],
        summary["n_tasks"] - summary["citation_not_applicable_tasks"],
    )
    log.info(
        "Citation existence unresolved: %d task",
        summary["citation_existence_unresolved_tasks"],
    )
    log.info(
        "Citation gold/matched/out:    gold=%d extracted=%d matched=%d outside=%d unresolved=%d",
        summary["citation_gold_total"],
        summary["citations_extracted_total"],
        summary["citations_matched_gold_total"],
        summary["citations_outside_gold_total"],
        summary["citations_unresolved_total"],
    )
    if any(summary["judge_call_counts"].values()):
        log.info(f"Criteri valutati:             {summary['criteria_evaluated']}")
        log.info(
            "Chiamate judge:               A=%d B=%d C=%d",
            summary["judge_call_counts"]["A"],
            summary["judge_call_counts"]["B"],
            summary["judge_call_counts"]["C"],
        )
        log.info(
            "Consenso:                     iniziali=%d tie-break=%d recovery=%d unresolved=%d",
            summary["initial_agreements"],
            summary["tie_breakers"],
            summary["recoveries"],
            summary["unresolved_criteria"],
        )
    log.info("=" * 50)

    return scores
