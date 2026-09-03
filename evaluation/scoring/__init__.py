"""Task scoring, citation diagnostics and batch summaries."""

from .service import (
    CitationExistenceScore,
    CitationGroundingError,
    criterion_result_from_judge_output,
    evaluate_citation_existence,
    format_citation_context,
    is_citation_scoring_applicable,
    map_citation_report,
    public_citation_status,
    score_batch,
    score_task,
    summarize_batch_scores,
    summarize_consensus_diagnostics,
)

__all__ = [
    "CitationExistenceScore",
    "CitationGroundingError",
    "criterion_result_from_judge_output",
    "evaluate_citation_existence",
    "format_citation_context",
    "is_citation_scoring_applicable",
    "map_citation_report",
    "public_citation_status",
    "score_batch",
    "score_task",
    "summarize_batch_scores",
    "summarize_consensus_diagnostics",
]
