"""Public batch-summary API."""

from .service import summarize_batch_scores, summarize_consensus_diagnostics

__all__ = ["summarize_batch_scores", "summarize_consensus_diagnostics"]
