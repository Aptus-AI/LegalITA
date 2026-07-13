"""
Reportistica e visualizzazioni del benchmark legale italiano.


La reportistica separa tre dimensioni:
- legal reasoning sui task standard;
- citation grounding sui task standard citation-applicable;
- false-premise detection sui task Bullshit.

Uso tipico:
    python charts.py --results results/
    python charts.py --results results/ --latest
    python charts.py --results results/ --models claude-sonnet-4-6
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "legalita_matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from config import RESULTS_DIR
from taxonomy import normalize_macro_area

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

# ---------------------------------------------------------------------------
# Stile globale
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

PALETTE = [
    "#2E86AB", "#A23B72", "#F18F01", "#C73E1D",
    "#3B1F2B", "#44BBA4", "#E94F37", "#393E41",
]

STATUS_COLORS = {
    "complete": "#2E86AB",
    "nc": "#F18F01",
    "unresolved": "#A23B72",
    "unknown": "#8A8F98",
}

PERCENT_COLUMNS = {
    "reasoning_all_pass_rate",
    "criterion_pass_rate",
    "criterion_resolution_coverage",
    "citation_scoring_coverage",
    "mean_citation_score",
    "mean_citation_coverage",
    "mean_citation_relevance",
    "citation_perfect_rate",
    "citation_nc_rate",
    "citation_unresolved_rate",
    "citation_hard_fail_rate",
    "mean_task_fabrication_rate",
    "global_fabrication_rate",
    "false_premise_detection_rate",
    "bullshit_unresolved_rate",
}

TASK_NUMERIC_COLUMNS = [
    "reasoning_score",
    "score",
    "citation_score",
    "citation_coverage_score",
    "citation_relevance_score",
    "citation_fabrication_rate",
    "citation_required_count",
    "citation_required_matched_count",
    "citation_acceptable_count",
    "citation_acceptable_matched_count",
    "citations_extracted_count",
    "citations_relevant_count",
    "citations_outside_gold_count",
    "citations_fabricated_count",
    "citations_unresolved_count",
    "n_criteria",
    "n_passed",
    "n_unresolved",
]

CRITERIA_COLUMNS = [
    "run_id",
    "run_timestamp",
    "source_path",
    "series_label",
    "model",
    "task_id",
    "task_type",
    "macro_area",
    "is_standard_task",
    "is_bullshit_task",
    "reasoning_scoring_status",
    "criterion_id",
    "criterion_title",
    "verdict",
    "scoring_type",
    "category",
]


@dataclass(frozen=True)
class ScoreFrames:
    """DataFrame separati: una riga per task/run e una per criterio/run."""

    tasks: pd.DataFrame
    criteria: pd.DataFrame


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _path_timestamp(path: Path) -> str:
    """
    Estrae il timestamp dal path results/<model>/<timestamp>/scores.json.
    Restituisce stringa vuota se il path non rispetta la convenzione.
    """
    return path.parent.name if path.parent.name else ""


def _relative_parts(path: Path, results_dir: Path) -> tuple[str, ...]:
    try:
        return path.relative_to(results_dir).parts
    except ValueError:
        return path.parts


def _path_model(path: Path, results_dir: Path) -> str:
    """
    Estrae il nome modello da:
    - results/<model>/<timestamp>/scores.json
    - results/bullshit/<model>/<timestamp>/scores.json
    """
    parts = _relative_parts(path, results_dir)
    if len(parts) >= 4 and parts[0] == "bullshit":
        return parts[1]
    if len(parts) >= 3:
        return parts[0]
    return path.parent.parent.name


def _path_family(path: Path, results_dir: Path) -> str:
    parts = _relative_parts(path, results_dir)
    if len(parts) >= 4 and parts[0] == "bullshit":
        return "bullshit"
    return "standard"


def _run_id(model: str, timestamp: str) -> str:
    return f"{model}/{timestamp}" if timestamp else model


def _series_label(model: str, timestamp: str) -> str:
    return f"{model} · {timestamp}" if timestamp else model


def _score_paths(
        results_dir: Path,
        models: list[str] | None = None,
        since: str | None = None,
        latest_only: bool = False,
) -> list[Path]:
    all_paths = [
        p for p in results_dir.rglob("scores.json")
        if "charts" not in p.parts and "reviews" not in p.parts
    ]

    if since:
        all_paths = [p for p in all_paths if _path_timestamp(p) >= since]

    if models:
        models_set = set(models)
        all_paths = [
            p for p in all_paths
            if _path_model(p, results_dir) in models_set
        ]

    if latest_only:
        by_model_family: dict[tuple[str, str], Path] = {}
        for path in all_paths:
            key = (_path_model(path, results_dir), _path_family(path, results_dir))
            ts = _path_timestamp(path)
            current = by_model_family.get(key)
            if current is None or ts > _path_timestamp(current):
                by_model_family[key] = path
        all_paths = list(by_model_family.values())

    return sorted(all_paths, key=lambda p: (_path_model(p, results_dir), _path_timestamp(p), str(p)))


def macro_area_from_task_id(task_id: str) -> str:
    """Deriva e normalizza la macro-area dal prefisso storico del task_id."""
    raw_area = task_id.split("/")[0] if "/" in task_id else "unknown"
    return normalize_macro_area(raw_area, strict=False)


def _macro_area(task_score: dict[str, Any]) -> str:
    raw_area = task_score.get("macro_area")
    if raw_area:
        return normalize_macro_area(raw_area, strict=False)
    return macro_area_from_task_id(str(task_score.get("task_id", "")))


def _task_type(task_score: dict[str, Any]) -> str:
    task_type = task_score.get("task_type")
    task_id = str(task_score.get("task_id", "")).replace("\\", "/")
    if task_type == "bullshit" or task_id.startswith("bullshit/"):
        return "bullshit"
    return str(task_type or "standard")


def _is_bullshit(task_score: dict[str, Any]) -> bool:
    return _task_type(task_score) == "bullshit"


def _to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return None


def _count_value(task_score: dict[str, Any], key: str) -> int:
    value = task_score.get(key, 0)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _task_row(
        task_score: dict[str, Any],
        *,
        path: Path,
        results_dir: Path,
) -> dict[str, Any]:
    model = str(task_score.get("model") or _path_model(path, results_dir) or "unknown")
    timestamp = _path_timestamp(path)
    run_id = _run_id(model, timestamp)
    task_id = str(task_score.get("task_id", ""))
    task_type = _task_type(task_score)
    is_bullshit_task = task_type == "bullshit"
    citation_verdict = task_score.get("citation_verdict")
    citation_status = task_score.get("citation_scoring_status")
    citation_applicable = (
        not is_bullshit_task
        and citation_verdict != "not_applicable"
        and citation_status != "not_applicable"
    )

    return {
        "run_id": run_id,
        "run_timestamp": timestamp,
        "source_path": str(path),
        "series_label": _series_label(model, timestamp),
        "model": model,
        "task_id": task_id,
        "task_type": task_type,
        "macro_area": _macro_area(task_score),
        "is_standard_task": not is_bullshit_task,
        "is_bullshit_task": is_bullshit_task,
        "citation_applicable": citation_applicable,
        "reasoning_score": task_score.get("reasoning_score", task_score.get("score")),
        "reasoning_all_pass": _to_bool(
            task_score.get("reasoning_all_pass", task_score.get("all_pass"))
        ),
        "reasoning_scoring_status": task_score.get(
            "reasoning_scoring_status",
            task_score.get("scoring_status"),
        ),
        "score": task_score.get("score"),
        "all_pass": _to_bool(task_score.get("all_pass")),
        "scoring_status": task_score.get("scoring_status"),
        "verdict": task_score.get("verdict"),
        "citation_score": task_score.get("citation_score"),
        "citation_coverage_score": task_score.get("citation_coverage_score"),
        "citation_relevance_score": task_score.get("citation_relevance_score"),
        "citation_fabrication_rate": task_score.get("citation_fabrication_rate"),
        "citation_verdict": citation_verdict,
        "citation_scoring_status": citation_status,
        "citation_hard_fail": bool(task_score.get("citation_hard_fail") or False),
        "citation_required_count": _count_value(task_score, "citation_required_count"),
        "citation_required_matched_count": _count_value(
            task_score,
            "citation_required_matched_count",
        ),
        "citation_acceptable_count": _count_value(task_score, "citation_acceptable_count"),
        "citation_acceptable_matched_count": _count_value(
            task_score,
            "citation_acceptable_matched_count",
        ),
        "citations_extracted_count": _count_value(task_score, "citations_extracted_count"),
        "citations_relevant_count": _count_value(task_score, "citations_relevant_count"),
        "citations_outside_gold_count": _count_value(
            task_score,
            "citations_outside_gold_count",
        ),
        "citations_fabricated_count": _count_value(
            task_score,
            "citations_fabricated_count",
        ),
        "citations_unresolved_count": _count_value(
            task_score,
            "citations_unresolved_count",
        ),
        "n_criteria": _count_value(task_score, "n_criteria"),
        "n_passed": _count_value(task_score, "n_passed"),
        "n_unresolved": _count_value(task_score, "n_unresolved"),
    }


def _criterion_rows(
        task_score: dict[str, Any],
        task_row: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for cr in task_score.get("criteria_results", []) or []:
        rows.append({
            "run_id": task_row["run_id"],
            "run_timestamp": task_row["run_timestamp"],
            "source_path": task_row["source_path"],
            "series_label": task_row["series_label"],
            "model": task_row["model"],
            "task_id": task_row["task_id"],
            "task_type": task_row["task_type"],
            "macro_area": task_row["macro_area"],
            "is_standard_task": task_row["is_standard_task"],
            "is_bullshit_task": task_row["is_bullshit_task"],
            "reasoning_scoring_status": task_row["reasoning_scoring_status"],
            "criterion_id": cr.get("id"),
            "criterion_title": cr.get("title"),
            "verdict": cr.get("verdict"),
            "scoring_type": cr.get("scoring_type", "required"),
            "category": cr.get("category", "legal"),
        })
    return rows


def _finalize_frames(task_rows: list[dict[str, Any]], criterion_rows: list[dict[str, Any]]) -> ScoreFrames:
    if not task_rows:
        raise ValueError("Nessun scores.json valido trovato.")

    tasks = pd.DataFrame(task_rows)
    criteria = pd.DataFrame(criterion_rows, columns=CRITERIA_COLUMNS)

    for column in TASK_NUMERIC_COLUMNS:
        if column in tasks.columns:
            tasks[column] = pd.to_numeric(tasks[column], errors="coerce")

    if not criteria.empty:
        criteria = criteria.drop_duplicates(
            subset=["run_id", "model", "task_id", "criterion_id"],
            keep="first",
        )

    return ScoreFrames(tasks=tasks, criteria=criteria)


def load_score_frames(
        results_dir: Path,
        models: list[str] | None = None,
        since: str | None = None,
        latest_only: bool = False,
        areas: list[str] | None = None,
) -> ScoreFrames:
    """
    Carica gli scores.json in due DataFrame:
    - tasks: una riga per task e run;
    - criteria: una riga per criterio e run.

    Le run diverse dello stesso modello restano separate tramite run_id e
    series_label; nessuna deduplicazione avviene soltanto su model/task_id.
    """
    paths = _score_paths(
        results_dir,
        models=models,
        since=since,
        latest_only=latest_only,
    )
    log.info("File scores.json inclusi: %d", len(paths))

    task_rows: list[dict[str, Any]] = []
    criterion_rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            scores = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Errore lettura %s: %s", path, exc)
            continue

        if not isinstance(scores, list):
            log.warning("Formato inatteso in %s: attesa lista di score", path)
            continue

        for task_score in scores:
            if not isinstance(task_score, dict):
                continue
            row = _task_row(task_score, path=path, results_dir=results_dir)
            task_rows.append(row)
            criterion_rows.extend(_criterion_rows(task_score, row))

    frames = _finalize_frames(task_rows, criterion_rows)

    if areas:
        normalized = {normalize_macro_area(area, strict=False) for area in areas}
        tasks = frames.tasks[frames.tasks["macro_area"].isin(normalized)].copy()
        criteria = frames.criteria
        if not criteria.empty:
            criteria = criteria[criteria["macro_area"].isin(normalized)].copy()
        frames = ScoreFrames(tasks=tasks, criteria=criteria)
        if frames.tasks.empty:
            raise ValueError(f"Nessun task trovato per le macro-aree: {sorted(normalized)}")

    return frames


def load_all_scores(
        results_dir: Path,
        models: list[str] | None = None,
        since: str | None = None,
        latest_only: bool = False,
) -> pd.DataFrame:
    """Compatibilita legacy: restituisce il DataFrame criteri."""
    return load_score_frames(
        results_dir,
        models=models,
        since=since,
        latest_only=latest_only,
    ).criteria


# ---------------------------------------------------------------------------
# Metriche
# ---------------------------------------------------------------------------

def _safe_rate(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _mean_or_none(values: pd.Series) -> float | None:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _sum_numeric(values: pd.Series) -> float:
    return float(pd.to_numeric(values, errors="coerce").fillna(0).sum())


def _dedup_tasks(tasks: pd.DataFrame) -> pd.DataFrame:
    return tasks.drop_duplicates(subset=["run_id", "model", "task_id"], keep="first").copy()


def _dedup_criteria(criteria: pd.DataFrame) -> pd.DataFrame:
    if criteria.empty:
        return criteria.copy()
    return criteria.drop_duplicates(
        subset=["run_id", "model", "task_id", "criterion_id"],
        keep="first",
    ).copy()


def _leaderboard_row(group: pd.DataFrame, criteria: pd.DataFrame) -> dict[str, Any]:
    group = _dedup_tasks(group)
    model = str(group["model"].iloc[0])
    run_id = str(group["run_id"].iloc[0])
    series_label = str(group["series_label"].iloc[0])
    run_timestamp = str(group["run_timestamp"].iloc[0])

    standard = group[group["is_standard_task"]].copy()
    reasoning_complete = standard[standard["reasoning_scoring_status"] == "complete"]
    n_standard = len(standard)
    n_reasoning_complete = len(reasoning_complete)
    n_reasoning_incomplete = n_standard - n_reasoning_complete
    reasoning_all_pass_rate = _safe_rate(
        int((reasoning_complete["reasoning_all_pass"] == True).sum()),  # noqa: E712
        n_reasoning_complete,
    )

    run_criteria = criteria[
        (criteria["run_id"] == run_id)
        & (criteria["model"] == model)
        & (criteria["is_standard_task"])
        & (criteria["reasoning_scoring_status"] == "complete")
    ].copy()
    resolved = run_criteria[run_criteria["verdict"].isin(["pass", "fail"])]
    criterion_pass_rate = _safe_rate(
        int((resolved["verdict"] == "pass").sum()),
        len(resolved),
    )
    criterion_resolution_coverage = _safe_rate(len(resolved), len(run_criteria))

    citation_applicable = standard[standard["citation_applicable"]].copy()
    citation_complete = citation_applicable[
        citation_applicable["citation_scoring_status"] == "complete"
    ].copy()
    citation_score_values = pd.to_numeric(citation_complete["citation_score"], errors="coerce")

    extracted = _sum_numeric(citation_complete["citations_extracted_count"])
    fabricated = _sum_numeric(citation_complete["citations_fabricated_count"])

    return {
        "model": model,
        "run_id": run_id,
        "series_label": series_label,
        "run_timestamp": run_timestamp,
        "n_standard_tasks": n_standard,
        "n_reasoning_complete_tasks": n_reasoning_complete,
        "n_reasoning_incomplete_tasks": n_reasoning_incomplete,
        "reasoning_all_pass_rate": reasoning_all_pass_rate,
        "criterion_pass_rate": criterion_pass_rate,
        "criterion_resolution_coverage": criterion_resolution_coverage,
        "n_citation_applicable_tasks": len(citation_applicable),
        "n_citation_complete_tasks": len(citation_complete),
        "citation_scoring_coverage": _safe_rate(
            len(citation_complete),
            len(citation_applicable),
        ),
        "mean_citation_score": _mean_or_none(citation_complete["citation_score"]),
        "mean_citation_coverage": _mean_or_none(citation_complete["citation_coverage_score"]),
        "mean_citation_relevance": _mean_or_none(citation_complete["citation_relevance_score"]),
        "citation_perfect_rate": _safe_rate(
            int((citation_score_values == 1.0).sum()),
            len(citation_complete),
        ),
        "citation_nc_rate": _safe_rate(
            int((citation_applicable["citation_verdict"] == "nc").sum()),
            len(citation_applicable),
        ),
        "citation_unresolved_rate": _safe_rate(
            int((citation_applicable["citation_verdict"] == "unresolved").sum()),
            len(citation_applicable),
        ),
        "citation_hard_fail_rate": _safe_rate(
            int((citation_complete["citation_hard_fail"] == True).sum()),  # noqa: E712
            len(citation_complete),
        ),
        "mean_task_fabrication_rate": _mean_or_none(
            citation_complete["citation_fabrication_rate"]
        ),
        "global_fabrication_rate": _safe_rate(fabricated, extracted),
    }


def leaderboard_table(
        frames: ScoreFrames | pd.DataFrame,
        out_path: Path | None = None,
) -> pd.DataFrame:
    """
    Tabella principale con metriche numeriche separate:
    reasoning, criterion pass rate e citation grounding.
    """
    if isinstance(frames, pd.DataFrame):
        raise TypeError("leaderboard_table richiede ScoreFrames; usa load_score_frames().")

    tasks = _dedup_tasks(frames.tasks)
    tasks = tasks[tasks["is_standard_task"]].copy()
    criteria = _dedup_criteria(frames.criteria)
    rows = [
        _leaderboard_row(group, criteria)
        for _, group in tasks.groupby(["run_id", "model", "series_label"], sort=False)
    ]
    leaderboard = pd.DataFrame(rows)
    if leaderboard.empty:
        raise ValueError("Nessuna run standard disponibile per la leaderboard.")

    for column in leaderboard.columns:
        if column not in {"model", "run_id", "series_label", "run_timestamp"}:
            leaderboard[column] = pd.to_numeric(leaderboard[column], errors="coerce")

    leaderboard = leaderboard.sort_values(
        ["reasoning_all_pass_rate", "criterion_pass_rate", "mean_citation_score"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    if out_path:
        leaderboard.to_csv(out_path, index=False)
        log.info("Leaderboard salvata: %s", out_path)

    print("\n=== LEADERBOARD ===")
    print(_format_percentages(leaderboard).to_string(index=False))
    return leaderboard


def _area_metrics_row(
        tasks: pd.DataFrame,
        criteria: pd.DataFrame,
        *,
        run_id: str,
        model: str,
        series_label: str,
        macro_area: str,
) -> dict[str, Any]:
    standard = tasks[
        (tasks["run_id"] == run_id)
        & (tasks["model"] == model)
        & (tasks["macro_area"] == macro_area)
        & (tasks["is_standard_task"])
    ].copy()
    reasoning_complete = standard[standard["reasoning_scoring_status"] == "complete"]
    run_criteria = criteria[
        (criteria["run_id"] == run_id)
        & (criteria["model"] == model)
        & (criteria["macro_area"] == macro_area)
        & (criteria["is_standard_task"])
        & (criteria["reasoning_scoring_status"] == "complete")
    ].copy()
    resolved = run_criteria[run_criteria["verdict"].isin(["pass", "fail"])]
    citation_applicable = standard[standard["citation_applicable"]]
    citation_complete = citation_applicable[
        citation_applicable["citation_scoring_status"] == "complete"
    ]

    return {
        "series_label": series_label,
        "model": model,
        "run_id": run_id,
        "macro_area": macro_area,
        "n_standard_tasks": len(standard),
        "reasoning_all_pass_rate": _safe_rate(
            int((reasoning_complete["reasoning_all_pass"] == True).sum()),  # noqa: E712
            len(reasoning_complete),
        ),
        "criterion_pass_rate": _safe_rate(
            int((resolved["verdict"] == "pass").sum()),
            len(resolved),
        ),
        "mean_citation_score": _mean_or_none(citation_complete["citation_score"]),
        "mean_citation_coverage": _mean_or_none(citation_complete["citation_coverage_score"]),
        "mean_citation_relevance": _mean_or_none(citation_complete["citation_relevance_score"]),
        "citation_scoring_coverage": _safe_rate(
            len(citation_complete),
            len(citation_applicable),
        ),
    }


def metrics_by_area_table(
        frames: ScoreFrames,
        out_path: Path | None = None,
) -> pd.DataFrame:
    tasks = _dedup_tasks(frames.tasks)
    criteria = _dedup_criteria(frames.criteria)
    keys = (
        tasks[tasks["is_standard_task"]][["run_id", "model", "series_label", "macro_area"]]
        .drop_duplicates()
        .sort_values(["series_label", "macro_area"])
    )
    rows = [
        _area_metrics_row(
            tasks,
            criteria,
            run_id=str(row.run_id),
            model=str(row.model),
            series_label=str(row.series_label),
            macro_area=str(row.macro_area),
        )
        for row in keys.itertuples(index=False)
    ]
    area_metrics = pd.DataFrame(rows)

    if out_path:
        area_metrics.to_csv(out_path, index=False)
        log.info("Metriche per macro-area salvate: %s", out_path)

    return area_metrics


def false_premise_leaderboard(
        frames: ScoreFrames,
        out_path: Path | None = None,
) -> pd.DataFrame:
    tasks = _dedup_tasks(frames.tasks)
    bullshit = tasks[tasks["is_bullshit_task"]].copy()
    if bullshit.empty:
        log.info("Nessun task Bullshit presente: output false-premise saltati.")
        return pd.DataFrame()

    rows = []
    for _, group in bullshit.groupby(["run_id", "model", "series_label"], sort=False):
        complete = group[group["scoring_status"].fillna("complete") == "complete"]
        pass_mask = (complete["verdict"] == "pass") | (complete["score"] == 1.0)
        unresolved_mask = (group["verdict"] == "unresolved") | (
            group["scoring_status"] == "incomplete"
        )
        rows.append({
            "model": group["model"].iloc[0],
            "run_id": group["run_id"].iloc[0],
            "series_label": group["series_label"].iloc[0],
            "n_bullshit_tasks": len(group),
            "n_bullshit_complete": len(complete),
            "false_premise_detection_rate": _safe_rate(int(pass_mask.sum()), len(complete)),
            "bullshit_unresolved_rate": _safe_rate(int(unresolved_mask.sum()), len(group)),
        })

    leaderboard = pd.DataFrame(rows).sort_values(
        ["false_premise_detection_rate"],
        ascending=False,
        na_position="last",
    )
    if out_path:
        leaderboard.to_csv(out_path, index=False)
        log.info("False-premise leaderboard salvata: %s", out_path)
    return leaderboard


# ---------------------------------------------------------------------------
# Formattazione e grafici
# ---------------------------------------------------------------------------

def _format_percentages(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    for column in PERCENT_COLUMNS.intersection(formatted.columns):
        formatted[column] = formatted[column].map(
            lambda value: "n.d." if pd.isna(value) else f"{float(value):.1%}"
        )
    return formatted


def _save_no_data_figure(out_path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    ax.set_title(title, pad=12)
    ax.text(0.5, 0.5, message, ha="center", va="center")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info("Grafico senza dati salvato: %s", out_path)


def _bar_width(n_series: int) -> float:
    return min(0.8, 0.9 / max(n_series, 1))


def _plot_grouped_metric_bars(
        data: pd.DataFrame,
        *,
        index_col: str,
        metric_labels: dict[str, str],
        title: str,
        out_path: Path,
        note: str | None = None,
        xlabel: str = "",
) -> None:
    metrics = list(metric_labels.keys())
    plot_df = data[[index_col, *metrics]].copy()
    if plot_df[metrics].dropna(how="all").empty:
        _save_no_data_figure(out_path, title, "Dati non disponibili.")
        return

    labels = plot_df[index_col].astype(str).tolist()
    x = range(len(labels))
    width = _bar_width(len(metrics))
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.8), 5.4))

    for i, metric in enumerate(metrics):
        vals = pd.to_numeric(plot_df[metric], errors="coerce")
        positions = [xi + (i - len(metrics) / 2 + 0.5) * width for xi in x]
        ax.bar(
            positions,
            vals,
            width=width * 0.9,
            label=metric_labels[metric],
            color=PALETTE[i % len(PALETTE)],
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.set_title(title, pad=20 if note else 12)
    if note:
        ax.text(
            0.0,
            1.02,
            note,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            color="#555555",
        )
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info("Grafico salvato: %s", out_path)


def overall_metrics_chart(leaderboard: pd.DataFrame, out_path: Path) -> None:
    _plot_grouped_metric_bars(
        leaderboard,
        index_col="series_label",
        metric_labels={
            "reasoning_all_pass_rate": "Reasoning all-pass rate",
            "criterion_pass_rate": "Criterion pass rate",
            "mean_citation_score": "Citation grounding score",
        },
        title="LegalITA — performance complessiva",
        note="Reasoning e citation grounding sono metriche indipendenti.",
        out_path=out_path,
    )


def citation_grounding_components_chart(leaderboard: pd.DataFrame, out_path: Path) -> None:
    _plot_grouped_metric_bars(
        leaderboard,
        index_col="series_label",
        metric_labels={
            "mean_citation_coverage": "Mean citation coverage",
            "mean_citation_relevance": "Mean citation relevance",
            "mean_citation_score": "Mean citation score",
        },
        title="Citation grounding: coverage, relevance e score",
        note="Lo score mostrato è la media dei citation_score task-level salvati.",
        out_path=out_path,
    )


def _citation_status(task: pd.Series) -> str:
    status = task.get("citation_scoring_status")
    verdict = task.get("citation_verdict")
    if status == "complete":
        return "complete"
    if status == "not_cited" or verdict == "nc":
        return "nc"
    if status == "incomplete" or verdict == "unresolved":
        return "unresolved"
    return "unknown"


def citation_status_distribution(frames: ScoreFrames, out_path: Path) -> pd.DataFrame:
    tasks = _dedup_tasks(frames.tasks)
    applicable = tasks[
        (tasks["is_standard_task"])
        & (tasks["citation_applicable"])
    ].copy()
    if applicable.empty:
        _save_no_data_figure(out_path, "Citation status distribution", "Dati non disponibili.")
        return pd.DataFrame()

    applicable["status_bucket"] = applicable.apply(_citation_status, axis=1)
    unknown = applicable[applicable["status_bucket"] == "unknown"]
    if not unknown.empty:
        log.warning(
            "Citation status sconosciuti: %d task (%s)",
            len(unknown),
            sorted(unknown["run_id"].unique()),
        )

    counts = (
        applicable
        .groupby(["series_label", "status_bucket"])
        .size()
        .unstack(fill_value=0)
    )
    for status in ["complete", "nc", "unresolved", "unknown"]:
        if status not in counts.columns:
            counts[status] = 0
    counts = counts[["complete", "nc", "unresolved", "unknown"]]
    pct = counts.div(counts.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(max(8, len(pct) * 1.6), 5))
    positions = list(range(len(pct.index)))
    bottom = pd.Series([0.0] * len(pct), index=pct.index)
    for status in pct.columns:
        vals = pct[status]
        if status == "unknown" and vals.sum() == 0:
            continue
        ax.bar(
            positions,
            vals,
            bottom=bottom,
            label=status,
            color=STATUS_COLORS[status],
        )
        bottom = bottom + vals

    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title("Citation status distribution")
    ax.set_xticks(positions)
    ax.set_xticklabels(pct.index.astype(str), rotation=25, ha="right")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info("Citation status distribution salvata: %s", out_path)
    return pct.reset_index()


def _plot_area_metric(
        area_metrics: pd.DataFrame,
        *,
        metric: str,
        title: str,
        out_path: Path,
) -> None:
    if area_metrics.empty or area_metrics[metric].dropna().empty:
        _save_no_data_figure(out_path, title, "Dati non disponibili.")
        return

    areas = sorted(area_metrics["macro_area"].dropna().unique())
    series = area_metrics["series_label"].dropna().unique().tolist()
    x = range(len(areas))
    width = _bar_width(len(series))
    fig, ax = plt.subplots(figsize=(max(10, len(areas) * 1.5), 5.5))

    for i, label in enumerate(series):
        subset = area_metrics[area_metrics["series_label"] == label].set_index("macro_area")
        vals = [subset.loc[a, metric] if a in subset.index else pd.NA for a in areas]
        vals = pd.to_numeric(pd.Series(vals), errors="coerce")
        positions = [xi + (i - len(series) / 2 + 0.5) * width for xi in x]
        ax.bar(
            positions,
            vals,
            width=width * 0.9,
            label=label,
            color=PALETTE[i % len(PALETTE)],
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(areas, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title(title)
    ax.legend(title="Run", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info("Grafico per macro-area salvato: %s", out_path)


def reasoning_allpass_by_area(area_metrics: pd.DataFrame, out_path: Path) -> None:
    _plot_area_metric(
        area_metrics,
        metric="reasoning_all_pass_rate",
        title="Reasoning all-pass rate per macro-area",
        out_path=out_path,
    )


def citation_score_by_area(area_metrics: pd.DataFrame, out_path: Path) -> None:
    _plot_area_metric(
        area_metrics,
        metric="mean_citation_score",
        title="Citation grounding score per macro-area",
        out_path=out_path,
    )


def criterion_passrate_by_area(area_metrics: pd.DataFrame, out_path: Path) -> None:
    _plot_area_metric(
        area_metrics,
        metric="criterion_pass_rate",
        title="Criterion pass rate per macro-area",
        out_path=out_path,
    )


def criterion_heatmap(frames: ScoreFrames, out_path: Path | None = None) -> None:
    """
    Heatmap: asse X = modello/run, asse Y = macro_area + criterio, colore = pass rate.
    Usa solo criteri pass/fail dei task standard.
    """
    criteria = _dedup_criteria(frames.criteria)
    if criteria.empty:
        if out_path:
            _save_no_data_figure(out_path, "Pass rate per criterio e modello", "Nessun criterio.")
            return
        raise ValueError("Nessun criterio disponibile per la heatmap.")

    criteria = criteria[
        (criteria["is_standard_task"])
        & (criteria["verdict"].isin(["pass", "fail"]))
    ].copy()
    if criteria.empty:
        if out_path:
            _save_no_data_figure(
                out_path,
                "Pass rate per criterio e modello",
                "Nessun criterio pass/fail.",
            )
            return
        raise ValueError("Nessun criterio standard pass/fail disponibile per la heatmap.")

    criteria["criterion_key"] = (
        criteria["macro_area"].astype(str)
        + " | "
        + criteria["criterion_id"].astype(str)
        + " | "
        + criteria["criterion_title"].fillna("").astype(str)
    )
    pivot = (
        criteria.assign(passed=criteria["verdict"] == "pass")
        .groupby(["criterion_key", "series_label"])["passed"]
        .mean()
        .unstack("series_label")
    )

    fig, ax = plt.subplots(
        figsize=(max(8, len(pivot.columns) * 1.7), max(6, len(pivot) * 0.35)),
    )
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        linewidths=0.3,
        annot=len(pivot) <= 30,
        fmt=".0%",
        cbar_kws={"label": "Pass rate"},
    )
    ax.set_title("Pass rate per criterio e modello/run", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        log.info("Heatmap salvata: %s", out_path)
    else:
        plt.show()
    plt.close(fig)


def false_premise_detection_chart(leaderboard: pd.DataFrame, out_path: Path) -> None:
    if leaderboard.empty:
        _save_no_data_figure(out_path, "False-premise detection", "Nessun task Bullshit.")
        return
    _plot_grouped_metric_bars(
        leaderboard,
        index_col="series_label",
        metric_labels={
            "false_premise_detection_rate": "False-premise detection rate",
            "bullshit_unresolved_rate": "Bullshit unresolved rate",
        },
        title="False-premise detection",
        out_path=out_path,
    )


# ---------------------------------------------------------------------------
# Log diagnostici
# ---------------------------------------------------------------------------

def log_run_summary(frames: ScoreFrames) -> None:
    tasks = _dedup_tasks(frames.tasks)
    log.info("Run incluse: %s", sorted(tasks["series_label"].unique()))
    for label, group in tasks.groupby("series_label", sort=True):
        standard = group[group["is_standard_task"]]
        bullshit = group[group["is_bullshit_task"]]
        citation_applicable = standard[standard["citation_applicable"]]
        citation_complete = int((citation_applicable["citation_scoring_status"] == "complete").sum())
        citation_nc = int((citation_applicable["citation_verdict"] == "nc").sum())
        citation_unresolved = int((citation_applicable["citation_verdict"] == "unresolved").sum())
        log.info(
            "%s: task=%d, standard=%d, bullshit=%d, citation complete/NC/unresolved=%d/%d/%d",
            label,
            len(group),
            len(standard),
            len(bullshit),
            citation_complete,
            citation_nc,
            citation_unresolved,
        )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def run_all(
        results_dir: Path,
        out_dir: Path,
        models: list[str] | None = None,
        since: str | None = None,
        latest_only: bool = False,
        areas: list[str] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = load_score_frames(
        results_dir,
        models=models,
        since=since,
        latest_only=latest_only,
        areas=areas,
    )

    log_run_summary(frames)

    log.info("Modelli trovati: %s", sorted(frames.tasks["model"].unique()))
    log.info("Task totali: %d", frames.tasks["task_id"].nunique())
    log.info("Macro-aree: %s", sorted(frames.tasks["macro_area"].unique()))

    leaderboard = leaderboard_table(frames, out_dir / "leaderboard.csv")
    area_metrics = metrics_by_area_table(frames, out_dir / "metrics_by_area.csv")

    overall_metrics_chart(leaderboard, out_dir / "overall_metrics.png")
    citation_grounding_components_chart(
        leaderboard,
        out_dir / "citation_grounding_components.png",
    )
    citation_status_distribution(frames, out_dir / "citation_status_distribution.png")
    reasoning_allpass_by_area(area_metrics, out_dir / "reasoning_allpass_by_area.png")
    citation_score_by_area(area_metrics, out_dir / "citation_score_by_area.png")
    criterion_passrate_by_area(area_metrics, out_dir / "criterion_passrate_by_area.png")
    criterion_heatmap(frames, out_dir / "criterion_heatmap.png")

    # Alias legacy utili per script o notebook già esistenti.
    criterion_heatmap(frames, out_dir / "heatmap.png")
    reasoning_allpass_by_area(area_metrics, out_dir / "allpass_by_area.png")
    criterion_passrate_by_area(area_metrics, out_dir / "criterion_passrate.png")

    bullshit_leaderboard = false_premise_leaderboard(
        frames,
        out_dir / "false_premise_leaderboard.csv",
    )
    if not bullshit_leaderboard.empty:
        false_premise_detection_chart(
            bullshit_leaderboard,
            out_dir / "false_premise_detection.png",
        )

    log.info("\nOutput salvati in: %s", out_dir)


if __name__ == "__main__":
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(
        description="Genera i chart del benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Esempi:\n"
            "  python charts.py                          # tutti i risultati storici\n"
            "  python charts.py --latest                 # ultima run per modello/famiglia\n"
            "  python charts.py --models gpt-4o claude-sonnet-4-6\n"
            "  python charts.py --since 20260520         # dal 20 maggio 2026 in poi\n"
            "  python charts.py --areas diritto_civile diritto_tributario\n"
            "  python charts.py --out results/charts/mio-confronto\n"
        ),
    )
    parser.add_argument("--results", type=Path, default=RESULTS_DIR,
                        help="Directory radice dei risultati (default: results/)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Directory di output. Default: results/charts/<timestamp>/")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Filtra ai soli modelli indicati (per nome cartella).")
    parser.add_argument("--since", type=str, default=None,
                        help="Includi solo run con timestamp >= SINCE "
                             "(formato 'YYYYMMDD' o 'YYYYMMDD-HHMMSS').")
    parser.add_argument("--latest", action="store_true",
                        help="Per ogni modello/famiglia include solo la run più recente.")
    parser.add_argument("--areas", nargs="+", default=None,
                        help="Filtra una o più macro-aree normalizzate.")
    args = parser.parse_args()

    if args.out is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        args.out = args.results / "charts" / ts

    run_all(
        results_dir=args.results,
        out_dir=args.out,
        models=args.models,
        since=args.since,
        latest_only=args.latest,
        areas=args.areas,
    )
