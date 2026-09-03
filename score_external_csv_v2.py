"""
Valuta risposte già prodotte da un sistema esterno
sui task esistenti del benchmark, senza interrogare nuovamente un modello
sotto esame.

Uso:
    python score_external_csv_v2.py --csv "Benchmark v3 - NEXTOS.csv" --model-name NEXTOS
    python score_external_csv_v2.py --csv "Benchmark v3 - NEXTOS.csv" --model-name NEXTOS --judge claude-sonnet-4-6
"""

import argparse
import logging
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import JUDGE_MODEL, build_judge_runtime_config
from evaluation.judge import Judge, create_judge_from_config
from evaluation.scoring import score_batch, summarize_batch_scores
from run_benchmark import load_tasks, save_scores, save_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def normalize_query(text: str) -> str:
    """
    Normalizza le query esclusivamente ai fini dell'abbinamento CSV/task.json.

    Gestisce differenze tipografiche frequenti negli export:
    - trattini brevi/lunghi (es. "-" vs "–")
    - apostrofi e virgolette tipografiche
    - spazi non separabili e spazi multipli
    - differenze di maiuscole/minuscole

    Il testo originale della domanda e della risposta non viene modificato.
    """
    text = unicodedata.normalize("NFKC", str(text).strip())

    text = text.translate(str.maketrans({
        "\u00a0": " ",   # spazio non separabile
        "\u2010": "-",   # hyphen
        "\u2011": "-",   # non-breaking hyphen
        "\u2012": "-",   # figure dash
        "\u2013": "-",   # en dash
        "\u2014": "-",   # em dash
        "\u2212": "-",   # minus sign
        "\u2018": "'",   # apostrofo singolo sinistro
        "\u2019": "'",   # apostrofo singolo destro
        "\u201b": "'",   # apostrofo alto rovesciato
        "\u2032": "'",   # prime usato come apostrofo
        "\u201c": '"',   # virgolette doppie sinistre
        "\u201d": '"',   # virgolette doppie destre
    }))

    text = re.sub(r"\s+", " ", text)
    return text.casefold()


def clean_exported_answer(text: str) -> str:
    """
    Rimuove l'intestazione inserita dall'export Aptus/NEXTOS, se presente:
    'Assistente AI di Aptus - 25/05/2026, 09:29:05'
    """
    text = str(text).strip()
    text = re.sub(
        r"^\s*Assistente AI di Aptus\s*-\s*\d{2}/\d{2}/\d{4},\s*\d{2}:\d{2}:\d{2}\s*",
        "",
        text,
        count=1,
    )
    return text.strip()


def load_external_outputs(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    required = {"Domanda", "Risposte"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Colonne mancanti nel CSV: {sorted(missing)}. "
            f"Colonne trovate: {list(df.columns)}"
        )

    df = df.copy()
    df["query_key"] = df["Domanda"].map(normalize_query)
    df["answer_clean"] = df["Risposte"].map(clean_exported_answer)

    if df["query_key"].duplicated().any():
        duplicates = df.loc[df["query_key"].duplicated(keep=False), "Domanda"].tolist()
        raise ValueError(
            "Domande duplicate nel CSV, impossibile abbinare in modo univoco: "
            f"{duplicates}"
        )

    if (df["answer_clean"].str.len() == 0).any():
        raise ValueError("Almeno una risposta risulta vuota dopo la pulizia dell'export.")

    return df


def run_external_scoring(
    csv_path: Path,
    model_name: str,
    judge_model: str,
    area: str | None = None,
    judge_strategy: str | None = None,
    judge_a_provider: str | None = None,
    judge_a_model: str | None = None,
    judge_b_provider: str | None = None,
    judge_b_model: str | None = None,
    judge_c_provider: str | None = None,
    judge_c_model: str | None = None,
) -> Path:
    log.info("VERSIONE V2 ATTIVA — normalizzazione tipografica CSV/task abilitata")
    log.info("Il citation grounding viene eseguito separatamente con legalita-grounding.")
    rows = load_external_outputs(csv_path)
    all_tasks = load_tasks(area=area)

    tasks_by_query: dict[str, object] = {}
    duplicate_task_queries: list[str] = []

    for task in all_tasks:
        key = normalize_query(task.query)
        if key in tasks_by_query:
            duplicate_task_queries.append(task.query)
        tasks_by_query[key] = task

    if duplicate_task_queries:
        raise ValueError(
            "Esistono task con query duplicate; non è sicuro abbinare il CSV. "
            f"Esempio: {duplicate_task_queries[0][:120]}"
        )

    matched_tasks = []
    outputs: dict[str, str] = {}
    unmatched: list[str] = []

    for _, row in rows.iterrows():
        task = tasks_by_query.get(row["query_key"])
        if task is None:
            unmatched.append(str(row["Domanda"]))
            continue
        matched_tasks.append(task)
        outputs[task.task_id] = row["answer_clean"]

    if unmatched:
        preview = "\n".join(f"- {q[:200]}" for q in unmatched)
        raise ValueError(
            f"{len(unmatched)} domande del CSV non corrispondono ad alcun task.json.\n"
            "Non eseguo uno scoring parziale per evitare associazioni errate.\n"
            f"Domande non abbinate:\n{preview}"
        )

    judge_config = build_judge_runtime_config(
        judge_strategy=judge_strategy,
        judge_a_provider=judge_a_provider,
        judge_a_model=judge_a_model,
        judge_b_provider=judge_b_provider,
        judge_b_model=judge_b_model,
        judge_c_provider=judge_c_provider,
        judge_c_model=judge_c_model,
        legacy_judge_model=judge_model,
    )
    log.info(f"Risposte esterne abbinate: {len(matched_tasks)}/{len(rows)}")
    log.info(f"Modello valutato: {model_name}")
    log.info(
        "Judge strategy: %s | A=%s/%s | B=%s/%s | C=%s/%s",
        judge_config.strategy,
        judge_config.judge_a.provider,
        judge_config.judge_a.model,
        judge_config.judge_b.provider,
        judge_config.judge_b.model or "-",
        judge_config.judge_c.provider,
        judge_config.judge_c.model or "-",
    )
    if judge_config.strategy == "single" and judge_config.judge_a.provider == "anthropic":
        judge = Judge(model=judge_config.judge_a.model)
    else:
        judge = create_judge_from_config(judge_config)
    scores = score_batch(
        matched_tasks,
        outputs,
        model_name,
        judge,
        citation_service=None,
        citation_grounding_enabled=False,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    model_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model_name).strip("-") or "external-model"
    run_id = f"{model_slug}/{timestamp}"
    out_path = save_scores(scores, run_id, include_model_call_fields=False)

    summary = summarize_batch_scores(scores)
    summary.pop("model_call", None)
    summary["model"] = model_name
    summary["run_id"] = run_id
    summary["scored_at"] = datetime.now(timezone.utc).isoformat()
    summary_path = save_summary(summary, run_id)

    log.info("=" * 50)
    log.info(f"Task valutati:       {len(scores)}")
    log.info(
        "Task completi:       %d (%d incompleti)",
        summary["n_complete"],
        summary["n_incomplete"],
    )
    log.info(
        "Reasoning all-pass:  %d/%d (%.1f%%)",
        summary["n_allpass"],
        summary["n_complete"],
        summary["reasoning_allpass_rate"] * 100,
    )
    log.info(f"Criterion pass rate: {summary['required_criterion_pass_rate']:.1%}")
    log.info(f"Unresolved rate:     {summary['unresolved_rate']:.1%}")
    log.info(
        "Citation perfect:    %.1f%% (%d/%d scored)",
        summary["citation_perfect_rate"] * 100,
        summary["citation_perfect_tasks"],
        len([score for score in scores if score.citation_score is not None]),
    )
    if summary["citation_mean_score"] is not None:
        log.info(f"Citation mean score: {summary['citation_mean_score']:.1%}")
    if summary["citation_mean_coverage_score"] is not None:
        log.info(f"Citation coverage:   {summary['citation_mean_coverage_score']:.1%}")
    if summary["citation_mean_relevance_score"] is not None:
        log.info(f"Citation relevance:  {summary['citation_mean_relevance_score']:.1%}")
    log.info(f"Citation NC rate:    {summary['citation_nc_rate']:.1%}")
    log.info(f"Citation unresolved: {summary['citation_unresolved_rate']:.1%}")
    log.info("Citation existence unresolved: %d task", summary["citation_existence_unresolved_tasks"])
    log.info(f"Scores:              {out_path}")
    log.info(f"Summary:             {summary_path}")
    log.info("=" * 50)

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Valuta un CSV di risposte esterne sui task esistenti del benchmark."
    )
    parser.add_argument("--csv", type=Path, required=True, help="CSV con colonne Domanda e Risposte.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="NEXTOS",
        help="Etichetta del sistema valutato nei risultati (default: NEXTOS).",
    )
    parser.add_argument(
        "--judge",
        type=str,
        default=JUDGE_MODEL,
        help="Alias legacy per il modello del Judge A (default: valore in config.py).",
    )
    parser.add_argument("--judge-strategy", choices=["single", "adaptive_majority"], default=None)
    parser.add_argument("--judge-a-provider", choices=["anthropic", "openai"], default=None)
    parser.add_argument("--judge-a-model", default=None)
    parser.add_argument("--judge-b-provider", choices=["anthropic", "openai"], default=None)
    parser.add_argument("--judge-b-model", default=None)
    parser.add_argument("--judge-c-provider", choices=["anthropic", "openai"], default=None)
    parser.add_argument("--judge-c-model", default=None)
    parser.add_argument(
        "--area",
        type=str,
        default=None,
        help="Opzionale: carica solo i task di una macro-area.",
    )
    args = parser.parse_args()

    run_external_scoring(
        csv_path=args.csv,
        model_name=args.model_name,
        judge_model=args.judge,
        area=args.area,
        judge_strategy=args.judge_strategy,
        judge_a_provider=args.judge_a_provider,
        judge_a_model=args.judge_a_model,
        judge_b_provider=args.judge_b_provider,
        judge_b_model=args.judge_b_model,
        judge_c_provider=args.judge_c_provider,
        judge_c_model=args.judge_c_model,
    )
