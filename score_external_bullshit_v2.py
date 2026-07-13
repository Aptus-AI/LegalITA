"""
Scoring bullshit v2 per risposte importate da CSV.

Esempi:
    python score_external_bullshit_v2.py --csv risposte.csv --model nextos
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from config import BULLSHIT_GOLD_PATH, JUDGE_MODEL, RESULTS_DIR
from evaluation.bullshit_judge import BullshitJudge, BullshitTask, load_bullshit_tasks
from run_bullshit_v2 import (
    create_run_dir,
    model_slug,
    save_outputs,
    score_and_save,
    select_tasks,
)


QUESTION_COLUMNS = (
    "Domanda bullshit",
    "Domanda",
    "Question",
    "question",
    "query",
    "Query",
)
ANSWER_COLUMNS = (
    "Risposte",
    "Risposta",
    "Answer",
    "answer",
    "response",
    "Response",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def normalize_question_for_matching(text: str) -> str:
    """Normalizza solo per associare CSV e gold; non altera i testi salvati."""
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def _cell_to_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def detect_column(columns: list[str], explicit: str | None, candidates: tuple[str, ...], kind: str) -> str:
    if explicit:
        if explicit not in columns:
            raise ValueError(f"Colonna {kind} non trovata: {explicit}. Colonne: {columns}")
        return explicit

    matches = [column for column in columns if column in candidates]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            f"Impossibile rilevare la colonna {kind}. Specifica --{kind}-column. "
            f"Colonne: {columns}"
        )
    raise ValueError(
        f"Colonna {kind} ambigua: {matches}. Specifica --{kind}-column."
    )


def build_gold_question_index(tasks: list[BullshitTask]) -> dict[str, BullshitTask]:
    grouped: dict[str, list[BullshitTask]] = defaultdict(list)
    for task in tasks:
        grouped[normalize_question_for_matching(task.query)].append(task)

    ambiguous = {
        key: [task.task_id for task in values]
        for key, values in grouped.items()
        if len(values) > 1
    }
    if ambiguous:
        raise ValueError(f"Domande gold ambigue dopo normalizzazione: {ambiguous}")

    return {key: values[0] for key, values in grouped.items()}


def load_external_outputs(
    csv_path: Path,
    tasks: list[BullshitTask],
    question_column: str | None = None,
    answer_column: str | None = None,
) -> dict[str, str]:
    """Carica un CSV e associa ogni risposta al task gold tramite domanda."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV non trovato: {csv_path.resolve()}")

    df = pd.read_csv(
        csv_path,
        encoding="utf-8-sig",
        sep=None,
        engine="python",
    )
    columns = list(df.columns)
    question_col = detect_column(columns, question_column, QUESTION_COLUMNS, "question")
    answer_col = detect_column(columns, answer_column, ANSWER_COLUMNS, "answer")

    gold_by_question = build_gold_question_index(tasks)
    outputs: dict[str, str] = {}
    seen_questions: dict[str, int] = {}

    for row_number, row in df.iterrows():
        raw_question = _cell_to_text(row[question_col])
        raw_answer = _cell_to_text(row[answer_col])
        if not raw_question and not raw_answer:
            continue
        if not raw_question:
            raise ValueError(f"Riga {row_number + 2}: domanda vuota.")
        if not raw_answer:
            raise ValueError(f"Riga {row_number + 2}: risposta vuota.")

        key = normalize_question_for_matching(raw_question)
        if key in seen_questions:
            raise ValueError(
                f"Domanda duplicata nel CSV alle righe {seen_questions[key]} e {row_number + 2}."
            )
        seen_questions[key] = row_number + 2

        task = gold_by_question.get(key)
        if task is None:
            raise ValueError(
                f"Riga {row_number + 2}: domanda non presente nel gold selezionato."
            )
        outputs[task.task_id] = raw_answer

    expected_ids = {task.task_id for task in tasks}
    found_ids = set(outputs)
    missing = sorted(expected_ids - found_ids)
    extra = sorted(found_ids - expected_ids)
    if missing or extra:
        raise ValueError(
            "CSV e gold selezionato non sono perfettamente allineati. "
            f"Mancanti: {missing}; extra: {extra}"
        )

    return outputs


def run_external_csv(
    csv_path: Path,
    model: str,
    gold_path: Path = BULLSHIT_GOLD_PATH,
    judge_model: str = JUDGE_MODEL,
    area: str | None = None,
    limit: int | None = None,
    judge_strategy: str | None = None,
    judge_a_provider: str | None = None,
    judge_a_model: str | None = None,
    judge_b_provider: str | None = None,
    judge_b_model: str | None = None,
    judge_c_provider: str | None = None,
    judge_c_model: str | None = None,
    question_column: str | None = None,
    answer_column: str | None = None,
    results_dir: Path = RESULTS_DIR,
    judge_factory: Callable[[str], BullshitJudge] | None = None,
) -> tuple[Path, Path, Path]:
    all_tasks = load_bullshit_tasks(gold_path)
    tasks = select_tasks(all_tasks, area=area, limit=limit)
    outputs = load_external_outputs(
        csv_path=csv_path,
        tasks=tasks,
        question_column=question_column,
        answer_column=answer_column,
    )

    out_dir = create_run_dir(model_slug(model), results_dir=results_dir)
    outputs_path = save_outputs(out_dir, tasks, outputs, model)
    judge = judge_factory(judge_model) if judge_factory else None
    scores_path, summary_path = score_and_save(
        tasks=tasks,
        outputs=outputs,
        evaluated_model=model,
        judge_model=judge_model,
        out_dir=out_dir,
        judge=judge,
        judge_strategy=judge_strategy,
        judge_a_provider=judge_a_provider,
        judge_a_model=judge_a_model,
        judge_b_provider=judge_b_provider,
        judge_b_model=judge_b_model,
        judge_c_provider=judge_c_provider,
        judge_c_model=judge_c_model,
        summary_extra={
            "source_csv": str(csv_path),
            "scored_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    print()
    print("=== IMPORT CSV BULLSHIT V2 ===")
    print(f"Modello valutato:        {model}")
    print(f"CSV sorgente:            {csv_path}")
    print(f"Outputs salvati in:      {outputs_path}")
    return outputs_path, scores_path, summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Valuta risposte bullshit v2 importate da CSV."
    )
    parser.add_argument("--csv", type=Path, required=True, help="CSV con domande e risposte.")
    parser.add_argument("--model", type=str, required=True, help="Nome modello/sistema valutato.")
    parser.add_argument("--judge", type=str, default=JUDGE_MODEL, help="Alias legacy per il modello del Judge A.")
    parser.add_argument("--judge-strategy", choices=["single", "adaptive_majority"], default=None)
    parser.add_argument("--judge-a-provider", choices=["anthropic", "openai"], default=None)
    parser.add_argument("--judge-a-model", default=None)
    parser.add_argument("--judge-b-provider", choices=["anthropic", "openai"], default=None)
    parser.add_argument("--judge-b-model", default=None)
    parser.add_argument("--judge-c-provider", choices=["anthropic", "openai"], default=None)
    parser.add_argument("--judge-c-model", default=None)
    parser.add_argument("--gold", type=Path, default=BULLSHIT_GOLD_PATH, help="Gold corrente bullshit v3.")
    parser.add_argument("--area", type=str, default=None, help="Filtra macro-area.")
    parser.add_argument("--limit", type=int, default=None, help="Campiona N task.")
    parser.add_argument("--question-column", type=str, default=None, help="Colonna domanda.")
    parser.add_argument("--answer-column", type=str, default=None, help="Colonna risposta.")
    return parser


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    run_external_csv(
        csv_path=args.csv,
        model=args.model,
        gold_path=args.gold,
        judge_model=args.judge,
        area=args.area,
        limit=args.limit,
        judge_strategy=args.judge_strategy,
        judge_a_provider=args.judge_a_provider,
        judge_a_model=args.judge_a_model,
        judge_b_provider=args.judge_b_provider,
        judge_b_model=args.judge_b_model,
        judge_c_provider=args.judge_c_provider,
        judge_c_model=args.judge_c_model,
        question_column=args.question_column,
        answer_column=args.answer_column,
    )


if __name__ == "__main__":
    main()
