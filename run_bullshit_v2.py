"""
Entrypoint provider-agnostic per il modulo adversarial Bullshit v2.

Esempi:
    python run_bullshit_v2.py --models claude-sonnet-4-6
    python run_bullshit_v2.py --models claude-sonnet-4-6 gpt-4o gemini-2.5-pro
    python run_bullshit_v2.py --models claude-sonnet-4-6 --generate-only
    python run_bullshit_v2.py --score-outputs results/bullshit/<model>/<run>/outputs.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import openai
from dotenv import load_dotenv

from config import (
    BULLSHIT_GOLD_PATH,
    JUDGE_MODEL,
    MODEL_MAX_TOKENS,
    MODEL_RETRIES,
    RANDOM_SEED,
    RESULTS_DIR,
)
from evaluation.bullshit_judge import (
    BullshitJudge,
    BullshitScore,
    BullshitTask,
    create_bullshit_judge_from_config,
    load_bullshit_tasks,
    score_bullshit_batch,
    summarize_bullshit_scores,
)
from model_query import (
    GEMINI_BASE_URL,
    GEMINI_PROVIDER_PREFIX,
    NOVITA_BASE_URL,
    NOVITA_GLM_52_MAX_TOKENS,
    NOVITA_PROVIDERS,
    default_anthropic_message_kwargs as _anthropic_message_kwargs,
    default_gemini_completion_kwargs as _gemini_completion_kwargs,
    default_novita_completion_kwargs as _novita_completion_kwargs,
    default_openai_completion_kwargs as _openai_completion_kwargs,
    model_request_kwargs_for_summary as _model_request_kwargs_for_summary,
    query_anthropic as _query_anthropic,
    query_anthropic_with_metrics as _query_anthropic_with_metrics,
    query_gemini as _query_gemini,
    query_gemini_with_metrics as _query_gemini_with_metrics,
    query_novita as _query_novita,
    query_novita_with_metrics as _query_novita_with_metrics,
    query_openai as _query_openai,
    query_openai_with_metrics as _query_openai_with_metrics,
    require_answer_text,
    require_result_text,
    run_query_with_retries,
)
from model_request_config import request_config_for_summary
from model_runtime import (
    ANTHROPIC_STREAMING_REQUIRED_ERROR,
    anthropic_response_text as _anthropic_response_text,
)
from usage_tracking import (
    ModelCallResult,
    score_model_dump,
    with_model_call_metrics,
)


DEFAULT_MODEL = "claude-sonnet-4-6"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

QueryFn = Callable[[str, str], str | ModelCallResult | None]


def model_slug(model: str) -> str:
    """Rende il nome modello sicuro per l'uso come cartella."""
    return model.replace("/", "-").replace(":", "-")


def select_tasks(
    tasks: list[BullshitTask],
    area: str | None = None,
    limit: int | None = None,
    seed: int = RANDOM_SEED,
) -> list[BullshitTask]:
    """Filtra o campiona i task per smoke test e run mirate."""
    selected = tasks

    if area:
        selected = [
            task
            for task in selected
            if task.macro_area.lower() == area.lower()
            or task.task_id.split("/")[0].lower() == area.lower()
        ]
        if not selected:
            raise ValueError(f"Nessun task trovato per l'area: {area}")

    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit deve essere maggiore di zero.")
        if limit < len(selected):
            rng = random.Random(seed)
            selected = sorted(rng.sample(selected, limit), key=lambda task: task.task_id)

    return selected


def query_model(model: str, query: str, max_retries: int = MODEL_RETRIES) -> str | None:
    """
    Interroga un modello supportato.

    Gli adapter provider sono condivisi con il benchmark standard tramite
    model_query; il routing per prefisso resta locale al modulo, cosi' i test
    possono sostituire i singoli adapter via monkeypatch.
    """
    if model.startswith(NOVITA_PROVIDERS):
        adapter = _query_novita
    elif model.startswith(GEMINI_PROVIDER_PREFIX):
        adapter = _query_gemini
    elif model.startswith("claude"):
        adapter = _query_anthropic
    elif any(model.startswith(prefix) for prefix in ("gpt", "o1", "o3", "o4")):
        adapter = _query_openai
    else:
        raise ValueError(f"Modello non supportato: {model}")

    answer = run_query_with_retries(
        adapter,
        model,
        query,
        max_retries=max_retries,
        log=log,
        validate=require_answer_text,
    )
    return answer.strip() if answer is not None else None


def query_model_with_metrics(
    model: str,
    query: str,
    max_retries: int = MODEL_RETRIES,
) -> ModelCallResult | None:
    if model.startswith(NOVITA_PROVIDERS):
        adapter = _query_novita_with_metrics
    elif model.startswith(GEMINI_PROVIDER_PREFIX):
        adapter = _query_gemini_with_metrics
    elif model.startswith("claude"):
        adapter = _query_anthropic_with_metrics
    elif any(model.startswith(prefix) for prefix in ("gpt", "o1", "o3", "o4")):
        adapter = _query_openai_with_metrics
    else:
        raise ValueError(f"Modello non supportato: {model}")

    return run_query_with_retries(
        adapter,
        model,
        query,
        max_retries=max_retries,
        log=log,
        validate=require_result_text,
    )


def create_run_dir(model: str, results_dir: Path = RESULTS_DIR) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = results_dir / "bullshit" / model_slug(model) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def generate_outputs(
    tasks: list[BullshitTask],
    model: str,
    delay_between: float,
    query_fn: QueryFn = query_model_with_metrics,
) -> dict[str, str]:
    """Interroga il modello sui task selezionati e restituisce {task_id: risposta}."""
    outputs, _ = generate_outputs_with_metrics(tasks, model, delay_between, query_fn)
    return outputs


def generate_outputs_with_metrics(
    tasks: list[BullshitTask],
    model: str,
    delay_between: float,
    query_fn: QueryFn = query_model_with_metrics,
) -> tuple[dict[str, str], dict[str, dict[str, object]]]:
    """Interroga il modello e restituisce output e metriche per task."""
    outputs: dict[str, str] = {}
    metrics_by_task: dict[str, dict[str, object]] = {}

    for index, task in enumerate(tasks, start=1):
        log.info("Query %d/%d [%s]: %s", index, len(tasks), model, task.task_id)
        result = query_fn(model, task.query)

        if result is None:
            log.error("%s [%s]: nessuna risposta salvata.", task.task_id, model)
        else:
            if isinstance(result, ModelCallResult):
                answer = result.text
                metrics_by_task[task.task_id] = result.metrics
            else:
                answer = result
            outputs[task.task_id] = answer

        if delay_between > 0 and index < len(tasks):
            time.sleep(delay_between)

    return outputs, metrics_by_task


def output_records(
    tasks: list[BullshitTask],
    outputs: dict[str, str],
    model: str,
) -> list[dict]:
    """Costruisce records serializzabili preservando la domanda originale del gold."""
    return [
        {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "model": model,
            "macro_area": task.macro_area,
            "difficulty": task.difficulty,
            "query": task.query,
            "response": outputs.get(task.task_id),
        }
        for task in tasks
    ]


def save_outputs(
    out_dir: Path,
    tasks: list[BullshitTask],
    outputs: dict[str, str],
    model: str,
) -> Path:
    """Salva le risposte grezze prima di qualunque valutazione."""
    out_path = out_dir / "outputs.json"
    out_path.write_text(
        json.dumps(output_records(tasks, outputs, model), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("Outputs grezzi salvati in: %s", out_path)
    return out_path


def load_outputs_for_scoring(path: Path) -> tuple[str, dict[str, str], list[str]]:
    """Carica un outputs.json precedentemente prodotto dal modulo bullshit."""
    if not path.exists():
        raise FileNotFoundError(f"Outputs non trovati: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("outputs.json vuoto o non valido.")

    model = str(data[0].get("model", DEFAULT_MODEL))
    outputs: dict[str, str] = {}
    task_ids: list[str] = []

    for item in data:
        task_id = str(item.get("task_id", "")).strip()
        response = item.get("response")
        if not task_id:
            continue
        task_ids.append(task_id)
        if isinstance(response, str) and response.strip():
            outputs[task_id] = response.strip()

    return model, outputs, task_ids


def save_scores_and_summary(
    out_dir: Path,
    scores: list[BullshitScore],
    summary: dict,
    *,
    include_model_call_fields: bool = True,
) -> tuple[Path, Path]:
    scores_path = out_dir / "scores.json"
    scores_path.write_text(
        json.dumps(
            [
                score_model_dump(
                    score,
                    include_model_call_fields=include_model_call_fields,
                )
                for score in scores
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return scores_path, summary_path


def print_summary(
    summary: dict,
    scores_path: Path,
    summary_path: Path,
    evaluated_model: str,
    judge_model: str,
) -> None:
    overall = summary["overall"]
    diagnostics = summary["diagnostics"]
    print()
    print("=== RISULTATI BULLSHIT V2 ===")
    print(f"Modello valutato:        {evaluated_model}")
    print(f"Judge:                   {judge_model}")
    if summary.get("judge_strategy"):
        print(f"Strategia judge:         {summary['judge_strategy']}")
    if summary.get("judge_models"):
        print(f"Judge models:            {summary['judge_models']}")
    print(f"Task valutati:           {overall['n_tasks']}")
    if overall.get("n_incomplete"):
        print(f"Task incompleti:         {overall['n_incomplete']}")
    print(
        f"Pass pieni:              {overall['pass']}/{overall.get('n_complete', overall['n_tasks'])} "
        f"({overall['all_pass_rate']:.1%})"
    )
    print(f"Fail:                    {overall['fail']}")
    print(f"All-pass rate:           {overall['all_pass_rate']:.1%}")
    print(
        f"Criteri passati:         {diagnostics['n_passed_criteria']}/"
        f"{diagnostics['n_criteria']} ({diagnostics['criterion_pass_rate']:.1%})"
    )
    if diagnostics.get("n_unresolved_criteria"):
        print(
            f"Criteri unresolved:      {diagnostics['n_unresolved_criteria']} "
            f"({diagnostics['unresolved_rate']:.1%})"
        )
    print(f"Scores salvati in:       {scores_path}")
    print(f"Summary salvato in:      {summary_path}")


def score_and_save(
    tasks: list[BullshitTask],
    outputs: dict[str, str],
    evaluated_model: str,
    judge_model: str,
    out_dir: Path,
    judge: BullshitJudge | None = None,
    judge_strategy: str | None = None,
    judge_a_provider: str | None = None,
    judge_a_model: str | None = None,
    judge_b_provider: str | None = None,
    judge_b_model: str | None = None,
    judge_c_provider: str | None = None,
    judge_c_model: str | None = None,
    model_call_metrics: dict[str, dict[str, object]] | None = None,
    summary_extra: dict | None = None,
) -> tuple[Path, Path]:
    """Valuta output gia raccolti con il judge bullshit e salva score/summary."""
    judge = judge or create_bullshit_judge_from_config(
        judge_strategy=judge_strategy,
        judge_a_provider=judge_a_provider,
        judge_a_model=judge_a_model,
        judge_b_provider=judge_b_provider,
        judge_b_model=judge_b_model,
        judge_c_provider=judge_c_provider,
        judge_c_model=judge_c_model,
        legacy_judge_model=judge_model,
    )
    scores = score_bullshit_batch(
        tasks=tasks,
        outputs=outputs,
        model=evaluated_model,
        judge=judge,
    )
    has_model_call_metrics = bool(model_call_metrics)
    scores = [
        with_model_call_metrics(
            score,
            (model_call_metrics or {}).get(score.task_id),
        )
        for score in scores
    ]
    summary = summarize_bullshit_scores(scores)
    summary["evaluated_model"] = evaluated_model
    summary["judge_model"] = judge.model
    summary["judge_strategy"] = getattr(judge, "strategy", "single")
    summary["judge_models"] = getattr(
        judge,
        "judge_models",
        {"A": getattr(judge, "model", judge_model)},
    )
    summary["scored_at"] = datetime.now(timezone.utc).isoformat()
    if summary_extra:
        summary.update(summary_extra)
    if not has_model_call_metrics:
        summary.pop("model_call", None)
    else:
        summary["model_request_config"] = request_config_for_summary(
            _model_request_kwargs_for_summary(evaluated_model)
        )

    scores_path, summary_path = save_scores_and_summary(
        out_dir,
        scores,
        summary,
        include_model_call_fields=has_model_call_metrics,
    )
    print_summary(summary, scores_path, summary_path, evaluated_model, judge.model)
    return scores_path, summary_path


def tasks_from_output_ids(all_tasks: list[BullshitTask], task_ids: list[str]) -> list[BullshitTask]:
    task_id_set = set(task_ids)
    tasks = [task for task in all_tasks if task.task_id in task_id_set]
    missing_gold = sorted(task_id_set - {task.task_id for task in all_tasks})
    if missing_gold:
        raise ValueError(f"Task negli outputs non presenti nel gold: {missing_gold}")
    if not tasks:
        raise ValueError("Nessun task valutabile trovato negli outputs.")
    return tasks


def run_models(
    models: list[str],
    tasks: list[BullshitTask],
    judge_model: str,
    delay_between: float,
    generate_only: bool,
    query_fn: QueryFn = query_model,
    results_dir: Path = RESULTS_DIR,
    judge_factory: Callable[[str], BullshitJudge] | None = None,
    judge_strategy: str | None = None,
    judge_a_provider: str | None = None,
    judge_a_model: str | None = None,
    judge_b_provider: str | None = None,
    judge_b_model: str | None = None,
    judge_c_provider: str | None = None,
    judge_c_model: str | None = None,
) -> list[Path]:
    """Esegue una run separata per ciascun modello e restituisce le directory create."""
    run_dirs: list[Path] = []

    for model in models:
        out_dir = create_run_dir(model, results_dir=results_dir)
        run_dirs.append(out_dir)
        outputs, model_call_metrics = generate_outputs_with_metrics(
            tasks=tasks,
            model=model,
            delay_between=delay_between,
            query_fn=query_fn,
        )
        outputs_path = save_outputs(out_dir, tasks, outputs, model)

        print()
        print(f"Modello:                 {model}")
        print(f"Risposte raccolte:       {len(outputs)}/{len(tasks)}")
        print(f"Outputs salvati in:      {outputs_path}")

        if generate_only:
            print("Judge non eseguito (--generate-only).")
            continue

        judge = judge_factory(judge_model) if judge_factory else None
        score_and_save(
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
            model_call_metrics=model_call_metrics,
        )

    return run_dirs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Esegue modelli API e/o il bullshit judge sui 40 task missing-document v2."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Uno o piu modelli da interrogare, es. claude-sonnet-4-6 gpt-4o gemini-2.5-pro.",
    )
    parser.add_argument(
        "--judge",
        type=str,
        default=JUDGE_MODEL,
        help="Alias legacy per il modello del Judge A (default: JUDGE_MODEL da config.py).",
    )
    parser.add_argument(
        "--judge-strategy",
        choices=["single", "adaptive_majority"],
        default=None,
        help="Strategia judge: single oppure adaptive_majority.",
    )
    parser.add_argument("--judge-a-provider", choices=["anthropic", "openai"], default=None)
    parser.add_argument("--judge-a-model", default=None)
    parser.add_argument("--judge-b-provider", choices=["anthropic", "openai"], default=None)
    parser.add_argument("--judge-b-model", default=None)
    parser.add_argument("--judge-c-provider", choices=["anthropic", "openai"], default=None)
    parser.add_argument("--judge-c-model", default=None)
    parser.add_argument(
        "--gold",
        type=Path,
        default=BULLSHIT_GOLD_PATH,
        help="Percorso del gold privato bullshit corrente.",
    )
    parser.add_argument(
        "--area",
        type=str,
        default=None,
        help="Limita la run a una macro-area o al prefisso task_id.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Campiona N task per smoke test.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="Pausa tra query al modello (default: 0.3 secondi).",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Genera e salva outputs.json senza eseguire il judge.",
    )
    parser.add_argument(
        "--score-outputs",
        type=Path,
        default=None,
        help="Valuta un outputs.json gia salvato senza interrogare modelli.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)

    all_tasks = load_bullshit_tasks(args.gold)

    if args.score_outputs:
        evaluated_model, outputs, task_ids = load_outputs_for_scoring(args.score_outputs)
        tasks = tasks_from_output_ids(all_tasks, task_ids)
        score_and_save(
            tasks=tasks,
            outputs=outputs,
            evaluated_model=evaluated_model,
            judge_model=args.judge,
            out_dir=args.score_outputs.parent,
            judge_strategy=args.judge_strategy,
            judge_a_provider=args.judge_a_provider,
            judge_a_model=args.judge_a_model,
            judge_b_provider=args.judge_b_provider,
            judge_b_model=args.judge_b_model,
            judge_c_provider=args.judge_c_provider,
            judge_c_model=args.judge_c_model,
        )
        return

    if not args.models:
        parser.error("--models e richiesto quando non usi --score-outputs.")

    tasks = select_tasks(all_tasks, area=args.area, limit=args.limit)
    log.info("Task bullshit selezionati: %d", len(tasks))

    run_models(
        models=args.models,
        tasks=tasks,
        judge_model=args.judge,
        delay_between=args.delay,
        generate_only=args.generate_only,
        judge_strategy=args.judge_strategy,
        judge_a_provider=args.judge_a_provider,
        judge_a_model=args.judge_a_model,
        judge_b_provider=args.judge_b_provider,
        judge_b_model=args.judge_b_model,
        judge_c_provider=args.judge_c_provider,
        judge_c_model=args.judge_c_model,
    )


if __name__ == "__main__":
    main()
