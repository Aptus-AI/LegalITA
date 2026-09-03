"""
Entrypoint principale del benchmark legale italiano.

Esegue la valutazione di uno o piu modelli su task.json esistenti. Il filtro
per area e sempre applicato in memoria, perche le directory storiche possono
avere prefissi legacy mentre il campo macro_area viene canonizzato a runtime.
"""

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import openai
from dotenv import load_dotenv

from legal_ita.config import (
    JUDGE_MODEL,
    MODEL_MAX_TOKENS,
    MODEL_RETRIES,
    RANDOM_SEED,
    RESULTS_DIR,
    TASKS_DIR,
    build_judge_runtime_config,
    validate_judge_runtime_config,
)
from evaluation.judge import Judge, create_judge_from_config
from evaluation.scoring import score_batch, summarize_batch_scores
from legal_ita.modeling.query import (
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
    require_result_text,
    run_query_with_retries,
)
from legal_ita.modeling.request_config import request_config_for_summary
from legal_ita.modeling.runtime import (
    ANTHROPIC_STREAMING_REQUIRED_ERROR,
    anthropic_response_text as _anthropic_response_text,
)
from legal_ita.grounding.service import (
    records_from_results,
    require_bundle,
    resolve_bundle_paths,
    run_grounding,
)
from legal_ita.schemas import BenchmarkTask, TaskScore
from legal_ita.taxonomy import normalize_macro_area
from legal_ita.modeling.usage import (
    ModelCallResult,
    score_model_dump,
    with_model_call_metrics,
)


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_tasks(
    tasks_dir: Path = TASKS_DIR,
    area: str | None = None,
    limit: int | None = None,
    seed: int = RANDOM_SEED,
) -> list[BenchmarkTask]:
    """
    Carica task.json da disco.

    Se area e valorizzata, accetta alias legacy o slug canonici e filtra dopo
    la validazione Pydantic, non sul path.
    """
    paths = sorted(tasks_dir.glob("**/task.json"))
    if not paths:
        raise FileNotFoundError(
            f"Nessun task.json trovato in {tasks_dir}\n"
            "Esegui prima la pipeline di generazione."
        )

    requested_area = normalize_macro_area(area, strict=True) if area else None
    tasks: list[BenchmarkTask] = []

    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data.setdefault("source_ecli", "")
            task = BenchmarkTask.model_validate(data)
        except Exception as exc:
            log.warning("Task non caricato %s: %s", path, exc)
            continue

        if requested_area and task.macro_area != requested_area:
            continue
        tasks.append(task)

    if not tasks:
        raise FileNotFoundError(
            f"Nessun task caricabile trovato in {tasks_dir}"
            + (f" per area {area!r}" if area else "")
        )

    if limit and limit < len(tasks):
        import random

        rng = random.Random(seed)
        tasks = sorted(rng.sample(tasks, limit), key=lambda task: task.task_id)

    log.info(
        "Task caricati: %d%s",
        len(tasks),
        f" [{requested_area}]" if requested_area else "",
    )
    return tasks


def query_model(model: str, query: str, max_retries: int = MODEL_RETRIES) -> str | None:
    """
    Manda la query al modello e restituisce la risposta.

    Supporta modelli Anthropic, OpenAI, Gemini e modelli serviti via Novita.
    Il routing per prefisso resta locale al modulo, cosi' i test possono
    sostituire i singoli adapter via monkeypatch.
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

    return run_query_with_retries(adapter, model, query, max_retries=max_retries, log=log)


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


def save_scores(
    scores: list[TaskScore],
    run_id: str,
    *,
    include_model_call_fields: bool = True,
) -> Path:
    """Scrive scores.json nella cartella results/<run_id>/."""
    out_dir = RESULTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scores.json"

    data = [
        score_model_dump(score, include_model_call_fields=include_model_call_fields)
        for score in scores
    ]
    out_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("Scores scritti: %s", out_path)
    return out_path


def save_summary(summary: dict, run_id: str) -> Path:
    out_dir = RESULTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    out_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("Summary scritto: %s", out_path)
    return out_path


def run(
    models: list[str],
    area: str | None = None,
    limit: int | None = None,
    judge_model: str = JUDGE_MODEL,
    judge_strategy: str | None = None,
    judge_a_provider: str | None = None,
    judge_a_model: str | None = None,
    judge_b_provider: str | None = None,
    judge_b_model: str | None = None,
    judge_c_provider: str | None = None,
    judge_c_model: str | None = None,
    delay_between: float = 0.3,
    skip_citation_grounding: bool = False,
) -> None:
    """
    Esegue modello + judge e, salvo ``skip_citation_grounding``, il citation
    grounding offline sulla run appena scritta (stesso backend locale di
    ``legalita-grounding``). Il bundle registry + profili viene verificato
    prima di qualsiasi chiamata API.
    """
    grounding_paths: tuple[Path, Path] | None = None
    if not skip_citation_grounding:
        grounding_paths = resolve_bundle_paths()
        require_bundle(
            *grounding_paths,
            hint="Oppure esegui con --skip-citation-grounding per il solo scoring giuridico.",
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
    validate_judge_runtime_config(judge_config)

    tasks = load_tasks(area=area, limit=limit)
    if judge_config.strategy == "single" and judge_config.judge_a.provider == "anthropic":
        judge = Judge(model=judge_config.judge_a.model)
    else:
        judge = create_judge_from_config(judge_config)
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

    if skip_citation_grounding:
        log.info("Citation grounding disattivato: eseguibile a parte con legalita-grounding.")
    else:
        log.info("Citation grounding offline attivo: verra' eseguito al termine dello scoring.")

    for model in models:
        log.info("\n%s", "=" * 50)
        log.info("Modello: %s", model)
        log.info("%s", "=" * 50)

        outputs: dict[str, str] = {}
        model_call_metrics: dict[str, dict[str, object]] = {}
        n_failed = 0

        for task in tasks:
            result = query_model_with_metrics(model, task.query)
            if result is None:
                n_failed += 1
                continue
            outputs[task.task_id] = result.text
            model_call_metrics[task.task_id] = result.metrics

            if delay_between > 0:
                time.sleep(delay_between)

        log.info("Risposte raccolte: %d ok, %d fallite", len(outputs), n_failed)
        scores = score_batch(
            tasks,
            outputs,
            model,
            judge,
            citation_service=None,
            citation_grounding_enabled=False,
        )
        scores = [
            with_model_call_metrics(score, model_call_metrics.get(score.task_id))
            for score in scores
        ]

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        model_slug = model.replace("/", "-")
        run_id = f"{model_slug}/{timestamp}"
        save_scores(scores, run_id)
        summary = summarize_batch_scores(scores)
        summary["model"] = model
        summary["run_id"] = run_id
        summary["scored_at"] = datetime.now(timezone.utc).isoformat()
        summary["model_request_config"] = request_config_for_summary(
            _model_request_kwargs_for_summary(model)
        )
        if grounding_paths is not None:
            summary.update(
                ground_run(RESULTS_DIR / run_id, grounding_paths, n_tasks=len(tasks))
            )
        save_summary(summary, run_id)


def ground_run(
    run_dir: Path,
    grounding_paths: tuple[Path, Path],
    *,
    n_tasks: int,
) -> dict:
    """
    Citation grounding offline della run in ``run_dir``.

    Scrive ``citation_grounding_v3.json`` / ``.md`` nella cartella della run e
    restituisce i campi da aggiungere a ``summary.json``. Un errore del
    grounding non invalida lo scoring: viene registrato nel summary.
    """
    registry_path, profiles_dir = grounding_paths
    try:
        records = records_from_results(run_dir)
        payload = run_grounding(
            records,
            registry_path=registry_path,
            profiles_dir=profiles_dir,
            out_dir=run_dir,
            n_tasks=n_tasks,
        )
    except Exception as exc:  # noqa: BLE001 - lo scoring resta valido
        log.error("Citation grounding non riuscito: %s", exc)
        return {"citation_grounding_status": "error", "citation_grounding_error": str(exc)}

    grounding = payload.get("summary") or {}
    log.info(
        "Citation grounding: GOG=%.1f%%  Coverage=%.1f%%  (%d task, backend=local, registry=%s)",
        float(grounding.get("gog") or 0) * 100,
        float(grounding.get("coverage") or 0) * 100,
        n_tasks,
        str(grounding.get("registry_built_at") or "unknown")[:10],
    )
    log.info("Report grounding: %s", run_dir / "citation_grounding_v3.json")
    return {
        "citation_grounding_status": "complete",
        "citation_grounding_report": str(run_dir / "citation_grounding_v3.json"),
        "gog": grounding.get("gog"),
        "coverage": grounding.get("coverage"),
        "gog_by_task": grounding.get("gog_by_task"),
        "coverage_by_task": grounding.get("coverage_by_task"),
        "gog_backend": grounding.get("gog_backend"),
        "registry_built_at": grounding.get("registry_built_at"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Esegui il benchmark legale italiano.")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help=(
            "Modelli da valutare "
            "(es. gpt-4o claude-sonnet-4-6 gemini-2.5-pro "
            "deepseek/deepseek-v4-pro zai-org/glm-5.2)."
        ),
    )
    parser.add_argument(
        "--area",
        type=str,
        default=None,
        help="Limita a una macro-area, alias legacy accettati.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limita il numero di task.",
    )
    parser.add_argument(
        "--judge",
        type=str,
        default=JUDGE_MODEL,
        help="Alias legacy per il modello del Judge A (default: da config.py).",
    )
    parser.add_argument(
        "--judge-strategy",
        choices=["single", "adaptive_majority"],
        default=None,
        help="Strategia judge: single oppure adaptive_majority.",
    )
    parser.add_argument(
        "--judge-a-provider",
        choices=["anthropic", "openai"],
        default=None,
        help="Provider Judge A.",
    )
    parser.add_argument("--judge-a-model", default=None, help="Modello Judge A.")
    parser.add_argument(
        "--judge-b-provider",
        choices=["anthropic", "openai"],
        default=None,
        help="Provider Judge B.",
    )
    parser.add_argument("--judge-b-model", default=None, help="Modello Judge B.")
    parser.add_argument(
        "--judge-c-provider",
        choices=["anthropic", "openai"],
        default=None,
        help="Provider Judge C.",
    )
    parser.add_argument("--judge-c-model", default=None, help="Modello Judge C.")
    parser.add_argument(
        "--skip-citation-grounding",
        action="store_true",
        help=(
            "Esegue solo lo scoring giuridico. Senza questo flag, al termine dello "
            "scoring viene eseguito il citation grounding offline (richiede il bundle "
            "registry + question profiles in data/citation_pool/)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(
        models=args.models,
        area=args.area,
        limit=args.limit,
        judge_model=args.judge,
        judge_strategy=args.judge_strategy,
        judge_a_provider=args.judge_a_provider,
        judge_a_model=args.judge_a_model,
        judge_b_provider=args.judge_b_provider,
        judge_b_model=args.judge_b_model,
        judge_c_provider=args.judge_c_provider,
        judge_c_model=args.judge_c_model,
        skip_citation_grounding=args.skip_citation_grounding,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
