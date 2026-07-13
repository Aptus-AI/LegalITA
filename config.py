"""
Configurazione globale del benchmark.
Tutte le costanti condivise tra i moduli stanno qui.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from taxonomy import (
    CANONICAL_MACRO_AREAS,
    MACRO_AREA_LABELS,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths - tutti relativi alla root del progetto
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PRIVATE_DIR = DATA_DIR / "private"
TASKS_DIR = ROOT_DIR / "tasks"
RESULTS_DIR = ROOT_DIR / "results"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

# Current benchmark/gold defaults.
BULLSHIT_GOLD_PATH = PRIVATE_DIR / "bullshit_tasks_v3_missing_documents_40.json"

CORPUS_ZIP = RAW_DIR / "sentenze.zip"
CORPUS_JSONL = PROCESSED_DIR / "corpus.jsonl"

# ---------------------------------------------------------------------------
# Corpus e tassonomia
# ---------------------------------------------------------------------------

# Sette macro-aree canoniche. Gli alias legacy sono definiti in taxonomy.py.
MACRO_AREE: dict[str, str] = MACRO_AREA_LABELS
MACRO_AREE_CANONICHE: tuple[str, ...] = CANONICAL_MACRO_AREAS

# Sezioni escluse dal benchmark.
EXCLUDED_DIVISIONS: set[str] = {"Sez. 7"}

# ---------------------------------------------------------------------------
# Benchmark - parametri di generazione
# ---------------------------------------------------------------------------

# Seed per riproducibilita di tutti i campionamenti random.
RANDOM_SEED: int = 42

# Numero massimo di principles sorgente mostrati al criteria builder.
MAX_SOURCE_PRINCIPLES: int = 5

# Numero massimo di criteri finali salvati per ciascun task.
MAX_CRITERIA: int = 4

# Lunghezza minima dei facts per includere una sentenza, in caratteri.
MIN_FACTS_LENGTH: int = 200

# Numero minimo di principles per includere una sentenza.
MIN_PRINCIPLES: int = 1

# ---------------------------------------------------------------------------
# Evaluation - parametri del judge e del criteria builder
# ---------------------------------------------------------------------------

JUDGE_TEMPERATURE: float = 0.0
JUDGE_MAX_TOKENS: int = int(os.environ.get("JUDGE_MAX_TOKENS", "3000"))
JUDGE_RETRIES: int = 3

GENERATOR_MODEL: str = "claude-sonnet-4-6"  # generazione query e criteri
JUDGE_MODEL: str = "claude-sonnet-4-6"      # valutazione delle risposte

JudgeStrategy = Literal["single", "adaptive_majority"]
JudgeProvider = Literal["anthropic", "openai"]
JudgeId = Literal["A", "B", "C"]

SUPPORTED_JUDGE_STRATEGIES: tuple[str, ...] = ("single", "adaptive_majority")
SUPPORTED_JUDGE_PROVIDERS: tuple[str, ...] = ("anthropic", "openai")
DEFAULT_JUDGE_STRATEGY: JudgeStrategy = "adaptive_majority"
DEFAULT_JUDGE_A_PROVIDER: JudgeProvider = "anthropic"
DEFAULT_JUDGE_A_MODEL: str = JUDGE_MODEL
DEFAULT_JUDGE_B_PROVIDER: JudgeProvider = "openai"
DEFAULT_JUDGE_B_MODEL: str = "gpt-5.5"
DEFAULT_JUDGE_C_PROVIDER: JudgeProvider = "anthropic"
DEFAULT_JUDGE_C_MODEL: str = "claude-opus-4-8"


@dataclass(frozen=True)
class JudgeEndpointConfig:
    judge_id: JudgeId
    provider: JudgeProvider
    model: str


@dataclass(frozen=True)
class JudgeRuntimeConfig:
    strategy: JudgeStrategy
    judge_a: JudgeEndpointConfig
    judge_b: JudgeEndpointConfig
    judge_c: JudgeEndpointConfig


def _env_value(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _provider(value: str | None, default: str) -> JudgeProvider:
    provider = (value or default).strip().lower()
    if provider not in SUPPORTED_JUDGE_PROVIDERS:
        # Il messaggio completo viene costruito dalla validazione runtime.
        return provider  # type: ignore[return-value]
    return provider  # type: ignore[return-value]


def build_judge_runtime_config(
    *,
    judge_strategy: str | None = None,
    judge_a_provider: str | None = None,
    judge_a_model: str | None = None,
    judge_b_provider: str | None = None,
    judge_b_model: str | None = None,
    judge_c_provider: str | None = None,
    judge_c_model: str | None = None,
    legacy_judge_model: str | None = None,
) -> JudgeRuntimeConfig:
    """Legge configurazione judge da override espliciti e variabili ambiente."""
    strategy = (
        judge_strategy
        or _env_value("JUDGE_STRATEGY")
        or DEFAULT_JUDGE_STRATEGY
    ).strip().lower()

    a_model = (
        judge_a_model
        or _env_value("JUDGE_A_MODEL")
        or legacy_judge_model
        or DEFAULT_JUDGE_A_MODEL
    )

    return JudgeRuntimeConfig(
        strategy=strategy,  # type: ignore[arg-type]
        judge_a=JudgeEndpointConfig(
            judge_id="A",
            provider=_provider(
                judge_a_provider or _env_value("JUDGE_A_PROVIDER"),
                DEFAULT_JUDGE_A_PROVIDER,
            ),
            model=a_model.strip(),
        ),
        judge_b=JudgeEndpointConfig(
            judge_id="B",
            provider=_provider(
                judge_b_provider or _env_value("JUDGE_B_PROVIDER"),
                DEFAULT_JUDGE_B_PROVIDER,
            ),
            model=(
                judge_b_model
                or _env_value("JUDGE_B_MODEL")
                or DEFAULT_JUDGE_B_MODEL
            ).strip(),
        ),
        judge_c=JudgeEndpointConfig(
            judge_id="C",
            provider=_provider(
                judge_c_provider or _env_value("JUDGE_C_PROVIDER"),
                DEFAULT_JUDGE_C_PROVIDER,
            ),
            model=(
                judge_c_model
                or _env_value("JUDGE_C_MODEL")
                or DEFAULT_JUDGE_C_MODEL
            ).strip(),
        ),
    )


def validate_judge_runtime_config(config: JudgeRuntimeConfig) -> None:
    """Valida strategia, provider, modelli e credenziali senza esporre secret."""
    if config.strategy not in SUPPORTED_JUDGE_STRATEGIES:
        raise RuntimeError(
            "Strategia judge non supportata: "
            f"{config.strategy!r}. Valori ammessi: {', '.join(SUPPORTED_JUDGE_STRATEGIES)}"
        )

    endpoints = [config.judge_a]
    if config.strategy == "adaptive_majority":
        endpoints.extend([config.judge_b, config.judge_c])

    errors: list[str] = []
    for endpoint in endpoints:
        if endpoint.provider not in SUPPORTED_JUDGE_PROVIDERS:
            errors.append(
                f"JUDGE_{endpoint.judge_id}_PROVIDER={endpoint.provider!r} non supportato"
            )
        if not endpoint.model:
            errors.append(f"JUDGE_{endpoint.judge_id}_MODEL mancante")
        key_name = _api_key_name(endpoint.provider)
        if key_name and not os.environ.get(key_name):
            errors.append(f"{key_name} mancante per Judge {endpoint.judge_id}")

    if errors:
        raise RuntimeError("Configurazione judge non valida: " + "; ".join(errors))

    if (
        config.strategy == "adaptive_majority"
        and config.judge_a.provider == config.judge_b.provider
        and config.judge_a.model == config.judge_b.model
    ):
        log.warning(
            "Judge A e Judge B usano stesso provider e modello (%s/%s): "
            "la maggioranza non e metodologicamente eterogenea.",
            config.judge_a.provider,
            config.judge_a.model,
        )


def _api_key_name(provider: str) -> str | None:
    if provider == "anthropic":
        return "ANTHROPIC_API_KEY"
    if provider == "openai":
        return "OPENAI_API_KEY"
    return None

# Temperatura per la riformulazione query + principles -> criteri PASS/FAIL.
BUILDER_TEMPERATURE: float = 0.0

# ---------------------------------------------------------------------------
# Structured court-rulings S3 resolver. Values are deployment-specific and must
# be provided by environment/CLI in live Aptus runs.
# ---------------------------------------------------------------------------

COURT_RULINGS_S3_BUCKET: str = os.environ.get(
    "COURT_RULINGS_S3_BUCKET",
    "",
).strip()
COURT_RULINGS_S3_PREFIX: str = os.environ.get(
    "COURT_RULINGS_S3_PREFIX",
    "",
).strip()

# ---------------------------------------------------------------------------
# Run - parametri dei modelli sotto esame
# ---------------------------------------------------------------------------

# Budget di output identico per tutti i modelli valutati.
# Include eventuali thinking/reasoning tokens per i modelli che li espongono.
MODEL_MAX_TOKENS: int = int(os.environ.get("MODEL_MAX_TOKENS", "16000"))

# Numero massimo di tentativi per query a un modello sotto esame.
MODEL_RETRIES: int = 3
