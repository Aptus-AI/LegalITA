"""
Modelli dati del benchmark.
Tutti i moduli importano da qui — non duplicare strutture dati altrove.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from taxonomy import normalize_macro_area


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

class Provvedimento(BaseModel):
    """
    Singola sentenza o ordinanza della Corte di Cassazione,
    caricata e validata dallo zip del corpus.
    """
    ecli_id:    str         # es. "ECLI_IT_CASS_2025_8915CIV"
    macro_area: str         # slug canonico, es. "diritto_civile"
    division:   str         # es. "Sez. 1"
    legal_area: str         # "CIV" o "PEN"
    doc_type:   str         # "SENT" o "ORD"
    date:       str         # es. "20250226" — stringa grezza da metadata
    facts:      str         # riassunto del caso
    principles: list[str]   # massime giuridiche estratte
    decision:   str         # dispositivo sintetico

    @field_validator("decision", mode="before")
    @classmethod
    def empty_if_none(cls, v: str | None) -> str:
        """Converte None in stringa vuota invece di lanciare errore."""
        return v or ""

    @field_validator("principles", mode="before")
    @classmethod
    def principles_not_empty(cls, v: list) -> list:
        """Garantisce che la lista non sia None."""
        return v or []

    @field_validator("macro_area", mode="before")
    @classmethod
    def macro_area_is_canonical(cls, v: str) -> str:
        """Accetta alias legacy ma salva sempre lo slug canonico."""
        return normalize_macro_area(v, strict=True)


# ---------------------------------------------------------------------------
# Benchmark tasks
# ---------------------------------------------------------------------------

class Criterion(BaseModel):
    """
    Singolo criterio di valutazione nel task.json.
    Corrisponde a un principle della sentenza sorgente,
    riformulato in formato PASS/FAIL per il judge.
    """
    id:             str   # es. "C-001"
    title:          str   # titolo breve del criterio
    match_criteria: str   # "PASS se... FAIL se..." — standard per il judge
    deliverables:   list[str] = ["response.md"]  
    scoring_type:   str = "required"
    category:       str = "legal"


class BenchmarkTask(BaseModel):
    """
    Task completo del benchmark.
    Serializzato come task.json nella cartella tasks/<macro_area>/<task_id>/.
    """
    task_id:     str            # es. "famiglia/prescrizione-nullita/001"
    task_type:   str | None = None
    query:       str            # query sintetica generata dal LLM
    macro_area:  str            # slug canonico; task_id puo restare legacy
    criteria:    list[Criterion]
    source_ecli: str            


    @field_validator("macro_area", mode="before")
    @classmethod
    def macro_area_is_canonical(cls, v: str) -> str:
        """Normalizza in memoria i task storici senza mutare task_id."""
        return normalize_macro_area(v, strict=True)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

JudgeId = Literal["A", "B", "C"]
JudgeProvider = Literal["anthropic", "openai"]
JudgeVerdict = Literal["pass", "fail"]
ConsensusVerdict = Literal["pass", "fail", "unresolved"]
JudgeStatus = Literal["ok", "error"]
ConsensusMethod = Literal[
    "initial_agreement",
    "tie_breaker",
    "recovery_agreement",
    "unresolved",
]
ScoringStatus = Literal["complete", "incomplete"]
CitationVerdict = Literal["pass", "fail", "nc", "unresolved", "not_applicable"]
CitationScoringStatus = Literal["complete", "not_cited", "incomplete", "not_applicable"]


class JudgeVote(BaseModel):
    """Voto prodotto da un singolo judge per un criterio binario."""

    judge_id: JudgeId
    provider: JudgeProvider
    model: str
    verdict: JudgeVerdict | None
    reasoning: str | None
    status: JudgeStatus
    error: str | None = None
    attempts: int = Field(ge=0)
    latency_ms: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_status_consistency(self):
        if self.status == "ok":
            if self.verdict not in ("pass", "fail"):
                raise ValueError("Un voto ok deve avere verdict pass/fail.")
            if self.error is not None:
                raise ValueError("Un voto ok non deve avere error valorizzato.")
        if self.status == "error" and self.error is None:
            raise ValueError("Un voto in errore deve avere un messaggio sintetico.")
        return self


class ConsensusResult(BaseModel):
    """Risultato deterministico del consenso tra judge."""

    verdict: ConsensusVerdict
    reasoning: str
    consensus_method: ConsensusMethod
    supporting_judges: list[JudgeId] = Field(default_factory=list)
    tie_breaker_used: bool
    judge_votes: list[JudgeVote] = Field(default_factory=list)

class CriterionResult(BaseModel):
    """Risultato del judge per un singolo criterio."""
    id:        str   # corrisponde a Criterion.id
    title:     str
    verdict:   str   # "pass", "fail" o "unresolved"
    reasoning: str
    scoring_type: str = "required"
    category:     str = "legal"
    consensus_method: str | None = None
    supporting_judges: list[str] = Field(default_factory=list)
    tie_breaker_used: bool = False
    judge_votes: list[JudgeVote] = Field(default_factory=list)


class TaskScore(BaseModel):
    """
    Output completo della valutazione per un singolo task.
    Serializzato in scores.json.
    """
    task_id:          str
    model:            str               # es. "gpt-4o", "claude-sonnet-4-6"
    model_output:     str               # risposta integrale del modello sotto esame
    model_call_provider: str | None = None
    model_call_latency_ms: int | None = Field(default=None, ge=0)
    model_call_input_tokens: int | None = Field(default=None, ge=0)
    model_call_output_tokens: int | None = Field(default=None, ge=0)
    model_call_total_tokens: int | None = Field(default=None, ge=0)
    model_call_cached_input_tokens: int | None = Field(default=None, ge=0)
    model_call_reasoning_tokens: int | None = Field(default=None, ge=0)
    model_call_estimated_cost_usd: float | None = Field(default=None, ge=0)
    model_call_cost_source: str | None = None
    model_call_usage: dict[str, object] = Field(default_factory=dict)
    citation_results: list[dict] = Field(default_factory=list)
    citation_counts:  dict[str, int] = Field(default_factory=dict)
    citation_hard_fail: bool = False
    citation_registry_built_at: str | None = None
    citation_registry_index_name: str | None = None
    citation_score: float | None = None
    citation_coverage_score: float | None = None
    citation_relevance_score: float | None = None
    citation_fabrication_rate: float | None = None
    citation_score_bounds: dict[str, float | None] = Field(default_factory=dict)
    citation_coverage_bounds: dict[str, float | None] = Field(default_factory=dict)
    citation_relevance_bounds: dict[str, float | None] = Field(default_factory=dict)
    citation_verdict: CitationVerdict = "unresolved"
    citation_scoring_status: CitationScoringStatus = "incomplete"
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
    citation_extraction_status: str | None = None
    citation_extraction_error_count: int = 0
    citation_extraction_diagnostics: dict[str, object] = Field(default_factory=dict)
    citation_extraction_attempt_diagnostics: list[dict[str, object]] = Field(default_factory=list)
    citation_failure_reasons: list[str] = Field(default_factory=list)
    citation_unresolved_reasons: list[str] = Field(default_factory=list)
    citation_coverage: dict[str, object] = Field(default_factory=dict)
    citation_relevance: dict[str, object] = Field(default_factory=dict)
    citation_existence: dict[str, object] = Field(default_factory=dict)
    evaluation_error: str | None = None
    score:            float | None      # 1.0 se all-pass, 0.0 altrimenti, None se incompleto
    all_pass:         bool | None
    reasoning_score:  float | None = None
    reasoning_all_pass: bool | None = None
    reasoning_scoring_status: ScoringStatus = "incomplete"
    content_all_pass: bool | None = None
    scoring_status:   ScoringStatus = "complete"
    n_criteria:       int
    n_passed:         int
    n_unresolved:     int = 0
    n_required:       int
    n_required_passed: int
    n_required_unresolved: int = 0
    n_bonus:          int
    n_bonus_passed:   int
    n_bonus_unresolved: int = 0
    required_pass_rate: float | None
    bonus_pass_rate:  float | None = None
    unresolved_rate:  float = 0.0
    summary:          str
    criteria_results: list[CriterionResult]
    judge_model:      str
    judge_strategy:   str = "single"
    judge_models:     dict[str, str] = Field(default_factory=dict)
    judge_diagnostics: dict[str, object] = Field(default_factory=dict)
    scored_at:        str               # ISO datetime
