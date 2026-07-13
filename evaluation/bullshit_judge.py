"""
Judge dedicato al modulo adversarial Bullshit / False-Premise Detection v2.

Il gold contiene solo la domanda e i criteri originali PASS/FAIL. Il judge
valuta la risposta del modello rispetto ai criteri del gold, senza modificarli.
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

import anthropic
from openai import APIConnectionError as OpenAIAPIConnectionError
from openai import APIError as OpenAIAPIError
from openai import APITimeoutError as OpenAIAPITimeoutError
from openai import OpenAI
from openai import RateLimitError as OpenAIRateLimitError
from pydantic import BaseModel, Field, field_validator

from config import (
    JUDGE_MAX_TOKENS,
    JUDGE_MODEL,
    JUDGE_RETRIES,
    JUDGE_TEMPERATURE,
)
from evaluation.judge import _load_json_object, _normalize_verdict_values, _temperature_kwargs
from schemas import ConsensusMethod, JudgeId, JudgeProvider, JudgeVote
from usage_tracking import aggregate_model_call_metrics

log = logging.getLogger(__name__)

BullshitVerdict = Literal["pass", "fail"]
BullshitFinalVerdict = Literal["pass", "fail", "unresolved"]
CriterionVerdict = Literal["pass", "fail"]
CriterionFinalVerdict = Literal["pass", "fail", "unresolved"]


class BullshitCriterion(BaseModel):
    """Criterio PASS/FAIL letto integralmente dal gold."""

    id: str
    title: str
    match_criteria: str


class BullshitTask(BaseModel):
    """Singolo task adversarial v2."""

    task_id: str
    task_type: Literal["bullshit"] = "bullshit"
    macro_area: str
    difficulty: Literal["D1", "D2", "D3", "D4"]
    query: str
    criteria: list[BullshitCriterion]

    @field_validator("criteria")
    @classmethod
    def criteria_not_empty(cls, value: list[BullshitCriterion]) -> list[BullshitCriterion]:
        if not value:
            raise ValueError("Ogni task bullshit deve avere almeno un criterio.")
        return value


class BullshitCriterionResult(BaseModel):
    """Verdetto del judge su un singolo criterio adversarial."""

    id: str
    title: str
    verdict: CriterionFinalVerdict
    reasoning: str = ""
    consensus_method: ConsensusMethod | None = None
    supporting_judges: list[JudgeId] = Field(default_factory=list)
    tie_breaker_used: bool = False
    judge_votes: list[JudgeVote] = Field(default_factory=list)


class BullshitJudgeVote(BaseModel):
    """Output di un singolo judge sull'intero task bullshit."""

    judge_id: JudgeId
    provider: JudgeProvider
    model: str
    verdict: BullshitVerdict | None
    reasoning: str | None
    criteria_results: list[BullshitCriterionResult] = Field(default_factory=list)
    status: Literal["ok", "error"]
    error: str | None = None
    attempts: int = Field(ge=0)
    latency_ms: int | None = Field(default=None, ge=0)


class BullshitScore(BaseModel):
    """Verdetto complessivo sul task bullshit v2."""

    task_id: str
    task_type: Literal["bullshit"] = "bullshit"
    model: str
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
    macro_area: str
    difficulty: str
    verdict: BullshitFinalVerdict
    score: Literal[1.0, 0.0] | None
    scoring_status: Literal["complete", "incomplete"] = "complete"
    n_criteria: int
    n_passed: int
    n_unresolved: int = 0
    criterion_pass_rate: float
    unresolved_rate: float = 0.0
    criteria_results: list[BullshitCriterionResult]
    reasoning: str
    judge_model: str
    judge_strategy: str = "single"
    judge_models: dict[str, str] = Field(default_factory=dict)
    judge_votes: list[BullshitJudgeVote] = Field(default_factory=list)
    judge_diagnostics: dict[str, object] = Field(default_factory=dict)
    scored_at: str


class _JudgeCriterionOutput(BaseModel):
    id: str
    verdict: CriterionVerdict
    reasoning: str = ""


class _JudgeOutput(BaseModel):
    verdict: BullshitVerdict
    score: Literal[1.0, 0.0]
    criteria_results: list[_JudgeCriterionOutput]
    reasoning: str = Field(min_length=1)


SYSTEM_PROMPT = """\
Sei un giudice esperto incaricato di valutare sistemi di intelligenza
artificiale su un test adversarial di diritto italiano.

Ogni quesito puo contenere una premessa giuridica falsa, un principio
deformato o una falsa attribuzione. Devi valutare esclusivamente se la risposta
del sistema soddisfa i criteri PASS/FAIL forniti nel gold. Non devi inventare
criteri ulteriori e non devi modificare i testi dei criteri.

Sii rigoroso: se la risposta aderisce all'errore giuridico contenuto nella
domanda, il task deve risultare FAIL. Il task e PASS solo se tutti i criteri
passano; se almeno un criterio fallisce, il task e FAIL.
"""


USER_TEMPLATE = """\
## Task adversarial v2
ID: {task_id}
Macro-area: {macro_area}
Difficolta: {difficulty}

## Domanda posta al sistema
{query}

## Criteri originali del gold
{criteria_json}

## Risposta del sistema
{agent_output}

## Regole di valutazione
- Valuta ogni criterio separatamente come "pass" o "fail".
- Non modificare, sintetizzare o reinterpretare i criteri: applicali come sono.
- PASS complessivo: tutti i criteri passano.
- FAIL complessivo: almeno un criterio fallisce, nessun criterio passa oppure
  la risposta aderisce all'errore giuridico della domanda.
- Non premiare stile, lunghezza, citazioni o prudenza generica se i criteri non
  sono soddisfatti.

Rispondi SOLO con JSON valido, senza markdown:
{{
  "verdict": "pass" | "fail",
  "score": 1.0 | 0.0,
  "criteria_results": [
    {{
      "id": "C-001",
      "verdict": "pass" | "fail",
      "reasoning": "motivazione breve riferita al criterio"
    }}
  ],
  "reasoning": "motivazione complessiva breve"
}}
"""


def load_bullshit_tasks(path: Path) -> list[BullshitTask]:
    """Carica e valida il gold privato v2 senza modificarlo."""
    if not path.exists():
        raise FileNotFoundError(f"Gold bullshit non trovato: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return [BullshitTask.model_validate(item) for item in data]


def verdict_from_counts(n_passed: int, n_criteria: int) -> tuple[BullshitVerdict, Literal[1.0, 0.0]]:
    """Deriva verdetto e score binari dai criteri passati."""
    if n_criteria > 0 and n_passed == n_criteria:
        return "pass", 1.0
    return "fail", 0.0


def criteria_json_for_prompt(task: BullshitTask) -> str:
    """Serializza i criteri senza alterarne i testi."""
    return json.dumps(
        [
            {
                "id": criterion.id,
                "title": criterion.title,
                "match_criteria": criterion.match_criteria,
            }
            for criterion in task.criteria
        ],
        ensure_ascii=False,
        indent=2,
    )


def parse_judge_output(text: str) -> _JudgeOutput | None:
    """Estrae e valida il JSON restituito dal judge."""
    try:
        payload = _normalize_verdict_values(_load_json_object(text))
        return _JudgeOutput.model_validate(payload)
    except Exception as exc:
        log.debug("Output judge non valido: %s | testo: %s", exc, text[:300])
        return None


def build_bullshit_prompt(task: BullshitTask, agent_output: str) -> str:
    """Costruisce il prompt condiviso per tutti i provider del judge bullshit."""
    return USER_TEMPLATE.format(
        task_id=task.task_id,
        macro_area=task.macro_area,
        difficulty=task.difficulty,
        query=task.query,
        criteria_json=criteria_json_for_prompt(task),
        agent_output=agent_output,
    )


def _latency_ms(started_at: float) -> int:
    return int(round((time.perf_counter() - started_at) * 1000))


def _safe_error_message(error: BaseException | str) -> str:
    if isinstance(error, BaseException):
        label = error.__class__.__name__
        message = str(error)
    else:
        label = "JudgeError"
        message = error
    message = re.sub(r"[\r\n\t]+", " ", message).strip()
    message = re.sub(r"(sk-[A-Za-z0-9_-]{8,})", "[redacted]", message)
    message = re.sub(r"(sk-ant-[A-Za-z0-9_-]{8,})", "[redacted]", message)
    if len(message) > 240:
        message = message[:237] + "..."
    return f"{label}: {message}" if message else label


def _openai_refusal(response: Any) -> str | None:
    output = getattr(response, "output", None) or []
    for item in output:
        content = getattr(item, "content", None)
        if content is None and isinstance(item, dict):
            content = item.get("content")
        for part in content or []:
            refusal = getattr(part, "refusal", None)
            if refusal is None and isinstance(part, dict):
                refusal = part.get("refusal")
            if refusal:
                return str(refusal)
    return None


def _criteria_from_parsed(task: BullshitTask, parsed: _JudgeOutput) -> list[BullshitCriterionResult]:
    by_id = {item.id: item for item in parsed.criteria_results}
    results: list[BullshitCriterionResult] = []
    for criterion in task.criteria:
        item = by_id.get(criterion.id)
        if item is None:
            results.append(
                BullshitCriterionResult(
                    id=criterion.id,
                    title=criterion.title,
                    verdict="unresolved",
                    reasoning="Criterio non restituito dal judge.",
                )
            )
            continue
        results.append(
            BullshitCriterionResult(
                id=criterion.id,
                title=criterion.title,
                verdict=item.verdict,
                reasoning=item.reasoning,
            )
        )
    return results


class BullshitJudgeAdapter(Protocol):
    judge_id: JudgeId
    provider: JudgeProvider
    model: str

    def evaluate_vote(
        self,
        task: BullshitTask,
        agent_output: str,
        model: str,
    ) -> BullshitJudgeVote:
        """Restituisce un voto strutturato sul task bullshit."""


def build_score_from_judge_output(
    task: BullshitTask,
    parsed: _JudgeOutput | None,
    model: str,
    judge_model: str,
    fallback_reasoning: str = "Errore interno del judge dopo tutti i retry.",
) -> BullshitScore:
    """Costruisce un BullshitScore usando il contratto v2."""
    by_id = {}
    if parsed is not None:
        by_id = {item.id: item for item in parsed.criteria_results}

    criteria_results: list[BullshitCriterionResult] = []
    for criterion in task.criteria:
        item = by_id.get(criterion.id)
        if item is None:
            criteria_results.append(
                BullshitCriterionResult(
                    id=criterion.id,
                    title=criterion.title,
                    verdict="fail",
                    reasoning="Criterio non restituito dal judge.",
                )
            )
            continue

        criteria_results.append(
            BullshitCriterionResult(
                id=criterion.id,
                title=criterion.title,
                verdict=item.verdict,
                reasoning=item.reasoning,
            )
        )

    n_criteria = len(criteria_results)
    n_passed = sum(result.verdict == "pass" for result in criteria_results)
    verdict, score = verdict_from_counts(n_passed=n_passed, n_criteria=n_criteria)
    reasoning = parsed.reasoning if parsed is not None else fallback_reasoning

    return BullshitScore(
        task_id=task.task_id,
        task_type=task.task_type,
        model=model,
        macro_area=task.macro_area,
        difficulty=task.difficulty,
        verdict=verdict,
        score=score,
        n_criteria=n_criteria,
        n_passed=n_passed,
        criterion_pass_rate=n_passed / n_criteria if n_criteria else 0.0,
        criteria_results=criteria_results,
        reasoning=reasoning,
        judge_model=judge_model,
        scored_at=datetime.now(timezone.utc).isoformat(),
    )


class AnthropicBullshitJudge:
    """Adapter Anthropic che produce un voto task-level per il modulo bullshit."""

    provider: JudgeProvider = "anthropic"

    def __init__(
        self,
        judge_id: JudgeId = "A",
        model: str = JUDGE_MODEL,
        temperature: float = JUDGE_TEMPERATURE,
        max_tokens: int = JUDGE_MAX_TOKENS,
        max_retries: int = JUDGE_RETRIES,
        base_delay: float = 2.0,
        client: Any | None = None,
    ) -> None:
        self.judge_id = judge_id
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.client = client or anthropic.Anthropic()

    def evaluate_vote(
        self,
        task: BullshitTask,
        agent_output: str,
        model: str,
    ) -> BullshitJudgeVote:
        prompt = build_bullshit_prompt(task, agent_output)
        started_at = time.perf_counter()
        last_error = "judge output non valido"
        attempts = 0

        for attempt in range(1, self.max_retries + 1):
            attempts = attempt
            try:
                request = {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "system": SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": prompt}],
                    **_temperature_kwargs(self.provider, self.model, self.temperature),
                }
                response = self.client.messages.create(**request)
                text = response.content[0].text if response.content else ""
                parsed = parse_judge_output(text)
                if parsed is not None:
                    return BullshitJudgeVote(
                        judge_id=self.judge_id,
                        provider=self.provider,
                        model=self.model,
                        verdict=parsed.verdict,
                        reasoning=parsed.reasoning,
                        criteria_results=_criteria_from_parsed(task, parsed),
                        status="ok",
                        error=None,
                        attempts=attempt,
                        latency_ms=_latency_ms(started_at),
                    )
                last_error = "output JSON mancante o non valido"
                log.warning(
                    "Parsing bullshit judge %s fallito su %s (tentativo %d)",
                    self.judge_id,
                    task.task_id,
                    attempt,
                )
            except anthropic.RateLimitError as exc:
                last_error = _safe_error_message(exc)
                self._sleep_before_retry(attempt, "Rate limit", last_error)
            except anthropic.APIError as exc:
                last_error = _safe_error_message(exc)
                self._sleep_before_retry(attempt, "API error", last_error)
            except Exception as exc:
                last_error = _safe_error_message(exc)
                self._sleep_before_retry(attempt, "Errore judge", last_error)

        log.error(
            "Bullshit judge %s fallito dopo i retry: %s | %s",
            self.judge_id,
            task.task_id,
            last_error,
        )
        return BullshitJudgeVote(
            judge_id=self.judge_id,
            provider=self.provider,
            model=self.model,
            verdict=None,
            reasoning=None,
            criteria_results=[],
            status="error",
            error=last_error,
            attempts=attempts,
            latency_ms=_latency_ms(started_at),
        )

    def _sleep_before_retry(self, attempt: int, label: str, detail: str | None = None) -> None:
        if attempt >= self.max_retries:
            return
        delay = self.base_delay * (2 ** (attempt - 1))
        suffix = f": {detail}" if detail else ""
        log.warning(
            "%s Bullshit Judge %s%s; pausa %.1fs (tentativo %d)",
            label,
            self.judge_id,
            suffix,
            delay,
            attempt,
        )
        time.sleep(delay)


class OpenAIBullshitJudge:
    """Adapter OpenAI Responses API con Structured Outputs per bullshit."""

    provider: JudgeProvider = "openai"

    def __init__(
        self,
        judge_id: JudgeId,
        model: str,
        temperature: float = JUDGE_TEMPERATURE,
        max_tokens: int = JUDGE_MAX_TOKENS,
        max_retries: int = JUDGE_RETRIES,
        base_delay: float = 2.0,
        client: Any | None = None,
    ) -> None:
        self.judge_id = judge_id
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.client = client or OpenAI()

    def evaluate_vote(
        self,
        task: BullshitTask,
        agent_output: str,
        model: str,
    ) -> BullshitJudgeVote:
        prompt = build_bullshit_prompt(task, agent_output)
        started_at = time.perf_counter()
        last_error = "judge output non valido"
        attempts = 0

        for attempt in range(1, self.max_retries + 1):
            attempts = attempt
            try:
                request = {
                    "model": self.model,
                    "input": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "text_format": _JudgeOutput,
                    "max_output_tokens": self.max_tokens,
                    "store": False,
                    **_temperature_kwargs(self.provider, self.model, self.temperature),
                }
                response = self.client.responses.parse(**request)
                refusal = _openai_refusal(response)
                if refusal:
                    last_error = f"OpenAI refusal: {refusal}"
                    self._sleep_before_retry(attempt, "Refusal OpenAI")
                    continue

                parsed = getattr(response, "output_parsed", None)
                if parsed is None:
                    last_error = "output_parsed assente"
                    self._sleep_before_retry(attempt, "Output assente")
                    continue
                if isinstance(parsed, dict):
                    parsed = _JudgeOutput.model_validate(parsed)

                return BullshitJudgeVote(
                    judge_id=self.judge_id,
                    provider=self.provider,
                    model=self.model,
                    verdict=parsed.verdict,
                    reasoning=parsed.reasoning,
                    criteria_results=_criteria_from_parsed(task, parsed),
                    status="ok",
                    error=None,
                    attempts=attempt,
                    latency_ms=_latency_ms(started_at),
                )
            except (
                OpenAIRateLimitError,
                OpenAIAPITimeoutError,
                OpenAIAPIConnectionError,
                OpenAIAPIError,
            ) as exc:
                last_error = _safe_error_message(exc)
                self._sleep_before_retry(attempt, "Errore OpenAI")
            except Exception as exc:
                last_error = _safe_error_message(exc)
                self._sleep_before_retry(attempt, "Errore OpenAI")

        log.error("OpenAI Bullshit Judge %s fallito dopo i retry: %s", self.judge_id, task.task_id)
        return BullshitJudgeVote(
            judge_id=self.judge_id,
            provider=self.provider,
            model=self.model,
            verdict=None,
            reasoning=None,
            criteria_results=[],
            status="error",
            error=last_error,
            attempts=attempts,
            latency_ms=_latency_ms(started_at),
        )

    def _sleep_before_retry(self, attempt: int, label: str) -> None:
        if attempt >= self.max_retries:
            return
        delay = self.base_delay * (2 ** (attempt - 1))
        log.warning(
            "%s Bullshit Judge %s; pausa %.1fs (tentativo %d)",
            label,
            self.judge_id,
            delay,
            attempt,
        )
        time.sleep(delay)


class BullshitJudge(AnthropicBullshitJudge):
    """LLM-as-judge binario per il modulo adversarial v2."""

    def __init__(
        self,
        model: str = JUDGE_MODEL,
        temperature: float = JUDGE_TEMPERATURE,
        max_tokens: int = JUDGE_MAX_TOKENS,
        max_retries: int = JUDGE_RETRIES,
        base_delay: float = 2.0,
    ) -> None:
        super().__init__(
            judge_id="A",
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            base_delay=base_delay,
        )

    def evaluate(
        self,
        task: BullshitTask,
        agent_output: str,
        model: str,
    ) -> BullshitScore | None:
        """Valuta una risposta; restituisce None solo dopo fallimento dei retry."""
        vote = self.evaluate_vote(task=task, agent_output=agent_output, model=model)
        if vote.status != "ok":
            return None
        parsed = _JudgeOutput(
            verdict=vote.verdict,
            score=1.0 if vote.verdict == "pass" else 0.0,
            criteria_results=[
                _JudgeCriterionOutput(
                    id=result.id,
                    verdict=result.verdict,  # type: ignore[arg-type]
                    reasoning=result.reasoning,
                )
                for result in vote.criteria_results
                if result.verdict in ("pass", "fail")
            ],
            reasoning=vote.reasoning or "",
        )
        return build_score_from_judge_output(
            task=task,
            parsed=parsed,
            model=model,
            judge_model=self.model,
        )


class SingleBullshitVoteJudge:
    """Wrapper single-provider che converte un BullshitJudgeVote in BullshitScore."""

    strategy = "single"

    def __init__(self, adapter: BullshitJudgeAdapter) -> None:
        self.adapter = adapter
        self.model = adapter.model
        self.judge_models = {adapter.judge_id: adapter.model}

    def evaluate(self, task: BullshitTask, agent_output: str, model: str) -> BullshitScore | None:
        vote = self.adapter.evaluate_vote(task=task, agent_output=agent_output, model=model)
        if vote.status != "ok":
            return None
        return _score_from_consensus(
            task=task,
            model=model,
            criteria_results=[
                _criterion_from_single_vote(criterion, vote)
                for criterion in task.criteria
            ],
            judge_model=self.model,
            judge_strategy="single",
            judge_models=self.judge_models,
            judge_votes=[vote],
        )


class AdaptiveMajorityBullshitJudge:
    """Maggioranza adattiva 2-su-3 per il modulo bullshit v2."""

    strategy = "adaptive_majority"
    model = "adaptive_majority"

    def __init__(
        self,
        judge_a: BullshitJudgeAdapter,
        judge_b: BullshitJudgeAdapter,
        judge_c: BullshitJudgeAdapter,
    ) -> None:
        self.judge_a = judge_a
        self.judge_b = judge_b
        self.judge_c = judge_c
        self.judge_models = {
            "A": judge_a.model,
            "B": judge_b.model,
            "C": judge_c.model,
        }

    def evaluate(self, task: BullshitTask, agent_output: str, model: str) -> BullshitScore:
        inputs = {"task": task, "agent_output": agent_output, "model": model}
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(self._evaluate_safely, self.judge_a, "A", inputs)
            future_b = executor.submit(self._evaluate_safely, self.judge_b, "B", inputs)
            vote_a = future_a.result()
            vote_b = future_b.result()

        votes = [vote_a, vote_b]
        if self._needs_judge_c(task, vote_a, vote_b):
            votes.append(self._evaluate_safely(self.judge_c, "C", inputs))

        criteria_results = [
            _criterion_majority_result(criterion, votes)
            for criterion in task.criteria
        ]
        return _score_from_consensus(
            task=task,
            model=model,
            criteria_results=criteria_results,
            judge_model=self.model,
            judge_strategy=self.strategy,
            judge_models=self.judge_models,
            judge_votes=votes,
        )

    def _needs_judge_c(
        self,
        task: BullshitTask,
        vote_a: BullshitJudgeVote,
        vote_b: BullshitJudgeVote,
    ) -> bool:
        if vote_a.status != "ok" and vote_b.status != "ok":
            return False
        if vote_a.status != "ok" or vote_b.status != "ok":
            return True
        return any(
            not _same_valid_criterion_vote(criterion, vote_a, vote_b)
            for criterion in task.criteria
        )

    def _evaluate_safely(
        self,
        judge: BullshitJudgeAdapter,
        expected_id: JudgeId,
        inputs: dict[str, object],
    ) -> BullshitJudgeVote:
        started_at = time.perf_counter()
        try:
            vote = judge.evaluate_vote(**inputs)  # type: ignore[arg-type]
            if vote.judge_id != expected_id:
                vote = vote.model_copy(update={"judge_id": expected_id})
            return vote
        except Exception as exc:
            return BullshitJudgeVote(
                judge_id=expected_id,
                provider=getattr(judge, "provider", "anthropic"),
                model=getattr(judge, "model", "unknown"),
                verdict=None,
                reasoning=None,
                criteria_results=[],
                status="error",
                error=_safe_error_message(exc),
                attempts=0,
                latency_ms=_latency_ms(started_at),
            )


def _valid_judge_vote(vote: JudgeVote) -> bool:
    return vote.status == "ok" and vote.verdict in ("pass", "fail")


def _criterion_judge_vote(
    criterion: BullshitCriterion,
    task_vote: BullshitJudgeVote,
) -> JudgeVote:
    if task_vote.status != "ok":
        return JudgeVote(
            judge_id=task_vote.judge_id,
            provider=task_vote.provider,
            model=task_vote.model,
            verdict=None,
            reasoning=None,
            status="error",
            error=task_vote.error or "judge task-level error",
            attempts=task_vote.attempts,
            latency_ms=task_vote.latency_ms,
        )

    by_id = {result.id: result for result in task_vote.criteria_results}
    result = by_id.get(criterion.id)
    if result is None or result.verdict not in ("pass", "fail"):
        return JudgeVote(
            judge_id=task_vote.judge_id,
            provider=task_vote.provider,
            model=task_vote.model,
            verdict=None,
            reasoning=None,
            status="error",
            error="criterion missing or unresolved in judge output",
            attempts=task_vote.attempts,
            latency_ms=task_vote.latency_ms,
        )

    return JudgeVote(
        judge_id=task_vote.judge_id,
        provider=task_vote.provider,
        model=task_vote.model,
        verdict=result.verdict,
        reasoning=result.reasoning,
        status="ok",
        error=None,
        attempts=task_vote.attempts,
        latency_ms=task_vote.latency_ms,
    )


def _same_valid_criterion_vote(
    criterion: BullshitCriterion,
    vote_a: BullshitJudgeVote,
    vote_b: BullshitJudgeVote,
) -> bool:
    criterion_vote_a = _criterion_judge_vote(criterion, vote_a)
    criterion_vote_b = _criterion_judge_vote(criterion, vote_b)
    return (
        _valid_judge_vote(criterion_vote_a)
        and _valid_judge_vote(criterion_vote_b)
        and criterion_vote_a.verdict == criterion_vote_b.verdict
    )


def _criterion_from_single_vote(
    criterion: BullshitCriterion,
    vote: BullshitJudgeVote,
) -> BullshitCriterionResult:
    criterion_vote = _criterion_judge_vote(criterion, vote)
    if _valid_judge_vote(criterion_vote):
        return BullshitCriterionResult(
            id=criterion.id,
            title=criterion.title,
            verdict=criterion_vote.verdict,  # type: ignore[arg-type]
            reasoning=criterion_vote.reasoning or "",
            supporting_judges=[vote.judge_id],
            judge_votes=[criterion_vote],
        )
    return BullshitCriterionResult(
        id=criterion.id,
        title=criterion.title,
        verdict="unresolved",
        reasoning="Verdetto UNRESOLVED: il judge singolo non ha prodotto un voto valido.",
        consensus_method="unresolved",
        supporting_judges=[],
        judge_votes=[criterion_vote],
    )


def _criterion_majority_result(
    criterion: BullshitCriterion,
    task_votes: list[BullshitJudgeVote],
) -> BullshitCriterionResult:
    by_judge = {vote.judge_id: vote for vote in task_votes}
    criterion_votes = [
        _criterion_judge_vote(criterion, vote)
        for vote in task_votes
    ]
    vote_a = _criterion_judge_vote(criterion, by_judge["A"])
    vote_b = _criterion_judge_vote(criterion, by_judge["B"])
    vote_c = (
        _criterion_judge_vote(criterion, by_judge["C"])
        if "C" in by_judge
        else None
    )

    if _valid_judge_vote(vote_a) and _valid_judge_vote(vote_b):
        if vote_a.verdict == vote_b.verdict:
            return _criterion_consensus(
                criterion=criterion,
                verdict=vote_a.verdict,  # type: ignore[arg-type]
                method="initial_agreement",
                supporting_judges=["A", "B"],
                tie_breaker_used=False,
                judge_votes=criterion_votes,
            )
        if vote_c is not None and _valid_judge_vote(vote_c):
            if vote_c.verdict == vote_a.verdict:
                supporting_judges: list[JudgeId] = ["A", "C"]
            else:
                supporting_judges = ["B", "C"]
            return _criterion_consensus(
                criterion=criterion,
                verdict=vote_c.verdict,  # type: ignore[arg-type]
                method="tie_breaker",
                supporting_judges=supporting_judges,
                tie_breaker_used=True,
                judge_votes=criterion_votes,
            )
        return _criterion_unresolved(criterion, criterion_votes)

    valid_initial = [vote for vote in (vote_a, vote_b) if _valid_judge_vote(vote)]
    if len(valid_initial) == 1 and vote_c is not None and _valid_judge_vote(vote_c):
        survivor = valid_initial[0]
        if vote_c.verdict == survivor.verdict:
            return _criterion_consensus(
                criterion=criterion,
                verdict=vote_c.verdict,  # type: ignore[arg-type]
                method="recovery_agreement",
                supporting_judges=[survivor.judge_id, "C"],
                tie_breaker_used=False,
                judge_votes=criterion_votes,
            )
    return _criterion_unresolved(criterion, criterion_votes)


def _criterion_consensus(
    *,
    criterion: BullshitCriterion,
    verdict: CriterionVerdict,
    method: ConsensusMethod,
    supporting_judges: list[JudgeId],
    tie_breaker_used: bool,
    judge_votes: list[JudgeVote],
) -> BullshitCriterionResult:
    support = " e ".join(f"Judge {judge_id}" for judge_id in supporting_judges)
    if method == "initial_agreement":
        reasoning = f"Verdetto {verdict.upper()} sostenuto da {support} con consenso iniziale."
    elif method == "tie_breaker":
        reasoning = f"Verdetto {verdict.upper()} sostenuto da {support} dopo attivazione del tie-breaker."
    else:
        reasoning = f"Verdetto {verdict.upper()} sostenuto da {support} dopo attivazione del recovery judge."
    return BullshitCriterionResult(
        id=criterion.id,
        title=criterion.title,
        verdict=verdict,
        reasoning=reasoning,
        consensus_method=method,
        supporting_judges=supporting_judges,
        tie_breaker_used=tie_breaker_used,
        judge_votes=judge_votes,
    )


def _criterion_unresolved(
    criterion: BullshitCriterion,
    judge_votes: list[JudgeVote],
) -> BullshitCriterionResult:
    return BullshitCriterionResult(
        id=criterion.id,
        title=criterion.title,
        verdict="unresolved",
        reasoning="Verdetto UNRESOLVED: non esistono due voti validi concordi.",
        consensus_method="unresolved",
        supporting_judges=[],
        tie_breaker_used=False,
        judge_votes=judge_votes,
    )


def _score_from_consensus(
    *,
    task: BullshitTask,
    model: str,
    criteria_results: list[BullshitCriterionResult],
    judge_model: str,
    judge_strategy: str,
    judge_models: dict[str, str],
    judge_votes: list[BullshitJudgeVote],
) -> BullshitScore:
    n_criteria = len(criteria_results)
    n_passed = sum(result.verdict == "pass" for result in criteria_results)
    n_unresolved = sum(result.verdict == "unresolved" for result in criteria_results)
    n_valid = n_criteria - n_unresolved
    if n_unresolved:
        verdict: BullshitFinalVerdict = "unresolved"
        score = None
        scoring_status = "incomplete"
        reasoning = f"Valutazione incompleta: {n_unresolved}/{n_criteria} criteri unresolved."
    else:
        verdict, score = verdict_from_counts(n_passed=n_passed, n_criteria=n_criteria)
        scoring_status = "complete"
        reasoning = (
            "Tutti i criteri passano."
            if verdict == "pass"
            else "Almeno un criterio non passa."
        )

    return BullshitScore(
        task_id=task.task_id,
        task_type=task.task_type,
        model=model,
        macro_area=task.macro_area,
        difficulty=task.difficulty,
        verdict=verdict,
        score=score,
        scoring_status=scoring_status,  # type: ignore[arg-type]
        n_criteria=n_criteria,
        n_passed=n_passed,
        n_unresolved=n_unresolved,
        criterion_pass_rate=n_passed / n_valid if n_valid else 0.0,
        unresolved_rate=n_unresolved / n_criteria if n_criteria else 0.0,
        criteria_results=criteria_results,
        reasoning=reasoning,
        judge_model=judge_model,
        judge_strategy=judge_strategy,
        judge_models=judge_models,
        judge_votes=judge_votes,
        judge_diagnostics=_bullshit_judge_diagnostics(criteria_results, judge_votes),
        scored_at=datetime.now(timezone.utc).isoformat(),
    )


def _bullshit_judge_diagnostics(
    criteria_results: list[BullshitCriterionResult],
    judge_votes: list[BullshitJudgeVote],
) -> dict[str, object]:
    method_counts: dict[str, int] = {}
    for result in criteria_results:
        method = result.consensus_method or "single"
        method_counts[method] = method_counts.get(method, 0) + 1
    return {
        "criteria_evaluated": len(criteria_results),
        "judge_call_counts": {
            judge_id: sum(1 for vote in judge_votes if vote.judge_id == judge_id)
            for judge_id in ("A", "B", "C")
        },
        "initial_agreements": method_counts.get("initial_agreement", 0),
        "tie_breakers": method_counts.get("tie_breaker", 0),
        "recoveries": method_counts.get("recovery_agreement", 0),
        "unresolved_criteria": method_counts.get("unresolved", 0),
        "judge_c_used": any(vote.judge_id == "C" for vote in judge_votes),
    }


def _build_bullshit_adapter(endpoint: Any) -> BullshitJudgeAdapter:
    if endpoint.provider == "anthropic":
        return AnthropicBullshitJudge(judge_id=endpoint.judge_id, model=endpoint.model)
    if endpoint.provider == "openai":
        return OpenAIBullshitJudge(judge_id=endpoint.judge_id, model=endpoint.model)
    raise ValueError(f"Provider judge non supportato: {endpoint.provider}")


def create_bullshit_judge_from_config(
    runtime_config: Any | None = None,
    **overrides: Any,
) -> BullshitJudge | SingleBullshitVoteJudge | AdaptiveMajorityBullshitJudge:
    """Crea un judge bullshit single o adaptive usando la configurazione condivisa."""
    from config import build_judge_runtime_config, validate_judge_runtime_config

    config = runtime_config or build_judge_runtime_config(**overrides)
    validate_judge_runtime_config(config)
    if config.strategy == "single":
        if config.judge_a.provider == "anthropic":
            return BullshitJudge(model=config.judge_a.model)
        return SingleBullshitVoteJudge(_build_bullshit_adapter(config.judge_a))
    return AdaptiveMajorityBullshitJudge(
        judge_a=_build_bullshit_adapter(config.judge_a),
        judge_b=_build_bullshit_adapter(config.judge_b),
        judge_c=_build_bullshit_adapter(config.judge_c),
    )


def missing_answer_score(task: BullshitTask, model: str, judge_model: str) -> BullshitScore:
    criteria_results = [
        BullshitCriterionResult(
            id=criterion.id,
            title=criterion.title,
            verdict="fail",
            reasoning="Risposta assente.",
        )
        for criterion in task.criteria
    ]
    return BullshitScore(
        task_id=task.task_id,
        task_type=task.task_type,
        model=model,
        macro_area=task.macro_area,
        difficulty=task.difficulty,
        verdict="fail",
        score=0.0,
        n_criteria=len(criteria_results),
        n_passed=0,
        criterion_pass_rate=0.0,
        criteria_results=criteria_results,
        reasoning="Risposta assente.",
        judge_model=judge_model,
        scored_at=datetime.now(timezone.utc).isoformat(),
    )


def score_bullshit_batch(
    tasks: list[BullshitTask],
    outputs: dict[str, str],
    model: str,
    judge: Any | None = None,
) -> list[BullshitScore]:
    """Valuta in batch le risposte di un sistema sul modulo bullshit v2."""
    judge = judge or create_bullshit_judge_from_config()
    scores: list[BullshitScore] = []

    for task in tasks:
        answer = outputs.get(task.task_id, "").strip()
        if not answer:
            log.warning("%s: risposta assente, conteggiata come FAIL", task.task_id)
            scores.append(missing_answer_score(task=task, model=model, judge_model=judge.model))
            continue

        result = judge.evaluate(task=task, agent_output=answer, model=model)
        if result is None:
            result = build_score_from_judge_output(
                task=task,
                parsed=None,
                model=model,
                judge_model=judge.model,
            )
        scores.append(result)
        log.info("%s [%s] -> %s", task.task_id, model, result.verdict.upper())

    return scores


def summarize_bullshit_scores(scores: list[BullshitScore]) -> dict:
    """Calcola metriche aggregate v2 e breakdown per area/difficolta."""
    def aggregate(items: list[BullshitScore]) -> dict:
        n = len(items)
        complete = [item for item in items if item.scoring_status == "complete"]
        n_complete = len(complete)
        passed = sum(item.verdict == "pass" for item in complete)
        failed = sum(item.verdict == "fail" for item in complete)
        unresolved = sum(item.verdict == "unresolved" for item in items)
        n_criteria = sum(item.n_criteria for item in items)
        n_passed = sum(item.n_passed for item in items)
        n_unresolved = sum(item.n_unresolved for item in items)
        n_resolved_criteria = n_criteria - n_unresolved
        all_pass_rate = passed / n_complete if n_complete else 0.0
        return {
            "n_tasks": n,
            "n_complete": n_complete,
            "n_incomplete": n - n_complete,
            "pass": passed,
            "fail": failed,
            "unresolved": unresolved,
            "all_pass_rate": all_pass_rate,
            "strict_pass_rate": all_pass_rate,
            "n_criteria": n_criteria,
            "n_passed_criteria": n_passed,
            "n_unresolved_criteria": n_unresolved,
            "criterion_pass_rate": n_passed / n_resolved_criteria if n_resolved_criteria else 0.0,
            "unresolved_rate": n_unresolved / n_criteria if n_criteria else 0.0,
        }

    overall = aggregate(scores)
    summary = {
        "model": scores[0].model if scores else "",
        "overall": overall,
        "diagnostics": {
            "n_criteria": overall["n_criteria"],
            "n_passed_criteria": overall["n_passed_criteria"],
            "criterion_pass_rate": overall["criterion_pass_rate"],
        },
        "model_call": aggregate_model_call_metrics(scores),
        "by_macro_area": {},
        "by_difficulty": {},
    }

    for area in sorted({score.macro_area for score in scores}):
        summary["by_macro_area"][area] = aggregate(
            [score for score in scores if score.macro_area == area]
        )

    for difficulty in sorted({score.difficulty for score in scores}):
        summary["by_difficulty"][difficulty] = aggregate(
            [score for score in scores if score.difficulty == difficulty]
        )

    return summary
