"""
Generazione di query sintetiche a partire da un Provvedimento.

Il LLM generatore vede facts, macro_area e un sottoinsieme dei principles
della decisione sorgente. I principles servono
soltanto a individuare il nodo giuridico non banale da trasformare in domanda:
la query non deve copiarli né anticiparne la soluzione.

Il sistema sottoposto a benchmark vede esclusivamente la query finale, mai i
principles utilizzati durante la costruzione del dataset.

Uso tipico:
    from benchmark.generator import QueryGenerator

    gen = QueryGenerator()
    query = gen.generate(provvedimento)
"""

import logging
import time
from typing import Any

import anthropic

from legal_ita.config import (
    GENERATOR_MODEL,
    MAX_SOURCE_PRINCIPLES,
)
from legal_ita.schemas import Provvedimento

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Prompt v1.1 — ricerca giurisprudenziale sentence-grounded
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
<role>
Sei un avvocato italiano con esperienza in contenzioso civile, del lavoro,
tributario e responsabilità civile, specializzato nella formulazione di
quesiti professionali per sistemi di ricerca giurisprudenziale sulla Corte
di Cassazione.
</role>

<task>
Il tuo compito è formulare UNA domanda da porre a un sistema specializzato
nella ricerca di giurisprudenza di legittimità italiana.

La domanda deve mettere alla prova la capacità del sistema di individuare e
ricostruire la soluzione adottata dalla Corte di Cassazione rispetto a una
fattispecie concreta, non soltanto di formulare una risposta astratta di
diritto.
</task>

<definition>
Un buon quesito di ricerca giurisprudenziale:
- parte da una fattispecie concreta;
- chiede quale orientamento, soluzione o criterio applicativo emerga dalla
  giurisprudenza di legittimità rispetto a quel problema;
- conserva gli elementi fattuali, processuali o probatori che rendono utile
  la ricerca della fonte;
- non incorpora già la risposta corretta;
- non richiede una ricostruzione enciclopedica dell'intera materia.
</definition>

<constraint>
La difficoltà del quesito deve derivare fedelmente dai fatti e dai principi
della decisione sorgente. Non inventare circostanze, contrasti
giurisprudenziali, orientamenti alternativi o questioni ulteriori soltanto
per rendere la domanda più complessa.
</constraint>
"""


USER_TEMPLATE_PLAIN = """\
<objective>
Formula UNA sola domanda in materia di {macro_area} da porre a un sistema
specializzato nella ricerca di giurisprudenza di legittimità italiana.

La domanda deve riguardare il caso descritto in <case> e il nodo giuridico
documentato in <source_principles>. Deve chiedere quale soluzione, orientamento
o criterio applicativo abbia adottato la Corte di Cassazione rispetto a una
fattispecie analoga, senza anticipare la risposta attesa.
</objective>

<rules>
  <rule id="1" name="sentence_grounded">
  Il quesito deve richiedere di individuare la soluzione adottata dalla
  giurisprudenza di legittimità rispetto al problema concreto descritto nei
  fatti. Non formulare una mera domanda astratta di diritto.
  </rule>

  <rule id="2" name="standalone">
  La domanda deve essere comprensibile senza accesso alla decisione sorgente.
  Inserisci i fatti necessari a comprendere la fattispecie e il punto
  controverso.
  </rule>

  <rule id="3" name="fedelta_fattuale">
  Usa soltanto circostanze presenti in <case>. Non inventare fatti, condotte,
  eccezioni, norme, domande processuali, date o sequenze procedimentali.
  </rule>

  <rule id="4" name="uso_riservato_principles">
  Usa <source_principles> soltanto per individuare il nodo giuridico centrale.
  Non copiare la massima, non citare la decisione sorgente e non anticipare la
  soluzione nella domanda.
  </rule>

  <rule id="5" name="ricerca_non_panoramica">
  La domanda può chiedere quale soluzione abbia adottato la Cassazione o quali
  elementi abbia ritenuto decisivi.
  Non chiedere panoramiche complete, contrasti, orientamenti minoritari o
  pronunce rappresentative, salvo che risultino esplicitamente documentati nei
  principles.
  </rule>

  <rule id="6" name="source_sensitive">
  Conserva almeno un fatto, snodo processuale, dato probatorio, elemento
  temporale o distinzione normativa specifica senza il quale la domanda si
  ridurrebbe a legal QA generale.
  </rule>

  <rule id="7" name="non_leading">
  Non inserire nella domanda la formula tecnica che costituisce la soluzione.
  Non formulare alternative in cui la risposta corretta sia manifestamente
  suggerita.
  </rule>

  <rule id="8" name="no_legal_qa_binario">
  Evita il formato "vale X oppure Y?" quando il quesito si risolverebbe nella
  scelta dell'alternativa più intuitiva.
  </rule>

  <rule id="9" name="singolo_nodo">
  Concentrati su un solo nodo giuridico centrale. Inserisci un secondo profilo
  soltanto se necessario per risolvere la fattispecie e supportato dai
  principles.
  </rule>

  <rule id="10" name="macro_area">
  La questione deve appartenere in modo centrale alla macro-area indicata.
  </rule>

  <rule id="11" name="registro">
  Usa un registro professionale forense, tecnico-pratico e naturale.
  Evita panoramiche generali e formulazioni accademiche.
  </rule>

  <rule id="12" name="privacy">
  Non usare placeholder tra parentesi quadre; usa soggetti generici se
  necessario.
  </rule>

  <rule id="13" name="non_autoreferenziale">
  Non fare riferimento al testo sorgente, al documento o al caso descritto.
  </rule>

  <rule id="14" name="forma">
  Scrivi una domanda articolata in una o due frasi.
  Mira a 70-110 parole; puoi arrivare a 150 parole soltanto se indispensabile
  per preservare gli elementi specifici che rendono utile la ricerca
  giurisprudenziale.
  Termina con un punto interrogativo e non aggiungere spiegazioni.
  </rule>

  <internal_check>
  Verifica internamente che:
  - la domanda richieda una soluzione della giurisprudenza di legittimità;
  - il nodo sia supportato dai principles;
  - la soluzione non sia già anticipata nella domanda;
  - almeno un fatto specifico del caso sia necessario per rispondere;
  - il tema appartenga alla macro-area indicata.
  Se necessario, riformula prima di produrre l'output.
  </internal_check>
</rules>

<case>
{facts}
</case>

<source_principles private="true">
I seguenti principi derivano dalla decisione sorgente e servono soltanto a
individuare il nodo giuridico documentato da trasformare in domanda.

Non citarli, non copiarli, non anticiparne la soluzione e non inferire
orientamenti ulteriori non espressamente supportati dal loro contenuto.

{principles}
</source_principles>

<output_format>
Rispondi con la sola domanda, senza prefissi, numerazione, spiegazioni o
virgolette.
</output_format>
"""


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class QueryGenerator:
    """
    Genera una query sintetica per ogni Provvedimento.

    Usa l'API Anthropic con retry e backoff esponenziale.
    """

    def __init__(
        self,
        model: str = GENERATOR_MODEL,
        max_retries: int = 3,
        base_delay: float = 2.0,
    ):
        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.client = anthropic.Anthropic()

    def _build_prompt(self, provvedimento: Provvedimento) -> str:
        """
        Costruisce il prompt ancorando la query ai facts e al nodo giuridico
        ricavabile dai principles della decisione sorgente.

        I principles sono visibili solo al generatore del dataset; non vengono
        mai forniti ai sistemi sottoposti al benchmark.
        """
        principles = provvedimento.principles[:MAX_SOURCE_PRINCIPLES]
        principles_text = "\n".join(f"- {principle}" for principle in principles)

        if not principles_text:
            principles_text = (
                "(nessun principio disponibile: preserva soltanto i fatti "
                "giuridicamente decisivi espliciti)"
            )

        return USER_TEMPLATE_PLAIN.format(
            macro_area=provvedimento.macro_area,
            facts=provvedimento.facts,
            principles=principles_text,
        )

    # ------------------------------------------------------------------
    # Generazione
    # ------------------------------------------------------------------

    def generate(
        self,
        provvedimento: Provvedimento,
    ) -> str | None:
        """
        Genera una query sintetica dal provvedimento.

        Args:
            provvedimento: Il provvedimento sorgente; usa facts, macro_area
                           e principles come ancora privata di generazione.

        Returns:
            La query generata come stringa, o None se tutti i retry falliscono.
        """
        prompt = self._build_prompt(provvedimento)

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=400,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )

                query = self._parse(response)
                if query:
                    return query

                log.warning(f"Risposta non valida (tentativo {attempt})")

            except anthropic.RateLimitError:
                delay = self.base_delay * (2 ** (attempt - 1))
                log.warning(
                    f"Rate limit — attendo {delay:.1f}s "
                    f"(tentativo {attempt})"
                )
                time.sleep(delay)

            except anthropic.APIError as error:
                delay = self.base_delay * (2 ** (attempt - 1))
                log.warning(
                    f"API error: {error} — attendo {delay:.1f}s "
                    f"(tentativo {attempt})"
                )
                time.sleep(delay)

        log.error(
            f"Generazione fallita dopo {self.max_retries} tentativi: "
            f"{provvedimento.ecli_id}"
        )
        return None

    def _parse(self, response: Any) -> str | None:
        """Estrae e valida in modo minimale la query dalla risposta del modello."""
        if not response.content:
            return None

        text = response.content[0].text.strip()

        # Rimuove eventuali prefissi tipo "Domanda:" o "Query:".
        for prefix in ("Domanda:", "Query:", "D:", "Q:"):
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

        # Normalizza eventuali a-capo prodotti dal modello.
        text = " ".join(text.split())

        if len(text) < 15:
            return None

        n_words = len(text.split())
        if n_words > 150:
            log.warning(
                f"Query scartata: {n_words} parole, limite massimo 150"
            )
            return None

        return text


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

def generate_batch(
    provvedimenti: list[Provvedimento],
    generator: QueryGenerator | None = None,
    delay_between: float = 0.5,
) -> list[tuple[Provvedimento, str]]:
    """
    Genera query per una lista di provvedimenti.

    Args:
        provvedimenti: Lista di provvedimenti da processare.
        generator: Istanza di QueryGenerator. Se None ne crea una.
        delay_between: Pausa in secondi tra una chiamata e l'altra
                       per evitare rate limit.

    Returns:
        Lista di coppie (Provvedimento, query).
        I provvedimenti per cui la generazione fallisce vengono saltati.
    """
    if generator is None:
        generator = QueryGenerator()

    results: list[tuple[Provvedimento, str]] = []
    n_failed = 0

    for index, provvedimento in enumerate(provvedimenti):
        query = generator.generate(provvedimento)

        if query is None:
            n_failed += 1
            continue

        results.append((provvedimento, query))

        if delay_between > 0 and index < len(provvedimenti) - 1:
            time.sleep(delay_between)

    log.info(
        f"Generazione completata: {len(results)} ok, {n_failed} falliti"
    )
    return results
