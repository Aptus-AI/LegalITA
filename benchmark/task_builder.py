"""
Costruisce i task.json del benchmark a partire da Provvedimento + query generata.

Per ogni provvedimento:
1. Chiama il LLM per riformulare i principles in criteri PASS/FAIL pertinenti alla query
2. Assembla il BenchmarkTask
3. Scrive task.json su disco (senza source_ecli — non pubblicato)

Uso tipico:
    from benchmark.task_builder import TaskBuilder

    builder = TaskBuilder()
    task = builder.build(provvedimento, query)
    builder.save(task)
"""

import json
import logging
import re
import time
from pathlib import Path

import anthropic

from legal_ita.config import (
    BUILDER_TEMPERATURE,
    GENERATOR_MODEL,
    MAX_CRITERIA,
    MAX_SOURCE_PRINCIPLES,
    TASKS_DIR,
)
from legal_ita.schemas import BenchmarkTask, Criterion, Provvedimento
from legal_ita.taxonomy import normalize_macro_area

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt per riformulazione query + principles → criteri PASS/FAIL
# ---------------------------------------------------------------------------

CRITERIA_SYSTEM = """\
<role>
Sei un esperto di valutazione di sistemi di intelligenza artificiale
applicati alla ricerca giurisprudenziale italiana.
</role>

<task>
Devi costruire criteri binari PASS/FAIL per valutare se una risposta prodotta
da un sistema legale ricostruisce correttamente la soluzione documentata dalla
giurisprudenza di legittimità rispetto alla fattispecie posta nella query.
</task>

<constraint>
La query definisce il perimetro della valutazione. I principles della
decisione sorgente servono a determinare la risposta corretta, ma non
autorizzano a valutare profili che la query non richiede o che non sono
necessari per rispondervi.
</constraint>
"""


CRITERIA_TEMPLATE = """\
<objective>
Trasforma i principi giuridici contenuti in <principles> in criteri binari
PASS/FAIL per valutare la risposta alla domanda contenuta in <query>.

Il benchmark misura la capacità del sistema di individuare e ricostruire la
soluzione adottata dalla Corte di Cassazione o dalla giurisprudenza di
legittimità rispetto alla fattispecie concreta descritta nella query.

I criteri devono essere sentence-grounded, pertinenti alla query, autonomi e
non ridondanti.
</objective>

<query>
{query}
</query>

<rules>
  <rule id="1" name="perimetro_query">
  La query definisce il perimetro del task.

  Genera criteri soltanto per:
  - la soluzione giurisprudenziale espressamente richiesta dalla domanda;
  - i presupposti fattuali, processuali o probatori indispensabili per
    applicare tale soluzione;
  - le conseguenze giuridiche o processuali richieste dalla domanda.

  Scarta principi corretti ma laterali, accessori o non necessari per
  rispondere alla query.
  </rule>

  <rule id="2" name="sentence_grounded">
  Usa i principles per individuare il contenuto giuridico corretto documentato
  dalla decisione sorgente.

  Non richiedere alla risposta:
  - una panoramica completa della materia;
  - orientamenti ulteriori o contrastanti non presenti nei principles;
  - pronunce rappresentative non documentate;
  - dettagli della motivazione non necessari per rispondere alla domanda.
  </rule>

  <rule id="3" name="core_criteria_only">
  Genera esclusivamente criteri CORE.

  Un criterio è CORE solo se una risposta che lo omette non può considerarsi
  corretta o sufficientemente completa rispetto alla domanda posta.

  Non generare criteri meramente migliorativi, ornamentali o utili soltanto a
  ottenere una risposta più estesa.
  </rule>

  <rule id="4" name="numero_criteri">
  Genera da 1 a 4 criteri, senza preferenza per il numero massimo.

  - Usa 1 criterio quando la domanda richiede una sola soluzione giuridica
    essenziale.
  - Usa 2 criteri quando la domanda richiede una soluzione e un profilo
    applicativo autonomo, come un presupposto, un dies a quo, un onere
    probatorio, un criterio di quantificazione o una conseguenza processuale.
  - Usa 3 criteri quando sono richiesti tre elementi autonomi e necessari.
  - Usa 4 criteri soltanto quando la query richiede realmente quattro profili
    distinti, tutti indispensabili e non sovrapponibili.

  Non aggiungere criteri soltanto per rendere il task più difficile o più
  articolato.
  </rule>

  <rule id="5" name="no_duplicazioni">
  Non produrre due criteri che misurano lo stesso contenuto giuridico sotto
  formulazioni differenti.

  In particolare:
  - non creare un criterio di "percorso argomentativo" che si limiti a
    ricombinare elementi già valutati in criteri precedenti;
  - non creare un criterio denominato "errore da evitare" quando il medesimo
    errore è già incluso nella clausola FAIL di un altro criterio;
  - non separare artificialmente una regola dalla sua immediata negazione;
  - non duplicare lo stesso principio distinguendo tra enunciazione generale
    e applicazione alla fattispecie, salvo che la domanda richieda
    autonomamente entrambe.

  Verifica finale: se una singola affermazione corretta o errata della risposta
  determinerebbe da sola il PASS o il FAIL di due criteri, fondi tali criteri
  in uno solo.
  </rule>

  <rule id="6" name="atomicita">
  Ogni criterio deve valutare un solo elemento giuridico autonomo.

  È ammesso includere nel criterio il presupposto minimo necessario a
  comprenderne l'applicazione, ma non imporre una catena di passaggi
  argomentativi quando tali passaggi sono già coperti da altri criteri.
  </rule>

  <rule id="7" name="esito_determinato">
  Ogni criterio deve indicare chiaramente quale contenuto della risposta
  costituisce PASS e quale errore costituisce FAIL.

  Non produrre criteri che chiedano soltanto alla risposta di:
  - "affrontare la questione";
  - "discutere il tema";
  - "precisare se e in quale misura";
  - "valutare gli effetti";

  senza specificare quale soluzione corretta debba emergere secondo i
  principles.
  </rule>

  <rule id="8" name="specificita">
  Il criterio deve premiare la soluzione giurisprudenziale specifica richiesta
  dalla query, non una risposta genericamente corretta sull'area tematica.

  Una risposta che richiama principi generali ma omette il nodo decisivo
  documentato dalla fonte deve fallire.
  </rule>

  <rule id="9" name="completezza_proporzionata">
  Una risposta non deve fallire soltanto perché:
  - non usa lo stesso lessico della sentenza;
  - non riproduce tutti i passaggi della motivazione;
  - non menziona dettagli della fonte non richiesti dalla query;
  - non sviluppa profili aggiuntivi rispetto al nodo posto dalla domanda.

  Richiedi soltanto la completezza necessaria per rispondere professionalmente
  alla query.
  </rule>

  <rule id="10" name="citazioni">
  Non valutare la presenza, l'esattezza o la completezza delle citazioni di
  sentenze, numeri di ruolo o riferimenti identificativi della pronuncia,
  salvo che la query chieda espressamente di individuare o verificare una
  specifica fonte.

  La fedeltà delle citazioni può essere misurata separatamente e non deve
  essere introdotta implicitamente nei criteri di questo task.
  </rule>

  <rule id="11" name="no_dettagli_superflui">
  Non trasformare in criteri:
  - nomi delle parti;
  - numeri di sentenza;
  - importi;
  - date specifiche;
  - dimensioni del bene;
  - elementi puramente descrittivi;

  salvo che tali dettagli costituiscano il presupposto giuridicamente decisivo
  richiesto dalla domanda.
  </rule>

  <rule id="12" name="formato">
  Ogni criterio deve:
  - avere un titolo breve e descrittivo;
  - iniziare con "PASS se...";
  - contenere "FAIL se...";
  - essere valutabile leggendo soltanto la risposta del sistema;
  - descrivere il contenuto giuridico atteso senza copiare meccanicamente il
    wording dei principles.
  </rule>

  <internal_check>
  Prima di produrre l'output, verifica ciascun criterio:

  1. È richiesto espressamente dalla query o indispensabile per rispondervi?
  2. Il contenuto corretto è determinato dai principles?
  3. Misura un elemento autonomo e non già coperto da un altro criterio?
  4. Una risposta corretta e sintetica potrebbe fallirlo soltanto perché non
     ripete l'intero ragionamento della sentenza?
  5. La clausola FAIL non è già identica all'errore valutato da un altro
     criterio?

  Elimina ogni criterio che non superi le verifiche 1, 2 o 3.
  Semplifica o elimina ogni criterio che fallisca le verifiche 4 o 5.
  </internal_check>
</rules>

<examples>
  <good_example name="un_solo_criterio_sufficiente">
  <query_example>
  In una procedura di sovraindebitamento, il decreto di rigetto
  dell'esdebitazione viene prima comunicato nel solo dispositivo e poi
  trasmesso integralmente con la motivazione. Quale soluzione ha adottato la
  Cassazione sul momento da cui decorre il termine per il ricorso?
  </query_example>

  <correct_criteria>
  [
    {{
      "title": "Comunicazione idonea a far decorrere il termine",
      "match_criteria": "PASS se la risposta afferma che il termine per il ricorso decorre dalla comunicazione integrale del provvedimento motivato e non dalla sola comunicazione del dispositivo. FAIL se fa decorrere il termine dalla comunicazione del solo dispositivo o non distingue i due momenti comunicativi."
    }}
  ]
  </correct_criteria>

  <explanation>
  Non aggiungere un criterio separato sul doppio momento comunicativo o sulla
  tardività del ricorso: la distinzione decisiva è già integralmente valutata
  dal criterio principale, salvo che la query chieda espressamente anche una
  conseguenza ulteriore.
  </explanation>
  </good_example>

  <good_example name="due_profili_autonomi">
  <query_example>
  Un fondo espropriato è destinato a servizi pubblici e verde attrezzato; i
  proprietari contestano la stima fondata su terreni edificabili limitrofi.
  Quale soluzione ha adottato la Cassazione sulla determinazione
  dell'indennità in presenza di vincoli conformativi e sul metodo di stima
  utilizzabile dal giudice?
  </query_example>

  <correct_criteria>
  [
    {{
      "title": "Rilevanza delle utilizzazioni intermedie",
      "match_criteria": "PASS se la risposta riconosce che, pur esclusa l'edificabilità del fondo per effetto dei vincoli conformativi, l'indennità può tener conto di utilizzazioni intermedie concretamente consentite dalla destinazione urbanistica. FAIL se limita automaticamente la stima al solo valore agricolo o valorizza il fondo come edificabile."
    }},
    {{
      "title": "Scelta motivata del metodo di stima",
      "match_criteria": "PASS se la risposta chiarisce che il giudice può scegliere il metodo di stima adeguato alle caratteristiche del fondo e alla pianificazione vigente, motivando la scelta. FAIL se indica un metodo inderogabilmente obbligatorio o prescinde dalla destinazione urbanistica rilevante."
    }}
  ]
  </correct_criteria>
  </good_example>

  <good_example name="quattro_profili_realmente_distinti">
  <query_example>
  Una domanda chiede espressamente: l'impugnabilità di un provvedimento,
  l'ambito applicativo di una norma, il presupposto fattuale richiesto per
  applicarla e la conseguenza processuale del vizio motivazionale.
  </query_example>

  <explanation>
  In questo caso sono ammessi quattro criteri soltanto se ciascun profilo è
  richiesto dalla query e ciascuno può essere soddisfatto o violato
  indipendentemente dagli altri.
  </explanation>
  </good_example>

  <bad_example name="duplicazione_percorso">
  Non generare, dopo criteri già dedicati alla regola e alla conseguenza, un
  ulteriore criterio come:
  "PASS se la risposta ricostruisce il percorso logico composto dalla regola,
  dal presupposto e dalla conseguenza".
  Tale criterio duplica i precedenti.
  </bad_example>

  <bad_example name="duplicazione_errore_da_evitare">
  Non generare un criterio autonomo come:
  "PASS se la risposta non equipara automaticamente assenza di titolo e
  pericolo concreto",
  se un criterio precedente contiene già:
  "FAIL se la risposta ritiene sufficiente la sola assenza di titolo per
  dimostrare il pericolo concreto".
  </bad_example>

  <bad_example name="criterio_senza_gold_determinato">
  Non generare criteri del tipo:
  "PASS se la risposta affronta se e in quale misura i termini processuali
  siano influenzati dalla sospensione della pubblicazione".
  Il criterio è invalido se non specifica quale soluzione debba emergere per
  ottenere PASS.
  </bad_example>
</examples>

<principles>
{principles}
</principles>

<output_format>
Rispondi SOLO con un array JSON valido, senza testo aggiuntivo, markdown o
backtick:

[
  {{
    "title": "Titolo breve del criterio",
    "match_criteria": "PASS se... FAIL se..."
  }}
]
</output_format>
"""


# ---------------------------------------------------------------------------
# TaskBuilder
# ---------------------------------------------------------------------------

class TaskBuilder:
    """
    Assembla BenchmarkTask da Provvedimento + query generata.
    Gestisce la riformulazione query-aware dei principles via LLM e la scrittura su disco.
    """

    def __init__(
            self,
            model: str = GENERATOR_MODEL,
            tasks_dir: Path = TASKS_DIR,
            max_retries: int = 3,
            base_delay: float = 2.0,
    ):
        self.model = model
        self.tasks_dir = tasks_dir
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.client = anthropic.Anthropic()

        # contatore progressivo per macro_area
        self._counters: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
            self,
            provvedimento: Provvedimento,
            query: str,
    ) -> BenchmarkTask | None:
        """
        Costruisce un BenchmarkTask da Provvedimento + query.

        Returns:
            BenchmarkTask completo, o None se la riformulazione fallisce.
        """
        criteria = self._build_criteria(
            provvedimento.principles,
            query=query,
        )
        if not criteria:
            log.error(f"Criteri non generati per {provvedimento.ecli_id}")
            return None

        task_id = self._next_task_id(provvedimento.macro_area)

        return BenchmarkTask(
            task_id=task_id,
            query=query,
            macro_area=provvedimento.macro_area,
            criteria=criteria,
            source_ecli=provvedimento.ecli_id,  # tracciabilità interna
        )

    def save(self, task: BenchmarkTask) -> Path:
        """
        Scrive task.json su disco.
        Il source_ecli NON viene incluso nel file pubblico.

        Returns:
            Path del file scritto.
        """
        task_dir = self.tasks_dir / task.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        out_path = task_dir / "task.json"

        # serializza escludendo source_ecli
        data = task.model_dump(exclude={"source_ecli"})

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        log.debug(f"Scritto: {out_path}")
        return out_path

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _next_task_id(self, macro_area: str) -> str:
        """Genera un task_id progressivo per macro_area."""
        n = self._counters.get(macro_area, 0) + 1
        self._counters[macro_area] = n
        return f"{macro_area}/{n:04d}"

    def _build_criteria(
            self,
            principles: list[str],
            query: str,
    ) -> list[Criterion] | None:
        """
        Chiama il LLM per riformulare i principles in criteri PASS/FAIL
        pertinenti alla query generata.
        Una sola chiamata per tutti i principles del provvedimento.

        I principles vengono limitati a MAX_CRITERIA per mantenere
        omogenea la difficoltà dei task: con lo scoring all-pass, un task
        con molti criteri è strutturalmente più difficile di uno con pochi.
        """
        principles = principles[:MAX_SOURCE_PRINCIPLES]
        principles_text = "\n".join(
            f"{i + 1}. {p}" for i, p in enumerate(principles)
        )
        prompt = CRITERIA_TEMPLATE.format(
            query=query,
            principles=principles_text,
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    system=CRITERIA_SYSTEM,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=BUILDER_TEMPERATURE,
                )
                criteria = self._parse_criteria(response)
                if criteria:
                    return criteria
                log.warning(f"Parsing criteri fallito (tentativo {attempt})")

            except anthropic.RateLimitError:
                delay = self.base_delay * (2 ** (attempt - 1))
                log.warning(f"Rate limit — attendo {delay:.1f}s (tentativo {attempt})")
                time.sleep(delay)

            except anthropic.APIError as e:
                delay = self.base_delay * (2 ** (attempt - 1))
                log.warning(f"API error: {e} — attendo {delay:.1f}s (tentativo {attempt})")
                time.sleep(delay)

        return None

    def _parse_criteria(self, response) -> list[Criterion] | None:
        """Estrae e valida i criteri dalla risposta del LLM."""
        if not response.content:
            return None

        text = response.content[0].text.strip()

        # rimuove eventuali markdown code fences
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)

        try:
            items = json.loads(text)
        except json.JSONDecodeError as e:
            log.debug(f"JSON decode error: {e}\nTesto: {text[:200]}")
            return None

        if not isinstance(items, list) or not items:
            return None

        criteria = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            title = item.get("title", "").strip()
            match = item.get("match_criteria", "").strip()
            if not title or not match:
                continue
            criteria.append(Criterion(
                id=f"C-{len(criteria) + 1:03d}",
                title=title,
                match_criteria=match,
            ))

        # garanzia: mai più di MAX_CRITERIA criteri per task,
        # anche se l'LLM ne restituisce di più del previsto
        criteria = criteria[:MAX_CRITERIA]

        return criteria if criteria else None


# ---------------------------------------------------------------------------
# Batch build
# ---------------------------------------------------------------------------

def build_batch(
        pairs: list[tuple[Provvedimento, str]],
        builder: TaskBuilder | None = None,
        save: bool = True,
        delay_between: float = 0.5,
) -> list[BenchmarkTask]:
    """
    Costruisce task per una lista di coppie (Provvedimento, query).

    Args:
        pairs:          Output di generate_batch().
        builder:        Istanza di TaskBuilder. Se None ne crea una.
        save:           Se True scrive i task.json su disco.
        delay_between:  Pausa tra una chiamata API e l'altra.

    Returns:
        Lista di BenchmarkTask costruiti con successo.
    """
    if builder is None:
        builder = TaskBuilder()

    tasks = []
    n_failed = 0

    for i, (prov, query) in enumerate(pairs):
        task = builder.build(prov, query)

        if task is None:
            n_failed += 1
            continue

        if save:
            builder.save(task)

        tasks.append(task)

        if delay_between > 0 and i < len(pairs) - 1:
            time.sleep(delay_between)

    log.info(f"Task costruiti: {len(tasks)} ok, {n_failed} falliti")
    return tasks


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    from dotenv import load_dotenv

    from benchmark.corpus import load_corpus_by_area
    from benchmark.generator import QueryGenerator, generate_batch
    from legal_ita.config import RANDOM_SEED

    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Genera i task.json del benchmark.")
    parser.add_argument(
        "--n-per-area",
        type=int,
        default=50,
        help="Numero di provvedimenti da campionare per macro-area (default: 50)",
    )
    parser.add_argument(
        "--area",
        type=str,
        default=None,
        help="Limita a una macro-area (es. diritto_civile; alias legacy accettati)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Pausa in secondi tra chiamate API (default: 0.5)",
    )
    args = parser.parse_args()

    # carica corpus stratificato per macro-area
    corpus_by_area = load_corpus_by_area(seed=args.seed)

    if args.area:
        requested_area = normalize_macro_area(args.area, strict=True)
        if requested_area not in corpus_by_area:
            log.error(f"Macro-area non trovata: {args.area} -> {requested_area}")
            log.error(f"Disponibili: {sorted(corpus_by_area.keys())}")
            sys.exit(1)
        corpus_by_area = {requested_area: corpus_by_area[requested_area]}

    # campiona N provvedimenti per area
    import random

    rng = random.Random(args.seed)
    provvedimenti = []
    for area, records in corpus_by_area.items():
        n = min(args.n_per_area, len(records))
        sample = rng.sample(records, n)
        log.info(f"  {area:<30} {n:>4} provvedimenti campionati")
        provvedimenti.extend(sample)

    log.info(f"Totale provvedimenti: {len(provvedimenti)}")

    # genera query sintetiche
    log.info("Generazione query sintetiche...")
    generator = QueryGenerator()
    pairs = generate_batch(provvedimenti, generator=generator, delay_between=args.delay)
    log.info(f"Query generate: {len(pairs)}")

    # costruisce e salva i task.json
    log.info("Costruzione task.json...")
    builder = TaskBuilder()
    tasks = build_batch(pairs, builder=builder, save=True, delay_between=args.delay)

    log.info("=" * 50)
    log.info(f"Task scritti: {len(tasks)}")
    log.info(f"Output: {builder.tasks_dir}")
