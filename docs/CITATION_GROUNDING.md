# Citation Grounding and Structural Gold v3

This document explains how the LegalITA citation-grounding and Structural Gold
v3 pipelines work, and why their implementation is not part of the public
repository. The public distribution runs the benchmark **only with citation
grounding disabled** (`--skip-citation-grounding`).

## What citation grounding is

Citation grounding is independent from legal reasoning evaluation. The judge
score answers the question "does the answer satisfy the legal PASS/FAIL
criteria of the task?"; citation grounding answers a different question: "do
the rulings cited in the answer actually exist, and are they the rulings the
task requires?"

Existence and relevance are two separate gates:

- **Existence**: does the citation identify a real, verifiable judicial
  decision?
- **Relevance**: is the identified decision one of the rulings required or
  accepted by the gold annotation of that task?

A ruling can exist and still be out of gold (it fails relevance); a citation
can match the gold identity while its existence remains unresolved. When the
ambiguity could change the result, the task-level citation verdict stays
`unresolved` rather than guessing.

## How the pipeline works

```text
model answer
-> LLM citation extractor (DeepSeek via Novita)
-> citation / ECLI normalization
-> Pinecone identity and metadata resolution
-> read-only S3 verification on structured court-ruling documents
-> grounding report per model
```

1. An LLM extractor parses the model answer and emits every jurisprudential
   citation in a normalized structured form (court, section, number, year,
   ECLI when derivable).
2. Each citation is resolved against dedicated Pinecone indexes by identity
   and metadata, not by semantic similarity: a semantic match may point to a
   thematically similar but different ruling, so it is never accepted as proof
   of identity.
3. For Gold v3, resolved candidates are verified against structured court
   ruling documents stored on S3 (read-only), which provide the authoritative
   text and metadata.
4. Every citation ends in an explicit status
   (`resolved_pinecone_exact`, `ambiguous_pinecone`, `not_found_in_index`,
   `suspected_fabricated`, `confirmed_fabricated`, …). Only
   `confirmed_fabricated` — fabrication proven inside the Pinecone/S3
   perimeter — produces a grounding hard fail. Everything unresolved stays
   diagnostic.

## What Structural Gold v3 is

Structural Gold v3 is the gold-builder pipeline that produced the reference
annotations. For each task it reconstructs the jurisprudential state of the
legal question and returns an issue-state structural classification supported
by retrieved case-law evidence:

```text
task query
-> Pinecone query retrieval on facts/principles indexes
-> ECLI deduplication
-> S3 hydration of structured ruling documents
-> controlled LLM semantic assessment
-> orientation grouping
-> structural classification with provenance
```

The retrieved rulings are evidence examples used to infer the state of the
question; the resulting gold object carries citations, provenance and
retrieval-coverage metadata for audit.

## Why it is not replicable

Both pipelines depend on infrastructure and data that cannot be redistributed:

- **Private Pinecone indexes** (facts and principles) built over a proprietary
  structured corpus of Italian Cassation rulings. Rebuilding them requires
  both the source corpus and the ingestion pipeline, neither of which is
  publicly distributable.
- **Structured S3 documents**: the authoritative, machine-readable versions of
  the rulings live in a private S3 bucket, produced by a proprietary document
  structuring platform.
- **Deployment credentials and naming**: index names, bucket names, prefixes,
  AWS profiles and API keys are deployment-specific secrets.
- **Corpus licensing**: the underlying case-law corpus cannot be re-published
  as part of an open-source repository.

Because a public copy of the code could never run — and to keep the public
surface aligned with what is actually reproducible — the implementation
(`evaluation/citations/`, `evaluation/grounding_metrics.py`, `gold_builder/`,
the `structural_v3` scripts) is not distributed on GitHub. The scoring
pipeline in this repository degrades gracefully: with citation grounding
disabled, citation fields are reported as not applicable and the judge-based
reasoning evaluation is unaffected.

## What remains reproducible

With the 107-task bundle and ordinary model/judge API keys:

```bash
python run_benchmark.py --models gpt-4o --skip-citation-grounding
python run_bullshit_v2.py --models gpt-4o
```

---

# Citation Grounding e Structural Gold v3 (italiano)

Questo documento spiega come funzionano le pipeline di citation grounding e
Structural Gold v3 di LegalITA, e perché la loro implementazione non fa parte
del repository pubblico. La distribuzione pubblica esegue il benchmark **solo
con citation grounding disattivato** (`--skip-citation-grounding`).

## Che cos'è il citation grounding

Il citation grounding è indipendente dalla valutazione del ragionamento
giuridico. Lo score del judge risponde alla domanda "la risposta soddisfa i
criteri PASS/FAIL del task?"; il citation grounding risponde a una domanda
diversa: "le sentenze citate nella risposta esistono davvero, e sono quelle
richieste dal task?"

Esistenza e pertinenza sono due gate separati:

- **Esistenza**: la citazione identifica una decisione giudiziaria reale e
  verificabile?
- **Pertinenza**: la decisione identificata è una delle sentenze richieste o
  accettate dal gold di quel task?

Una sentenza può esistere ed essere fuori gold (fallisce la pertinenza); una
citazione può coincidere con l'identità del gold mentre la sua esistenza resta
non risolta. Quando l'ambiguità può cambiare il risultato, il verdetto
citazionale del task resta `unresolved` invece di tirare a indovinare.

## Come funziona la pipeline

```text
risposta del modello
-> estrattore LLM delle citazioni (DeepSeek via Novita)
-> normalizzazione citazioni / ECLI
-> risoluzione di identità e metadati su Pinecone
-> verifica read-only su documenti strutturati S3
-> report di grounding per modello
```

1. Un estrattore LLM analizza la risposta e produce ogni citazione
   giurisprudenziale in forma strutturata normalizzata (corte, sezione,
   numero, anno, ECLI quando derivabile).
2. Ogni citazione viene risolta su indici Pinecone dedicati per identità e
   metadati, non per similarità semantica: un match semantico può puntare a
   una sentenza simile per tema ma diversa, quindi non è mai accettato come
   prova di identità.
3. Per il Gold v3, i candidati risolti vengono verificati sui documenti
   strutturati delle sentenze conservati su S3 (in sola lettura), che
   forniscono testo e metadati autoritativi.
4. Ogni citazione termina in uno status esplicito
   (`resolved_pinecone_exact`, `ambiguous_pinecone`, `not_found_in_index`,
   `suspected_fabricated`, `confirmed_fabricated`, …). Solo
   `confirmed_fabricated` — fabbricazione provata dentro il perimetro
   Pinecone/S3 — produce un hard fail di grounding. Tutto ciò che resta non
   risolto rimane diagnostico.

## Che cos'è Structural Gold v3

Structural Gold v3 è la pipeline di gold building che ha prodotto le
annotazioni di riferimento. Per ogni task ricostruisce lo stato
giurisprudenziale della questione giuridica e restituisce una classificazione
strutturale sostenuta da evidenze giurisprudenziali recuperate:

```text
query del task
-> retrieval Pinecone su indici facts/principles
-> deduplicazione ECLI
-> idratazione S3 dei documenti strutturati
-> valutazione semantica LLM controllata
-> raggruppamento per orientamenti
-> classificazione strutturale con provenance
```

Le sentenze recuperate sono esempi di evidenza usati per inferire lo stato
della questione; l'oggetto gold risultante conserva citazioni, provenance e
metadati di copertura del retrieval a fini di audit.

## Perché non è replicabile

Entrambe le pipeline dipendono da infrastruttura e dati non redistribuibili:

- **Indici Pinecone privati** (facts e principles) costruiti su un corpus
  strutturato proprietario di sentenze di Cassazione. Ricostruirli richiede
  sia il corpus sorgente sia la pipeline di ingestione, nessuno dei due
  distribuibile pubblicamente.
- **Documenti strutturati S3**: le versioni autoritative e machine-readable
  delle sentenze risiedono in un bucket S3 privato, prodotte da una
  piattaforma proprietaria di strutturazione documentale.
- **Credenziali e naming di deployment**: nomi degli indici, bucket, prefissi,
  profili AWS e API key sono segreti specifici del deployment.
- **Licenze del corpus**: il corpus giurisprudenziale sottostante non può
  essere ripubblicato in un repository open source.

Poiché una copia pubblica del codice non potrebbe comunque essere eseguita — e
per mantenere la superficie pubblica allineata a ciò che è davvero
riproducibile — l'implementazione (`evaluation/citations/`,
`evaluation/grounding_metrics.py`, `gold_builder/`, gli script
`structural_v3`) non è distribuita su GitHub. La pipeline di scoring di questo
repository degrada in modo controllato: con citation grounding disattivato i
campi citazionali risultano non applicabili e la valutazione del ragionamento
tramite judge non è influenzata.

## Cosa resta riproducibile

Con il pacchetto dei 107 task e normali API key di modello e judge:

```bash
python run_benchmark.py --models gpt-4o --skip-citation-grounding
python run_bullshit_v2.py --models gpt-4o
```
