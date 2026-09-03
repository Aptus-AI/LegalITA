# LegalITA

LegalITA is a benchmark for evaluating language models on Italian case-law
reasoning tasks. It includes criterion-based legal evaluation, adversarial
false-premise evaluation and a reproducible local citation-grounding pipeline.

Citation grounding runs entirely on local files: a versioned SQLite registry
of ECLI identifiers and one question profile per task, supplied as a separate
data bundle. It requires no access to Aptus systems.

## Getting oriented: what you can reproduce today

A fresh clone gives you the LegalITA code, but not the benchmark tasks. The 107
task definitions are distributed as a separate ZIP bundle, available on
request, so that the codebase remains small and the evaluation material can be
versioned as a single artifact.

The bundle is distributed on request. Ask for it by e-mail, with subject
"tasks legalITA", to:

```text
jacopo.grandi@aptus.ai
artenis@aptus.ai
```

SHA-256 of `legalita_tasks_107.zip`:

```text
806c6bef773007606a0841b298a3f8de00a793a695d3e4f6a566d3b15881f6eb
```

The bundle contains two evaluation sets:

- 67 case-law reasoning tasks, each stored as an individual `task.json`;
- 40 missing-document detection tasks, stored in
  `bullshit_tasks_v3_missing_documents_40.json`.

The filenames and task identifiers in the bundle are already normalized. Do
not rename the task folders or edit the `task_id` fields. After extracting the
archive, place the files in the repository as follows:

```text
LegalITA/
├── tasks/
│   ├── diritto_civile/<task-number>/task.json
│   ├── tributario/<task-number>/task.json
│   └── lavoro/<task-number>/task.json
└── data/
    └── private/
        └── bullshit_tasks_v3_missing_documents_40.json
```

If the archive is named `legalita_tasks_107.zip` and has been extracted into a
directory with the same name, the placement can be performed from the project
root with:

```bash
mkdir -p tasks data/private
cp -R legalita_tasks_107/tasks/. tasks/
cp legalita_tasks_107/data/private/bullshit_tasks_v3_missing_documents_40.json data/private/
```

The `data/private` name is a runtime convention inherited from the project
layout. It means that the file is kept outside the Git repository; it does not
mean that the separately distributed 40-task JSON is secret or unavailable for
benchmark use.

With the bundle in place, two parts of LegalITA are reproducible with ordinary
model and judge API credentials:

```bash
# Run the 67-task case-law reasoning benchmark (legal scoring only).
# Without --skip-citation-grounding the offline citation grounding also runs
# at the end and needs the grounding bundle described below.
python run_benchmark.py --models gpt-4o --skip-citation-grounding

# Run the 40 missing-document detection tasks.
python run_bullshit_v2.py --models gpt-4o
```

The first command measures whether a model's answer satisfies the legal
PASS/FAIL criteria attached to each reasoning task. The second measures whether
the model notices that documents invoked by the question were not supplied,
instead of inventing their contents or giving document-specific advice.

These runs still require API keys for the evaluated model and the configured
judge or judges. With the grounding bundle in place, drop
`--skip-citation-grounding` and the same command also computes GOG and
Coverage at the end of the run; see the Citation Grounding section below and
[docs/CITATION_GROUNDING.md](docs/CITATION_GROUNDING.md).

## Guida introduttiva: cosa puoi riprodurre oggi

Un clone appena scaricato contiene il codice di LegalITA, ma non i task del
benchmark. Le 107 definizioni dei task vengono distribuite in un pacchetto ZIP
separato, disponibile su richiesta: in questo modo il repository resta leggero
e il materiale di valutazione può essere versionato come un unico artefatto.

Il pacchetto si richiede via e-mail, con oggetto "tasks legalITA", a:

```text
jacopo.grandi@aptus.ai
artenis@aptus.ai
```

SHA-256 di `legalita_tasks_107.zip`:

```text
806c6bef773007606a0841b298a3f8de00a793a695d3e4f6a566d3b15881f6eb
```

Il pacchetto contiene due insiemi di valutazione:

- 67 task di ragionamento giurisprudenziale, ciascuno salvato in un proprio
  `task.json`;
- 40 task di rilevazione di documenti mancanti, raccolti nel file
  `bullshit_tasks_v3_missing_documents_40.json`.

I nomi delle cartelle e gli identificativi dei task sono già normalizzati. Non
occorre rinominare nulla né modificare i campi `task_id`. Dopo aver estratto lo
ZIP, i file devono trovarsi in questa posizione:

```text
LegalITA/
├── tasks/
│   ├── diritto_civile/<numero-task>/task.json
│   ├── tributario/<numero-task>/task.json
│   └── lavoro/<numero-task>/task.json
└── data/
    └── private/
        └── bullshit_tasks_v3_missing_documents_40.json
```

Se l'archivio si chiama `legalita_tasks_107.zip` ed è stato estratto in una
cartella omonima, dalla root del progetto si possono collocare i file con:

```bash
mkdir -p tasks data/private
cp -R legalita_tasks_107/tasks/. tasks/
cp legalita_tasks_107/data/private/bullshit_tasks_v3_missing_documents_40.json data/private/
```

Il nome `data/private` è una convenzione ereditata dalla struttura del runtime:
indica che il file resta fuori dal repository Git. Non significa che il JSON dei
40 task, distribuito separatamente, sia segreto o indisponibile per l'uso nel
benchmark.

Una volta collocati i file, due componenti di LegalITA sono riproducibili con le
normali credenziali API del modello e dei judge:

```bash
# Esegue i 67 task di ragionamento (solo scoring giuridico).
# Senza --skip-citation-grounding, al termine parte anche il citation grounding
# offline, che richiede il bundle descritto più avanti.
python run_benchmark.py --models gpt-4o --skip-citation-grounding

# Esegue i 40 task di rilevazione dei documenti mancanti.
python run_bullshit_v2.py --models gpt-4o
```

Il primo comando misura se la risposta del modello soddisfa i criteri giuridici
PASS/FAIL associati a ciascun task. Il secondo controlla se il modello riconosce
che i documenti richiamati dalla domanda non sono stati forniti, invece di
inventarne il contenuto o proporre una strategia fondata su atti che non ha
potuto leggere.

Le due esecuzioni richiedono comunque le chiavi API del modello valutato e dei
judge configurati. Con il bundle di grounding installato, togliendo
`--skip-citation-grounding` lo stesso comando calcola anche GOG e Coverage al
termine della run; vedere la sezione Citation Grounding più avanti e
[docs/CITATION_GROUNDING.md](docs/CITATION_GROUNDING.md).

## Setup

```bash
pip install -e .
```

The editable install exposes stable command-line entry points. The historical
root scripts remain as compatibility shims, so existing commands continue to
work while new integrations can use `legalita-benchmark`,
`legalita-grounding`, `legalita-score-csv`, and the other `legalita-*`
commands.

Pipelines that call external providers need a local `.env` file in the project
root:

```text
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
```

Il grounding usa soltanto `OPENAI_API_KEY`, per l'estrazione delle citazioni
(evitabile con `--fast-path`); registry e profili sono file locali.

## Adaptive 2-of-3 Judge

The standard benchmark can evaluate each binary PASS/FAIL criterion with an
adaptive three-judge majority.

Default strategy:

```text
JUDGE_STRATEGY=adaptive_majority

JUDGE_A_PROVIDER=anthropic
JUDGE_A_MODEL=claude-sonnet-4-6

JUDGE_B_PROVIDER=openai
JUDGE_B_MODEL=gpt-5.5

JUDGE_C_PROVIDER=anthropic
JUDGE_C_MODEL=claude-opus-4-8
```

Workflow:

1. Judge A and Judge B receive the same input in parallel: task question, model
   answer, criterion title, PASS/FAIL definition, and precomputed citation
   grounding context.
2. If A and B agree, their verdict is final and Judge C is not called.
3. If A and B disagree, or if one of them has a technical failure, Judge C
   evaluates the same material independently.
4. A final `pass` or `fail` is emitted only when at least two valid votes agree.
   API errors, timeouts, and parsing failures become `unresolved`, not automatic
   FAIL verdicts.

Temporary single-judge mode:

```text
JUDGE_STRATEGY=single
JUDGE_A_PROVIDER=anthropic
JUDGE_A_MODEL=claude-sonnet-4-6
```

Example run:

```bash
python run_benchmark.py --models gpt-4o --limit 1 --skip-citation-grounding \
  --judge-strategy adaptive_majority \
  --judge-a-provider anthropic --judge-a-model claude-sonnet-4-6 \
  --judge-b-provider openai --judge-b-model gpt-5.5 \
  --judge-c-provider anthropic --judge-c-model claude-opus-4-8
```

The theoretical judge-call cost is:

```text
calls = 2N + D
```

where `N` is the number of criteria and `D` is the number of disagreements or
technical recoveries that require Judge C.

An `unresolved` criterion does not count as PASS or FAIL. If a required
criterion is unresolved, the task is marked with `scoring_status="incomplete"`,
`score=null`, and `all_pass=null`. Incomplete tasks are excluded from all-pass
and criterion pass-rate denominators; `unresolved_rate` is reported separately.

The same A/B/C judge logic is used by the adversarial bullshit v2 module. In
that module, A and B evaluate the whole task, and C is called once only if at
least one criterion needs tie-break or recovery.

Smoke run for one bullshit task:

```bash
python run_bullshit_v2.py --models gpt-4o --limit 1 \
  --judge-strategy adaptive_majority \
  --judge-a-provider anthropic --judge-a-model claude-sonnet-4-6 \
  --judge-b-provider openai --judge-b-model gpt-5.5 \
  --judge-c-provider anthropic --judge-c-model claude-opus-4-8
```

## Citation Grounding — now available in the open release

Citation grounding runs entirely on local files. The registry of ECLI
identifiers and the 67 per-task gold profiles used for the comparison are
already built and are distributed as a single ZIP bundle, on request, so that
nobody has to rebuild them. Ask for it by e-mail, with subject
"grounding bundle legalITA", to the addresses listed above.

SHA-256 of `legalita_grounding_bundle.zip` (about 40 MB compressed, 190 MB
extracted):

```text
28a99ea160e68102d1dd8566056856e430f6c35e7f46bf3b22f3fe080ae64f38
```

### What the bundle contains

```text
legalita-grounding-bundle/
├── manifest.json                     snapshot date, row counts, checksums
├── registry/
│   └── ecli_registry_v1.sqlite       read-only registry of ECLI identifiers
└── question_profiles/
    ├── index.json
    └── <macro-area>_<task>.json      67 gold profiles, one per task
```

### Where to put it

The runtime looks for the bundle in `data/citation_pool/`, at the project
root (the same directory that contains `pyproject.toml`). Do not rename the
files or the folders. From the project root:

```bash
unzip legalita_grounding_bundle.zip -d /tmp/legalita-bundle
mkdir -p data/citation_pool
cp -R /tmp/legalita-bundle/legalita-grounding-bundle/. data/citation_pool/
```

The result must be exactly:

```text
LegalITA/
└── data/
    └── citation_pool/
        ├── manifest.json
        ├── registry/ecli_registry_v1.sqlite
        └── question_profiles/index.json  (+ 67 task files)
```

`data/` is ignored by Git, so the bundle never ends up in a commit. If you
keep the bundle elsewhere, point the runtime to it with
`--citation-registry` / `--question-profiles`, or with the environment
variables `LEGALITA_CITATION_REGISTRY_PATH` and
`LEGALITA_QUESTION_PROFILES_DIR`.

### Check the installation

```bash
shasum -a 256 data/citation_pool/registry/ecli_registry_v1.sqlite
# must match "registry/ecli_registry_v1.sqlite" in data/citation_pool/manifest.json

legalita-grounding --results results/<provider>/<run> --backend local --fast-path
```

The second command must print a `GOG=... Coverage=... backend=local
registry=<snapshot date>` line. If it prints `Registry locale non trovato` or
`Question profiles non trovati`, the bundle is not in the expected location.

### Run it

With the bundle in place, `legalita-benchmark` runs the offline grounding by
default at the end of the scoring and adds `gog`, `coverage`, `gog_backend`
and `registry_built_at` to the run's `summary.json`; the detailed report is
written next to it as `citation_grounding_v3.{json,md}`. The bundle is checked
before any API call. Pass `--skip-citation-grounding` for legal scoring only:

```bash
legalita-benchmark --models gpt-4o                             # scoring + grounding
legalita-benchmark --models gpt-4o --skip-citation-grounding   # scoring only
```

Grounding can also be run on its own, on a completed run (a directory
containing `scores.json` or `outputs.json`) or on a CSV of answers from an
external system (matched by `task_id`, or by question text when the column is
missing):

```bash
legalita-grounding --results results/<provider>/<run> --backend local
legalita-grounding --csv answers.csv --model SYSTEM_NAME --backend local
```

The standalone report is written to
`results/grounding-offline/<run>/citation_grounding_v3.{json,md}`. The only
network call is the citation extractor (`OPENAI_API_KEY`); `--fast-path` skips
it and evaluates explicit ECLI identifiers only. Use `--task-ids` for a quick
trial on a few tasks. GOG and Coverage are averaged over the tasks present in
the input (or in the run, inside `legalita-benchmark`); pass `--n-tasks 67` to
compare with the full benchmark, counting missing tasks as zero. Reports are
comparable only when produced with the same bundle: the snapshot date is recorded in every report. Statuses, snapshot
limits and troubleshooting are documented in
[docs/CITATION_GROUNDING.md](docs/CITATION_GROUNDING.md).

## Citation Grounding — ora disponibile nella release aperta

Il citation grounding lavora interamente su file locali. Il registry degli
identificativi ECLI e i 67 profili gold per task usati nel confronto sono già
costruiti e vengono distribuiti in un unico pacchetto ZIP, su richiesta, così
che nessuno debba ricostruirli. Si richiede via e-mail, con oggetto
"grounding bundle legalITA", agli indirizzi indicati sopra.

SHA-256 di `legalita_grounding_bundle.zip` (circa 40 MB compresso, 190 MB
estratto):

```text
28a99ea160e68102d1dd8566056856e430f6c35e7f46bf3b22f3fe080ae64f38
```

### Contenuto del pacchetto

```text
legalita-grounding-bundle/
├── manifest.json                     data dello snapshot, conteggi, checksum
├── registry/
│   └── ecli_registry_v1.sqlite       registry in sola lettura degli ECLI
└── question_profiles/
    ├── index.json
    └── <macro-area>_<task>.json      67 profili gold, uno per task
```

### Dove va posizionato

Il runtime cerca il pacchetto in `data/citation_pool/`, nella root del
progetto (la stessa cartella che contiene `pyproject.toml`). Non rinominare
file né cartelle. Dalla root del progetto:

```bash
unzip legalita_grounding_bundle.zip -d /tmp/legalita-bundle
mkdir -p data/citation_pool
cp -R /tmp/legalita-bundle/legalita-grounding-bundle/. data/citation_pool/
```

Il risultato deve essere esattamente:

```text
LegalITA/
└── data/
    └── citation_pool/
        ├── manifest.json
        ├── registry/ecli_registry_v1.sqlite
        └── question_profiles/index.json  (+ 67 file dei task)
```

`data/` è ignorata da Git, quindi il pacchetto non finisce mai in un commit.
Se preferisci tenerlo altrove, indica il percorso con `--citation-registry` /
`--question-profiles`, oppure con le variabili d'ambiente
`LEGALITA_CITATION_REGISTRY_PATH` e `LEGALITA_QUESTION_PROFILES_DIR`.

### Verifica dell'installazione

```bash
shasum -a 256 data/citation_pool/registry/ecli_registry_v1.sqlite
# deve coincidere con "registry/ecli_registry_v1.sqlite" in data/citation_pool/manifest.json

legalita-grounding --results results/<provider>/<run> --backend local --fast-path
```

Il secondo comando deve stampare una riga `GOG=... Coverage=... backend=local
registry=<data snapshot>`. Se stampa `Registry locale non trovato` o
`Question profiles non trovati`, il pacchetto non è nella posizione attesa.

### Esecuzione

Con il bundle installato, `legalita-benchmark` esegue il grounding offline di
default al termine dello scoring e aggiunge `gog`, `coverage`, `gog_backend` e
`registry_built_at` al `summary.json` della run; il report dettagliato viene
scritto accanto, come `citation_grounding_v3.{json,md}`. Il bundle viene
verificato prima di qualsiasi chiamata API. Con `--skip-citation-grounding` si
esegue il solo scoring giuridico:

```bash
legalita-benchmark --models gpt-4o                             # scoring + grounding
legalita-benchmark --models gpt-4o --skip-citation-grounding   # solo scoring
```

Il grounding si può anche eseguire da solo, su una run già completata
(directory con `scores.json` o `outputs.json`) oppure su un CSV di risposte di
un sistema esterno (associate per `task_id`, o per testo della domanda se la
colonna manca):

```bash
legalita-grounding --results results/<provider>/<run> --backend local
legalita-grounding --csv risposte.csv --model NOME_SISTEMA --backend local
```

Il report dell'esecuzione autonoma viene scritto in
`results/grounding-offline/<run>/citation_grounding_v3.{json,md}`. L'unica
chiamata di rete è l'estrattore di citazioni (`OPENAI_API_KEY`); `--fast-path`
la evita e valuta solo gli ECLI espliciti. `--task-ids` permette una prova
rapida su pochi task. GOG e Coverage sono medie sui task presenti nell'input
(o nella run, dentro `legalita-benchmark`); con `--n-tasks 67` si confronta
con il benchmark completo, contando zero i task assenti. I report sono
confrontabili solo se prodotti con lo stesso pacchetto: la data dello snapshot è registrata in ogni report. Stati,
limiti dello snapshot e problemi comuni sono documentati in
[docs/CITATION_GROUNDING.md](docs/CITATION_GROUNDING.md).

## Canonical Taxonomy

Runtime taxonomy uses seven canonical macro-areas:

```text
diritto_civile
diritto_tributario
diritto_commerciale
diritto_penale
diritto_lavoro
diritto_amministrativo
diritto_processuale
```

Historical aliases are normalized in memory by `taxonomy.py`. Historical task
IDs and result paths are not renamed. A task ID such as `civile_generale/0001`
remains unchanged, while its runtime `macro_area` is normalized.

`build_corpus.py` classifies Cassation rulings with the pair
`legalArea + division`:

```text
PEN + Sez. 1, 2, 3, 4, 5, 6 or U -> diritto_penale
CIV + Sez. 1, 2, 3 or U          -> diritto_civile
CIV + Sez. 5                     -> diritto_tributario
CIV + Sez. L                     -> diritto_lavoro
Sez. 7                           -> excluded
unknown combinations             -> excluded with warning
```

## Pipeline Commands

The corpus-preprocessing and task-generation commands below require
`data/raw/sentenze.zip`, an internal export of Cassation rulings (per-ruling
facts, principles and decision) that is not distributed. They are used by the
maintainers to build new tasks; external users do not need them, since the
107 tasks are delivered already built in the task bundle. Neither the
benchmark nor the citation grounding needs the text of any ruling: grounding
works on ECLI identifiers only.

I comandi di preprocessing del corpus e di generazione dei task richiedono
`data/raw/sentenze.zip`, un export interno delle sentenze di Cassazione
(fatti, principi e decisione per ciascun provvedimento) che non è distribuito.
Servono ai manutentori per costruire nuovi task; gli utenti esterni non ne
hanno bisogno, perché i 107 task arrivano già costruiti nel bundle. Né il
benchmark né il citation grounding richiedono il testo delle sentenze: il
grounding lavora solo su identificativi ECLI.

```bash
# Preprocess the Cassation corpus (maintainers only, needs data/raw/sentenze.zip)
legalita-build-corpus

# Build future tasks with canonical slugs (maintainers only)
python -m benchmark.task_builder --n-per-area 50

# Build tasks for one area; historical aliases are accepted
python -m benchmark.task_builder --area diritto_penale --n-per-area 20
python -m benchmark.task_builder --area civile_generale --n-per-area 20

# Run benchmark evaluation
legalita-benchmark --models gpt-4o claude-sonnet-4-6 gemini-2.5-pro

# Run a single macro-area, filtered in memory
legalita-benchmark --models gemini-2.5-pro --area diritto_civile

# Generate reports
legalita-charts --latest
```

## Repository Layout

```text
legal_ita/
├── cli/             command-line entry points and orchestration
├── grounding/       local citation-grounding service
├── modeling/        provider adapters, request config, runtime and usage
├── config.py        shared runtime constants
├── schemas.py       Pydantic data models
└── taxonomy.py      canonical taxonomy and aliases
benchmark/           corpus loading, preprocessing and task generation
evaluation/
├── citations/       citation extraction and local registry/profile access
├── scoring/         task, citation and summary scoring APIs
└── reporting/       reports, leaderboard and charts
scripts/             release and data-bundle utilities
*.py (root)          deprecated compatibility shims for historical commands
```

## Repository and separately distributed data

Distributed separately, on request, as a task bundle:

```text
tasks/                                                  67 reasoning tasks
data/private/bullshit_tasks_v3_missing_documents_40.json  40 missing-document tasks
```

Not distributed with the public project:

```text
data/raw/sentenze.zip          internal rulings export, maintainers only
artifacts/
results/
.env
internal workflow, application backend, and gold-building components
```

Distributed separately, on request, as `legalita_grounding_bundle.zip`:

```text
data/citation_pool/registry/ecli_registry_v1.sqlite     ECLI registry snapshot
data/citation_pool/question_profiles/                   67 gold question profiles
data/citation_pool/manifest.json                        snapshot date, counts and checksums
```

Included in the repository:

```text
source code
schemas
orchestration scripts
documentation
```

The task bundle, generated results, runtime artifacts, credentials, and local
test suites remain ignored by Git. Keeping a path outside Git does not by itself
make its contents confidential; it separates release artifacts from source
code and prevents generated or deployment-specific material from being
committed accidentally.

## Licensing

LegalITA source code is distributed under the [MIT License](LICENSE). Portions
of the benchmark schema and scoring approach were adapted from Harvey LAB; the
corresponding attribution is retained in [NOTICE](NOTICE).

The original task definitions, evaluation criteria, labels, annotations, and
benchmark metadata in the separately distributed 107-task bundle are licensed
under the [Creative Commons Attribution 4.0 International
license](https://creativecommons.org/licenses/by/4.0/); the full license terms
are included in the bundle as `LICENSE-TASKS.md`. This license does not extend
to third-party material beyond the rights held by the LegalITA rights holder.

## Licenze

Il codice sorgente di LegalITA è distribuito con [licenza MIT](LICENSE). Alcune
convenzioni degli schemi e dell'impostazione dello scoring sono state adattate
da Harvey LAB; la relativa attribuzione è conservata in [NOTICE](NOTICE).

Le definizioni originali dei task, i criteri di valutazione, le etichette, le
annotazioni e i metadati del pacchetto separato da 107 task sono distribuiti con
licenza [Creative Commons Attribuzione 4.0
Internazionale](https://creativecommons.org/licenses/by/4.0/); i termini
completi della licenza sono inclusi nel pacchetto come `LICENSE-TASKS.md`.
La licenza non si estende a eventuali materiali di terzi oltre i diritti
effettivamente detenuti dal titolare dei diritti di LegalITA.

## Local Validation

For a lightweight syntax check:

```bash
python -m compileall .
python -c "from legal_ita.cli.benchmark import load_tasks; print('ok')"
```

Live benchmark runs require the model and judge API keys described above.
Citation grounding additionally requires the separately distributed local
registry and question-profile bundle; see
[docs/CITATION_GROUNDING.md](docs/CITATION_GROUNDING.md).
