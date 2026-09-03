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
# Run the 67-task case-law reasoning benchmark without citation grounding.
python run_benchmark.py --models gpt-4o --skip-citation-grounding

# Run the 40 missing-document detection tasks.
python run_bullshit_v2.py --models gpt-4o
```

The first command measures whether a model's answer satisfies the legal
PASS/FAIL criteria attached to each reasoning task. The second measures whether
the model notices that documents invoked by the question were not supplied,
instead of inventing their contents or giving document-specific advice.

These runs still require API keys for the evaluated model and the configured
judge or judges. Citation grounding can be run separately with the public
local bundle; see [docs/CITATION_GROUNDING.md](docs/CITATION_GROUNDING.md).

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
# Esegue i 67 task di ragionamento senza citation grounding.
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
judge configurati. Il citation grounding può essere eseguito separatamente con
il bundle locale pubblico; vedere
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

## Citation Grounding

Citation grounding is separate from legal-reasoning scoring. It extracts the
rulings cited in an answer, checks their identity against the local registry,
and compares them with the profile of the corresponding question. The command
reports the GOG and Coverage metrics without contacting an Aptus service:

```bash
legalita-grounding --results results/<provider>/<run> --backend local
```

The registry and question profiles (about 190 MB) are distributed separately
because the registry is too large for the source repository. Like the task
bundle, they are available on request at the e-mail addresses above, with
subject "grounding bundle legalITA". Setup, bundle layout, snapshot limits and
report semantics are documented in
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

```bash
# Preprocess the Cassation corpus
legalita-build-corpus

# Build future tasks with canonical slugs
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
data/raw/sentenze.zip
artifacts/
results/
.env
internal workflow, application backend, and gold-building components
```

Distributed separately, on request, as a grounding bundle:

```text
data/citation_pool/registry/ecli_registry_v1.sqlite     ECLI registry snapshot
data/citation_pool/question_profiles/                   67 question profiles
data/citation_pool/manifest.json                        counts and checksums
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
