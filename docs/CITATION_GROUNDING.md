# Citation grounding locale

Il grounding pubblico valuta le citazioni di una run LegalITA senza richiedere
accesso a servizi proprietari. Per ogni citazione verifica due aspetti distinti:

1. l'identità della decisione, tramite un registry ECLI SQLite in sola lettura;
2. la pertinenza per il quesito, tramite un question profile versionato.

Una decisione esistente ma non pertinente non contribuisce a GOG o Coverage.
Citazioni ambigue o non verificabili restano `unresolved`: il calcolo è
fail-closed e non assegna credito sulla base di un'ipotesi.

## Installazione

```bash
python -m pip install -e .
```

L'estrazione delle citazioni testuali richiede:

```text
OPENAI_API_KEY=...
```

Il registry e i profili sono distribuiti separatamente dal codice. Estrarre il
bundle in modo da ottenere:

```text
data/citation_pool/
├── registry/
│   ├── ecli_registry_v1.sqlite
│   └── manifest.json
└── question_profiles/
    ├── index.json
    └── <macro-area>_<task>.json
```

Questi file sono ignorati da Git. Il report registra sempre la data dello
snapshot e la versione dei profili, così due risultati possono essere
confrontati soltanto quando usano lo stesso bundle.

Chi mantiene un registry compatibile può produrre un archivio con il formato
pubblico minimo usando:

```bash
python scripts/build_public_grounding_bundle.py \
  --registry /percorso/ecli_registry_v1.sqlite \
  --profiles /percorso/question_profiles \
  --out-dir legalita-grounding-bundle \
  --zip
```

Il builder conserva soltanto i campi necessari al runtime, genera un manifest
con conteggi e checksum e rifiuta di sovrascrivere una destinazione esistente.

## Utilizzo

Per una directory contenente `scores.json` o `outputs.json`:

```bash
legalita-grounding --results results/<provider>/<run> --backend local
```

Per un CSV contenente `task_id` e risposta:

```bash
legalita-grounding --csv risposte.csv --model NOME --backend local
```

Se il CSV non contiene `task_id`, il comando associa la colonna della domanda
al testo presente nei question profiles. I nomi delle colonne possono essere
specificati esplicitamente:

```bash
legalita-grounding \
  --csv risposte.csv \
  --question-column Domanda \
  --answer-column Risposta \
  --model NOME
```

Percorsi alternativi possono essere forniti tramite CLI:

```bash
legalita-grounding \
  --results results/<run> \
  --citation-registry /percorso/registry.sqlite \
  --question-profiles /percorso/question_profiles
```

oppure tramite ambiente:

```text
LEGALITA_CITATION_REGISTRY_PATH=data/citation_pool/registry/ecli_registry_v1.sqlite
LEGALITA_QUESTION_PROFILES_DIR=data/citation_pool/question_profiles
```

`--fast-path` evita la chiamata all'estrattore quando tutti i riferimenti sono
già espressi come ECLI completi o URL riconoscibili.

## Metriche

Per ogni task `i`:

```text
GOG_i = citazioni esistenti e allineate / citazioni estratte
Coverage_i = 1 se esiste almeno una citazione allineata, altrimenti 0
```

GOG e Coverage sono le medie sui 67 task. Un task senza citazioni contribuisce
con zero; task mancanti dalla run contribuiscono ugualmente con zero, salvo che
si imposti esplicitamente un diverso `--n-tasks` per una prova parziale.

## Output

Ogni esecuzione usa una nuova directory quando il percorso predefinito esiste
già e genera:

```text
results/grounding-offline/<run>/citation_grounding_v3.json
results/grounding-offline/<run>/citation_grounding_v3.md
```

Il terminale mostra soltanto:

```text
GOG=xx.x%  Coverage=xx.x%  Tasks=67  backend=local  registry=<data>
results=<directory-del-report>
```

Il JSON conserva il dettaglio per task e citazione, inclusi l'ECLI risolto, la
relazione con il profilo, gli eventuali limiti di copertura dello snapshot e i
tier di evidenza che hanno giustificato l'allineamento.
