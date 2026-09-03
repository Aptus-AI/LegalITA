# Citation grounding locale (`legalita-grounding --backend local`)

Il citation grounding è separato dallo scoring giuridico. Il judge risponde
alla domanda "la risposta soddisfa i criteri PASS/FAIL del task?"; il grounding
risponde a due domande diverse, per ogni sentenza citata nella risposta:

1. **Identità**: la decisione citata esiste ed è identificabile in modo univoco?
2. **Rilevanza**: quella decisione risolve la questione posta dal quesito?

Il backend `local` risponde a entrambe con due file distribuiti a parte, senza
contattare alcun servizio Aptus: un **registry ECLI** in SQLite (snapshot
versionato dell'indice delle decisioni) e un **question profile** per ciascuno
dei 67 task. L'unica chiamata di rete che resta è quella all'estrattore di
citazioni (un modello OpenAI, vedi sotto), evitabile con `--fast-path` quando
le citazioni sono già espresse come ECLI espliciti.

Il calcolo è fail-closed: citazioni ambigue o non verificabili restano
`unresolved` e non ricevono mai credito.

## Cosa serve

| Cosa | Dove | Note |
|---|---|---|
| Registry ECLI | `data/citation_pool/registry/ecli_registry_v1.sqlite` | ~190 MB, sola lettura. Namespace inclusi nello snapshot corrente: `CASS` (Cassazione), `MER` (tribunali e corti d'appello), `CONT` (Corte dei conti), `COST` (Corte costituzionale). La data dello snapshot è nella tabella `meta` (`built_at`) |
| Manifest del bundle | `data/citation_pool/manifest.json` | conteggi per namespace e checksum SHA-256 dei file |
| Question profiles | `data/citation_pool/question_profiles/<task>.json` + `index.json` | uno per task (`diritto_civile_0001.json`, ...): testo del quesito, `resolving_rulings` con tier di evidenza, decisioni marginali/irrilevanti recuperate |
| Chiave OpenAI | `.env` → `OPENAI_API_KEY` | solo per l'estrattore di citazioni; non serve con `--fast-path` |

Il codice si installa con:

```bash
python -m pip install -e .
```

Il bundle registry + profili (`legalita_grounding_bundle.zip`, SHA-256 nel
README) è distribuito separatamente dal codice, già costruito: il registry è
troppo grande per il repository sorgente e i profili gold non sono
ricostruibili senza le pipeline interne. Si richiede via e-mail agli stessi
contatti indicati nel README per il bundle dei task, con oggetto "grounding
bundle legalITA". Lo zip contiene una cartella `legalita-grounding-bundle/`;
il suo contenuto va copiato in `data/citation_pool/` nella root del progetto:

```bash
unzip legalita_grounding_bundle.zip -d /tmp/legalita-bundle
mkdir -p data/citation_pool
cp -R /tmp/legalita-bundle/legalita-grounding-bundle/. data/citation_pool/
```

Risultato atteso:

```text
data/citation_pool/
├── manifest.json
├── registry/
│   └── ecli_registry_v1.sqlite
└── question_profiles/
    ├── index.json
    └── <macro-area>_<numero-task>.json   (67 file)
```

`data/` è ignorata da Git. Percorsi alternativi possono essere indicati da
riga di comando (`--citation-registry`, `--question-profiles`) o da ambiente:

```text
LEGALITA_CITATION_REGISTRY_PATH=data/citation_pool/registry/ecli_registry_v1.sqlite
LEGALITA_QUESTION_PROFILES_DIR=data/citation_pool/question_profiles
```

### Estrattore di citazioni

Le citazioni scritte in linguaggio naturale ("Cass. Sez. Un. n. 41994/2021",
"ord. n. 20522 del 30 luglio 2019") vengono estratte da un modello OpenAI
tramite Responses API. Il modello predefinito è quello usato nei report
LegalITA ed è letto da `CITATION_EXTRACTOR_MODEL`; può essere cambiato con
`--extractor-model` o nel `.env`:

```text
OPENAI_API_KEY=...
CITATION_EXTRACTOR_MODEL=<modello OpenAI con Responses API e output strutturato>
```

Un estrattore diverso da quello predefinito produce risultati non
confrontabili con i report ufficiali: il modello usato è registrato nel JSON
di output.

## Comandi

Grounding di una run già eseguita, cioè una directory che contiene
`scores.json` (prodotto da `legalita-benchmark`) oppure `outputs.json`, una
lista di oggetti con `task_id` e `model_output` (o `response`):

```bash
legalita-grounding --results results/<provider>/<run> --backend local
```

Da un CSV di risposte esterne. Se il CSV ha una colonna `task_id` viene usata;
altrimenti la colonna della domanda viene associata al testo del quesito
presente nei question profiles:

```bash
legalita-grounding --csv risposte.csv --model NOME_SISTEMA --backend local

legalita-grounding --csv risposte.csv --model NOME_SISTEMA --backend local \
  --question-column Domanda --answer-column Risposta
```

Opzioni utili:

```text
--task-ids diritto_civile/0001 lavoro/0003   prova su pochi task
--n-tasks N                                  denominatore delle medie (default 67)
--fast-path                                  salta l'estrattore LLM: valuta solo
                                             ECLI espliciti e URL riconoscibili
--extractor-model / --extractor-timeout-seconds / --extractor-max-retries
--out-dir                                    directory del report
```

`--backend` accetta solo `local`: è esplicito per distinguere i report da
quelli prodotti dalla pipeline interna, che non è distribuita.

## Cosa succede dentro

1. **Estrazione.** Da ogni risposta si raccolgono le citazioni attraverso tre
   canali fusi insieme: URL che contengono un ECLI, ECLI scritti in chiaro
   (regex) e, per tutto il resto, l'estrattore LLM. Con `--fast-path`
   l'estrattore viene saltato; le citazioni testuali che non sono ECLI
   completi risultano allora non estratte, e il report lo segnala
   (`fast_path_skipped`).
2. **Identità.** Ogni citazione viene trasformata in uno o più ECLI candidati
   (per la Cassazione, numero e anno generano la variante civile e quella
   penale) e cercata nel registry locale
   (`evaluation/citations/local_registry.py`, `local_resolver.py`). Esiti:

   ```text
   resolved_local_registry_exact              trovata, metadati coerenti
   resolved_local_registry_incomplete         trovata, citazione priva di alcuni elementi
   resolved_local_registry_metadata_mismatch  trovata, ma sezione/data/tipo non coincidono
   ambiguous_local_registry                   più candidati, nessun elemento per scegliere
   not_found_in_index                         nessun candidato nel registry
   outside_index_scope                        decisione di corti non italiane (CEDU, CGUE):
                                              fuori dal perimetro del registry
   ```

3. **Rilevanza.** Gli ECLI risolti vengono confrontati con il profilo del
   quesito: se compaiono tra le `resolving_rulings` la citazione è
   `issue_aligned`; se compaiono tra le decisioni recuperate ma non risolutive è
   `retrieved_only`; altrimenti `outside_profile`. Le citazioni
   `not_found_in_index` diventano `fabricated_or_not_found`, salvo che il
   registry non possa pronunciarsi (vedi limiti): in quel caso restano
   `unresolved` con una `offline_coverage_note`. Ambigue e con metadati
   discordanti restano `unresolved`.
4. **Metriche.** Per il task `i`:

   ```text
   GOG_i      = citazioni issue_aligned / citazioni estratte   (0 se non cita nulla)
   Coverage_i = 1 se almeno una citazione è issue_aligned, altrimenti 0
   ```

   GOG e Coverage sono le medie sui 67 task. Un task assente dalla run
   contribuisce con zero, salvo che si imposti `--n-tasks` per una prova
   parziale.

## Output

Ogni esecuzione scrive in una nuova directory (se il percorso predefinito
esiste già viene aggiunto un timestamp):

```text
results/grounding-offline/<run>/citation_grounding_v3.json   report completo
results/grounding-offline/<run>/citation_grounding_v3.md     tabella per task
```

A schermo:

```text
GOG=xx.x%  Coverage=xx.x%  Tasks=67  backend=local  registry=<data snapshot>
results=<directory del report>
```

Nel JSON ogni citazione porta lo stato di identità (`final_status`),
l'`matched_ecli`, la relazione con il profilo (`gold_v3_relation`),
`issue_profile_match`, i tier di evidenza delle sentenze del profilo
corrispondenti (`offline_tiers`, es. `A_lawyer_bonus`, `A_criteria_slot`,
`B_gold_relevant`) e, quando serve, `offline_coverage_note`. Il `summary`
contiene `gog`, `coverage`, `gog_by_task`, `coverage_by_task`,
`gog_backend="local"` e `registry_built_at`. Due report sono confrontabili solo
se dichiarano lo stesso snapshot del registry e la stessa versione dei profili.

I file `summary.json` prodotti da `legalita-benchmark` contengono anche campi
come `global_grounding` e `required_citation_coverage_rate`: sono metriche del
vecchio citation gold interno, non GOG e Coverage. Nel repository pubblico il
grounding si esegue esclusivamente con `legalita-grounding`.

## Limiti dello snapshot

Il registry è una fotografia: quello che non contiene non può essere né
confermato né smentito. Il report distingue questi casi dalle citazioni
inventate.

- **Giurisdizioni italiane non incluse nello snapshot** (es. Corti di
  giustizia tributaria, TAR e Consiglio di Stato, ABF): le citazioni restano
  `unresolved` con nota `giurisdizione [...] non presente nel registry
  offline`, mai `fabricated`.
- **Annate di Cassazione poco coperte** (in pratica, le decisioni anteriori
  al 2014): `not_found_in_index` con nota `copertura CASS insufficiente per
  l'anno [...]`.
- **Omonimi civile/penale** citati senza sezione (es. "Cass. n. 8053/2014"
  quando esistono sia la CIV sia la PEN): `ambiguous_local_registry`, quindi
  `unresolved`. Basta indicare la sezione perché la citazione si risolva.
- **Decisioni di merito citate senza ECLI** ("Trib. Milano, 12 marzo 2021"):
  non verificabili offline, `unresolved` con nota.
- **Decisioni successive a `built_at`**: sconosciute al registry.
- **Estrattore LLM**: tra due esecuzioni può segmentare diversamente le
  citazioni multiple. Questo vale per qualsiasi pipeline basata su estrazione
  LLM; `--fast-path` elimina la variabilità ma copre solo gli ECLI espliciti.

## Produrre o aggiornare un bundle

Chi dispone di un registry compatibile (tabelle `registry` e `meta` con lo
stesso schema) e di una directory di question profiles può generare un bundle
nel formato pubblico minimo con:

```bash
python scripts/build_public_grounding_bundle.py \
  --registry /percorso/ecli_registry_v1.sqlite \
  --profiles /percorso/question_profiles \
  --out-dir legalita-grounding-bundle \
  --zip
```

Il builder usa solo la libreria standard, conserva i soli campi necessari al
runtime, scrive `manifest.json` nella root del bundle con conteggi e checksum, e
rifiuta di sovrascrivere una destinazione esistente. Le pipeline interne che
alimentano registry e profili non fanno parte del repository pubblico.

## Problemi comuni

- `Registry locale non trovato` / `Question profiles non trovati` → il bundle
  non è stato estratto in `data/citation_pool/`, oppure
  `LEGALITA_CITATION_REGISTRY_PATH` / `LEGALITA_QUESTION_PROFILES_DIR` puntano
  altrove.
- `OPENAI_API_KEY non impostata in .env` → serve la chiave per l'estrattore,
  oppure usare `--fast-path` se le risposte contengono ECLI espliciti.
- `Formato risultati non valido` → `outputs.json` deve essere una lista di
  oggetti con `task_id` e `model_output`; una directory di
  `legalita-benchmark` contiene già `scores.json` nel formato giusto.
- Molte citazioni `unresolved` con `offline_coverage_note` → sono limiti dello
  snapshot, non errori: vedere la sezione precedente.
- Tempi lunghi o timeout dell'estrattore → `--extractor-timeout-seconds`,
  oppure `--task-ids` a blocchi.
