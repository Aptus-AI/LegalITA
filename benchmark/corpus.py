"""
Loader del corpus processato.
Legge corpus.jsonl prodotto da build_corpus.py e restituisce
liste di Provvedimento, con supporto a filtri per macro_area.

Uso tipico:
    from benchmark.corpus import load_corpus

    tutti    = load_corpus()
    civile   = load_corpus(macro_area="diritto_civile")
    campione = load_corpus(macro_area="lavoro", n=200, seed=42)
"""

import json
import logging
import random
from pathlib import Path

from legal_ita.config import CORPUS_JSONL, RANDOM_SEED
from legal_ita.schemas import Provvedimento
from legal_ita.taxonomy import normalize_macro_area

log = logging.getLogger(__name__)


def load_corpus(
        path: Path = CORPUS_JSONL,
        macro_area: str | None = None,
        n: int | None = None,
        seed: int = RANDOM_SEED,
) -> list[Provvedimento]:
    """
    Carica il corpus da JSONL.

    Args:
        path:       Path al corpus.jsonl. Default da config.
        macro_area: Se specificata, restituisce solo i provvedimenti
                    di quella macro-area.
        n:          Se specificato, restituisce un campione casuale
                    di n provvedimenti. Applica DOPO il filtro macro_area.
        seed:       Seed per riproducibilità del campionamento.

    Returns:
        Lista di Provvedimento validati.

    Raises:
        FileNotFoundError: se corpus.jsonl non esiste.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Corpus non trovato: {path}\n"
            f"Esegui prima: python build_corpus.py"
        )

    requested_area = (
        normalize_macro_area(macro_area, strict=True)
        if macro_area is not None
        else None
    )

    records: list[Provvedimento] = []
    n_skipped = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                prov = Provvedimento.model_validate(data)
            except Exception as e:
                log.debug(f"Riga saltata: {e}")
                n_skipped += 1
                continue

            if requested_area and prov.macro_area != requested_area:
                continue

            records.append(prov)

    if n_skipped:
        log.warning(f"Righe saltate per errore: {n_skipped}")

    log.info(
        f"Caricati {len(records)} provvedimenti"
        + (f" [{requested_area}]" if requested_area else "")
    )

    # campionamento
    if n is not None:
        if n > len(records):
            log.warning(
                f"n={n} > disponibili={len(records)}, restituisco tutti."
            )
        else:
            rng = random.Random(seed)
            records = rng.sample(records, n)
            log.info(f"Campione: {len(records)} provvedimenti (seed={seed})")

    return records


def load_corpus_by_area(
        path: Path = CORPUS_JSONL,
        seed: int = RANDOM_SEED,
) -> dict[str, list[Provvedimento]]:
    """
    Carica il corpus raggruppato per macro_area.
    Utile per campionamento stratificato.

    Returns:
        Dict {macro_area: [Provvedimento, ...]}
    """
    all_records = load_corpus(path=path, seed=seed)

    grouped: dict[str, list[Provvedimento]] = {}
    for prov in all_records:
        grouped.setdefault(prov.macro_area, []).append(prov)

    for area, records in grouped.items():
        log.info(f"  {area:<30} {len(records):>5} provvedimenti")

    return grouped
