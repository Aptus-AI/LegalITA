"""
Tassonomia canonica delle macro-aree del benchmark.

I dati storici possono contenere slug legacy; il codice runtime deve
normalizzarli in memoria senza rinominare file, task_id o risultati esistenti.
"""

from __future__ import annotations

import re
import unicodedata


DIRITTO_CIVILE = "diritto_civile"
DIRITTO_TRIBUTARIO = "diritto_tributario"
DIRITTO_COMMERCIALE = "diritto_commerciale"
DIRITTO_PENALE = "diritto_penale"
DIRITTO_LAVORO = "diritto_lavoro"
DIRITTO_AMMINISTRATIVO = "diritto_amministrativo"
DIRITTO_PROCESSUALE = "diritto_processuale"


CANONICAL_MACRO_AREAS: tuple[str, ...] = (
    DIRITTO_CIVILE,
    DIRITTO_TRIBUTARIO,
    DIRITTO_COMMERCIALE,
    DIRITTO_PENALE,
    DIRITTO_LAVORO,
    DIRITTO_AMMINISTRATIVO,
    DIRITTO_PROCESSUALE,
)


MACRO_AREA_LABELS: dict[str, str] = {
    DIRITTO_CIVILE: "Diritto civile",
    DIRITTO_TRIBUTARIO: "Diritto tributario",
    DIRITTO_COMMERCIALE: "Diritto commerciale",
    DIRITTO_PENALE: "Diritto penale",
    DIRITTO_LAVORO: "Diritto del lavoro",
    DIRITTO_AMMINISTRATIVO: "Diritto amministrativo",
    DIRITTO_PROCESSUALE: "Diritto processuale",
}


def _key(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").strip())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold().replace("&", " e ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


_ALIASES_RAW: dict[str, str] = {
    # Canonical slugs.
    **{area: area for area in CANONICAL_MACRO_AREAS},

    # Civile.
    "civile_generale": DIRITTO_CIVILE,
    "responsabilita_civile": DIRITTO_CIVILE,
    "responsabilita civile": DIRITTO_CIVILE,
    "proprieta_successioni": DIRITTO_CIVILE,
    "proprieta successioni": DIRITTO_CIVILE,
    "proprieta e successioni": DIRITTO_CIVILE,
    "famiglia": DIRITTO_CIVILE,
    "successioni": DIRITTO_CIVILE,
    "condominio": DIRITTO_CIVILE,
    "immobiliare": DIRITTO_CIVILE,
    "contratti": DIRITTO_CIVILE,
    "credito": DIRITTO_CIVILE,
    "danno_risarcimento": DIRITTO_CIVILE,
    "danno risarcimento": DIRITTO_CIVILE,
    "sinistri_stradali": DIRITTO_CIVILE,
    "sinistri stradali": DIRITTO_CIVILE,
    "assicurazioni private": DIRITTO_CIVILE,
    "assicurazioni_private": DIRITTO_CIVILE,
    "bancario": DIRITTO_CIVILE,
    "privacy": DIRITTO_CIVILE,
    "sanitario": DIRITTO_CIVILE,
    "civile contrattuale": DIRITTO_CIVILE,
    "civile_contrattuale": DIRITTO_CIVILE,
    "privacy e procedura civile": DIRITTO_CIVILE,
    "privacy procedura civile": DIRITTO_CIVILE,

    # Tributario.
    "tributario": DIRITTO_TRIBUTARIO,
    "diritto tributario": DIRITTO_TRIBUTARIO,

    # Commerciale.
    "societario": DIRITTO_COMMERCIALE,
    "commerciale": DIRITTO_COMMERCIALE,
    "fallimentare": DIRITTO_COMMERCIALE,
    "concorsuale": DIRITTO_COMMERCIALE,
    "crisi d'impresa": DIRITTO_COMMERCIALE,
    "crisi di impresa": DIRITTO_COMMERCIALE,
    "crisi_impresa": DIRITTO_COMMERCIALE,

    # Penale.
    "penale": DIRITTO_PENALE,
    "penale_colposo": DIRITTO_PENALE,
    "penale colposo": DIRITTO_PENALE,
    "penale_cautelare": DIRITTO_PENALE,
    "penale cautelare": DIRITTO_PENALE,
    "diritto penale": DIRITTO_PENALE,

    # Lavoro.
    "lavoro": DIRITTO_LAVORO,
    "diritto lavoro": DIRITTO_LAVORO,
    "diritto del lavoro": DIRITTO_LAVORO,
    "previdenza": DIRITTO_LAVORO,
    "previdenziale": DIRITTO_LAVORO,
    "diritto previdenziale": DIRITTO_LAVORO,

    # Amministrativo.
    "amministrativo": DIRITTO_AMMINISTRATIVO,
    "diritto amministrativo": DIRITTO_AMMINISTRATIVO,
    "immigrazione": DIRITTO_AMMINISTRATIVO,

    # Processuale.
    "processuale": DIRITTO_PROCESSUALE,
    "diritto processuale": DIRITTO_PROCESSUALE,
    "procedura civile": DIRITTO_PROCESSUALE,
    "processuale civile": DIRITTO_PROCESSUALE,
    "diritto processuale civile": DIRITTO_PROCESSUALE,
    "procedura penale": DIRITTO_PROCESSUALE,
    "processuale penale": DIRITTO_PROCESSUALE,
    "diritto processuale penale": DIRITTO_PROCESSUALE,
    "procedura amministrativa": DIRITTO_PROCESSUALE,
    "processuale amministrativo": DIRITTO_PROCESSUALE,
    "diritto processuale amministrativo": DIRITTO_PROCESSUALE,
    "procedura tributaria": DIRITTO_PROCESSUALE,
    "processuale tributario": DIRITTO_PROCESSUALE,
    "diritto processuale tributario": DIRITTO_PROCESSUALE,
}


LEGACY_MACRO_AREA_ALIASES: dict[str, str] = {
    _key(alias): target for alias, target in _ALIASES_RAW.items()
}


def normalize_macro_area(value: object, strict: bool = True) -> str:
    """
    Restituisce lo slug canonico per una macro-area.

    In modalita non strict, un valore sconosciuto viene restituito come slug
    normalizzato; e utile per aggregare risultati storici senza fallire su
    artefatti non standard.
    """
    key = _key(value)

    # Regola esplicita di priorita: "privacy e procedura civile" resta civile.
    if "privacy" in key and "procedura_civile" in key:
        return DIRITTO_CIVILE

    if key in LEGACY_MACRO_AREA_ALIASES:
        return LEGACY_MACRO_AREA_ALIASES[key]

    if key.startswith(("penale_", "diritto_penale_")) or key.endswith("_penale"):
        return DIRITTO_PENALE

    if strict:
        raise ValueError(f"Macro-area sconosciuta: {value!r}")

    return key or ""


def classify_cassazione_macro_area(
    legal_area: object,
    division: object,
) -> str | None:
    """
    Classifica una pronuncia Cassazione usando sempre legalArea + division.

    Ritorna None per Sez. 7 e per combinazioni non supportate.
    """
    area = str(legal_area or "").strip().upper()
    div = str(division or "").strip()

    if div == "Sez. 7":
        return None

    if area == "PEN" and div in {"Sez. 1", "Sez. 2", "Sez. 3", "Sez. 4", "Sez. 5", "Sez. 6", "Sez. U"}:
        return DIRITTO_PENALE

    if area == "CIV" and div in {"Sez. 1", "Sez. 2", "Sez. 3", "Sez. U"}:
        return DIRITTO_CIVILE

    if area == "CIV" and div == "Sez. 5":
        return DIRITTO_TRIBUTARIO

    if area == "CIV" and div == "Sez. L":
        return DIRITTO_LAVORO

    return None
