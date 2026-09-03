from __future__ import annotations

import re
import unicodedata
from typing import Any


SUPPORTED_NAMESPACES = (
    "CASS",
    "MER",
    "ADM",
    "COST",
    "CONT",
    "TAX",
    "ABF",
    "COVIP",
)

OUTSIDE_INDEX_NAMESPACES = ("CEDU", "CGUE")
METADATA_NUMBER_WIDTH = 8

DOC_TYPE_MAP = {
    "SENTENZA": "SENT",
    "SENT": "SENT",
    "ORDINANZA": "ORD",
    "ORD": "ORD",
    "DECRETO": "DEC",
    "DEC": "DEC",
}

VENUE_CODES = {
    "AGRIGENTO": "AG",
    "ANCONA": "AN",
    "AOSTA": "AO",
    "BARI": "BA",
    "BOLOGNA": "BO",
    "BRESCIA": "BS",
    "CAGLIARI": "CA",
    "CALTANISSETTA": "CL",
    "CAMPOBASSO": "CB",
    "CATANIA": "CT",
    "CATANZARO": "CZ",
    "FIRENZE": "FI",
    "GENOVA": "GE",
    "L'AQUILA": "AQ",
    "AQUILA": "AQ",
    "LECCE": "LE",
    "MESSINA": "ME",
    "MILANO": "MI",
    "NAPOLI": "NA",
    "PALERMO": "PA",
    "PERUGIA": "PG",
    "POTENZA": "PZ",
    "REGGIO CALABRIA": "RC",
    "ROMA": "RM",
    "SALERNO": "SA",
    "TORINO": "TO",
    "TRENTO": "TN",
    "TRIESTE": "TS",
    "VENEZIA": "VE",
}


def normalize_year(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 1000 <= value <= 9999 else None
    if isinstance(value, float):
        if not value.is_integer():
            return None
        return normalize_year(int(value))

    text = str(value).strip()
    if not re.fullmatch(r"\d{4}", text):
        return None
    return int(text)


def normalize_ecli(value: Any) -> str | None:
    """Normalize ECLI strings and URL slug variants to canonical colon form."""
    if value is None:
        return None
    raw = str(value).strip().upper()
    if not raw:
        return None

    match = re.search(r"ECLI[:_][A-Z]{2}[:_][A-Z0-9]+[:_]\d{4}[:_][A-Z0-9.-]+", raw)
    token = match.group(0) if match else raw
    token = token.replace("_", ":")
    parts = token.split(":")
    if len(parts) < 5 or parts[0] != "ECLI":
        return None
    return ":".join(parts[:4] + ["".join(parts[4:])])


def normalize_number(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not re.fullmatch(r"\d+", text):
        return None
    stripped = text.lstrip("0")
    return stripped or "0"


def normalize_nrg(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if re.fullmatch(r"\d+", text) else None


def normalize_metadata_number(value: Any) -> str | None:
    number = normalize_number(value)
    if number is None:
        return None
    return number.zfill(METADATA_NUMBER_WIDTH)


def normalize_doc_type(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper().rstrip(".")
    return DOC_TYPE_MAP.get(text)


def normalize_code(value: Any) -> str | None:
    if value is None:
        return None
    text = re.sub(r"[^A-Za-z0-9]", "", str(value).strip().upper())
    return text or None


def normalize_legal_area(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper().rstrip(".")
    if text in {"CIV", "CIVILE"}:
        return "CIV"
    if text in {"PEN", "PENALE"}:
        return "PEN"
    return None


def normalize_cass_sector(value: Any) -> str | None:
    if value is None:
        return None
    text = _ascii_words(str(value))
    compact = re.sub(r"\s+", "", text)
    if re.search(r"\bCASSSU\b", text) or re.search(r"\bCASS\s+SU\b", text):
        return "SU"
    if re.search(r"\bCASSLAV\b", text) or re.search(r"\bCASS\s+LAV\b", text):
        return "L"
    if compact in {"CASSSU", "SU", "SSUU", "SEZUN", "SEZUNITE", "SEZIONIUNITE"}:
        return "SU"
    if compact in {"CASSLAV", "L", "LAV", "LAVORO", "SEZL", "SEZLAV", "SEZLAVORO"}:
        return "L"
    if "SEZIONI UNITE" in text or "SEZ UNITE" in text or "SS UU" in text:
        return "SU"
    if "SEZ LAV" in text or "SEZ LAVORO" in text or re.search(r"\bSEZ L\b", text):
        return "L"
    return None


def normalize_division(value: Any) -> str | None:
    if value is None:
        return None
    text = _ascii_words(str(value))
    if not text:
        return None
    text = re.sub(r"^(SEZ|SEZIONE)\s+", "", text)
    return text or None


def _ascii_words(value: str) -> str:
    value = value.replace("\u2019", "'").replace("\u2018", "'")
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = ascii_text.replace("`", "'").replace("'", " ")
    ascii_text = re.sub(r"[^A-Za-z0-9]+", " ", ascii_text)
    return re.sub(r"\s+", " ", ascii_text).strip().upper()


def _venue_from_text(value: str) -> str | None:
    text = _ascii_words(value)
    if text in VENUE_CODES:
        return VENUE_CODES[text]
    for name, code in sorted(VENUE_CODES.items(), key=lambda item: len(item[0]), reverse=True):
        if name in text:
            return code
    return None


def derive_authority_codes(
    authority: str | None,
    namespace: str | None = None,
) -> tuple[str | None, str | None]:
    if not authority:
        return None, None

    text = _ascii_words(authority)
    namespace = (namespace or "").upper()
    venue = _venue_from_text(text)

    if namespace == "MER" or "TRIBUNALE" in text or "CORTE D APPELLO" in text or "CORTE DI APPELLO" in text:
        if not venue:
            return None, None
        if "TRIBUNALE" in text:
            return f"TR{venue}", venue
        if "CORTE D APPELLO" in text or "CORTE DI APPELLO" in text:
            return f"CA{venue}", venue

    if "CONSIGLIO DI STATO" in text or "CONS STATO" in text:
        return "CDS", None

    if "TAR" in text or "T A R" in text:
        if venue:
            return f"TAR{venue}", venue
        return None, None

    return None, venue


def infer_namespace_from_ecli(ecli: str) -> str | None:
    parts = str(ecli).upper().split(":", 4)
    if len(parts) != 5 or parts[0] != "ECLI" or parts[1] != "IT":
        return None

    court_code = parts[2]
    if court_code in {"CASS", "COST", "CONT", "ABF", "COVIP"}:
        return court_code
    if court_code == "CDS" or court_code.startswith("TAR"):
        return "ADM"
    if court_code.startswith("CG1") or court_code.startswith("CG2"):
        return "TAX"
    if court_code.startswith("CA") or court_code.startswith("TR"):
        return "MER"
    return None


def namespace_for_citation(citation: Any) -> str | None:
    ecli = getattr(citation, "ecli", None)
    if ecli:
        namespace = infer_namespace_from_ecli(ecli)
        if namespace:
            return namespace

    namespace = getattr(citation, "suggested_namespace", None)
    if namespace is None:
        return None
    namespace = str(namespace).strip().upper()
    return namespace if namespace in SUPPORTED_NAMESPACES else None


def build_exact_ecli(
    namespace: str | None,
    year: Any,
    number: Any,
    *,
    legal_area: str | None = None,
    court: str | None = None,
    venue: str | None = None,
    doc_type: str | None = None,
    nrg: str | None = None,
) -> str | None:
    namespace = (namespace or "").upper()
    year_value = normalize_year(year)
    number_value = normalize_number(number)
    legal_area = normalize_legal_area(legal_area)
    court = normalize_code(court)
    venue = normalize_code(venue)
    doc_type = normalize_doc_type(doc_type)
    nrg_value = normalize_nrg(nrg)

    if year_value is None or number_value is None:
        return None

    if namespace == "CASS":
        if legal_area is None:
            return None
        return f"ECLI:IT:CASS:{year_value}:{number_value}{legal_area}"

    if namespace == "COST":
        return f"ECLI:IT:COST:{year_value}:{number_value}"

    if namespace == "ABF":
        return f"ECLI:IT:ABF:{year_value}:{number_value}"

    if namespace == "ADM":
        if not court or not doc_type:
            return None
        return f"ECLI:IT:{court}:{year_value}:{number_value}{doc_type}"

    if namespace == "MER":
        if not court or not nrg_value:
            return None
        return f"ECLI:IT:{court}:{year_value}:{number_value}.{nrg_value}"

    if namespace == "TAX":
        if not court or not venue or not doc_type:
            return None
        return f"ECLI:IT:{court}{venue}:{year_value}:{number_value}{doc_type}"

    if namespace == "CONT":
        if not venue:
            return None
        return f"ECLI:IT:CONT:{year_value}:{number_value}{venue}"

    return None


def build_ecli_prefix(
    namespace: str | None,
    year: Any,
    number: Any,
    *,
    legal_area: str | None = None,
    court: str | None = None,
    venue: str | None = None,
    doc_type: str | None = None,
    nrg: str | None = None,
) -> str | None:
    namespace = (namespace or "").upper()
    year_value = normalize_year(year)
    number_value = normalize_number(number)
    legal_area = normalize_legal_area(legal_area)
    court = normalize_code(court)
    venue = normalize_code(venue)
    doc_type = normalize_doc_type(doc_type)
    nrg_value = normalize_nrg(nrg)

    if year_value is None or number_value is None:
        return None

    if namespace == "CASS" and legal_area is None:
        return f"ECLI:IT:CASS:{year_value}:{number_value}"

    if namespace == "MER":
        if not court:
            return None
        if nrg_value:
            return f"ECLI:IT:{court}:{year_value}:{number_value}.{nrg_value}"
        return f"ECLI:IT:{court}:{year_value}:{number_value}."

    if namespace == "ADM" and court and doc_type is None:
        return f"ECLI:IT:{court}:{year_value}:{number_value}"

    if namespace == "TAX" and court and venue and doc_type is None:
        return f"ECLI:IT:{court}{venue}:{year_value}:{number_value}"

    if namespace == "CONT" and venue is None:
        return f"ECLI:IT:CONT:{year_value}:{number_value}"

    return None


def cass_ecli_base(year: Any, number: Any) -> str | None:
    year_value = normalize_year(year)
    number_value = normalize_number(number)
    if year_value is None or number_value is None:
        return None
    return f"ECLI:IT:CASS:{year_value}:{number_value}"


def cass_ecli_candidates(
    year: Any,
    number: Any,
    *,
    legal_area: str | None = None,
) -> tuple[list[str], list[str]]:
    base = cass_ecli_base(year, number)
    if base is None:
        return [], []

    area = normalize_legal_area(legal_area)
    if area == "CIV":
        return [base + "CIV"], [base + "PEN"]
    if area == "PEN":
        return [base + "PEN"], [base + "CIV"]
    return [base + "CIV", base + "PEN"], []
