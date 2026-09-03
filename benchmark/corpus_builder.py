"""
Preprocessing del corpus Cassazione.

Legge data/raw/sentenze.zip, filtra i provvedimenti supportati, normalizza le
macro-aree canoniche e produce data/processed/corpus.jsonl.

Uso:
    python build_corpus.py
    python build_corpus.py --zip data/raw/sentenze.zip
    python build_corpus.py --limit 1000
"""

import argparse
import json
import logging
import re
import zipfile
from pathlib import Path

from pydantic import ValidationError
from tqdm import tqdm

from legal_ita.config import (
    CORPUS_JSONL,
    CORPUS_ZIP,
    EXCLUDED_DIVISIONS,
    MIN_FACTS_LENGTH,
    MIN_PRINCIPLES,
)
from legal_ita.schemas import Provvedimento
from legal_ita.taxonomy import classify_cassazione_macro_area


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _decode(raw: bytes) -> str:
    """Decodifica bytes con fallback latin-1."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")


def _extract_ecli_id(path: str) -> str | None:
    """Estrae l'ecli_id dal path del file nello zip."""
    m = re.search(r"(ECLI_IT_[A-Z]+_\d+_[\w-]+)", path)
    return m.group(1) if m else None


def _group_by_ecli(json_files: list[str]) -> dict[str, dict[str, str]]:
    """Raggruppa i path per ecli_id, separando info_ e metadata_."""
    groups: dict[str, dict[str, str]] = {}
    for path in json_files:
        ecli_id = _extract_ecli_id(path)
        if not ecli_id:
            continue
        groups.setdefault(ecli_id, {})
        if "info_" in path:
            groups[ecli_id]["info"] = path
        elif "metadata_" in path:
            groups[ecli_id]["metadata"] = path
    return groups


def build_corpus(
    zip_path: Path,
    out_path: Path,
    limit: int | None = None,
) -> int:
    """
    Legge lo zip, filtra, valida e scrive corpus.jsonl.

    La macro-area viene sempre classificata dalla coppia legalArea + division.
    Le combinazioni non previste dalla tassonomia sono escluse con warning
    aggregato finale.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Apro zip: %s", zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        all_files = z.namelist()

    json_files = [
        name for name in all_files
        if name.endswith(".json") and not name.startswith("__MACOSX")
    ]
    log.info("File JSON reali: %d", len(json_files))

    groups = _group_by_ecli(json_files)
    log.info("Provvedimenti unici: %d", len(groups))

    if limit:
        keys = list(groups.keys())[:limit]
        groups = {key: groups[key] for key in keys}
        log.info("Limite applicato: %d", limit)

    n_written = 0
    n_no_cass = 0
    n_excluded = 0
    n_no_map = 0
    n_short = 0
    n_no_princ = 0
    n_invalid = 0
    n_missing = 0
    unknown_combos: dict[tuple[str, str], int] = {}

    with zipfile.ZipFile(zip_path, "r") as z, out_path.open("w", encoding="utf-8") as out:
        for ecli_id, paths in tqdm(groups.items(), desc="Processing"):
            if "ECLI_IT_CASS" not in ecli_id:
                n_no_cass += 1
                continue

            if "info" not in paths or "metadata" not in paths:
                n_missing += 1
                continue

            try:
                with z.open(paths["info"]) as f:
                    info = json.loads(_decode(f.read()))
                with z.open(paths["metadata"]) as f:
                    meta = json.loads(_decode(f.read()))
            except Exception:
                n_invalid += 1
                continue

            cr = meta.get("crMetadata") or {}
            division = cr.get("division") or ""
            legal_area = (cr.get("legalArea") or "").strip().upper()

            if division in EXCLUDED_DIVISIONS:
                n_excluded += 1
                continue

            macro_area = classify_cassazione_macro_area(legal_area, division)
            if macro_area is None:
                n_no_map += 1
                combo = (legal_area or "(vuota)", division or "(vuota)")
                unknown_combos[combo] = unknown_combos.get(combo, 0) + 1
                continue

            facts = info.get("facts") or ""
            if len(facts) < MIN_FACTS_LENGTH:
                n_short += 1
                continue

            principles = info.get("principles") or []
            if len(principles) < MIN_PRINCIPLES:
                n_no_princ += 1
                continue

            try:
                prov = Provvedimento(
                    ecli_id=ecli_id,
                    macro_area=macro_area,
                    division=division,
                    legal_area=legal_area,
                    doc_type=cr.get("docType") or "",
                    date=cr.get("publicationDate") or "",
                    facts=facts,
                    principles=principles,
                    decision=info.get("decision"),
                )
            except ValidationError as exc:
                log.debug("Validation error %s: %s", ecli_id, exc)
                n_invalid += 1
                continue

            out.write(prov.model_dump_json() + "\n")
            n_written += 1

    log.info("=" * 50)
    log.info("Scritti:               %6d", n_written)
    log.info("Esclusi (Sez. 7):      %6d", n_excluded)
    log.info("Esclusi (non CASS):    %6d", n_no_cass)
    log.info("Esclusi (combo scon):  %6d", n_no_map)
    log.info("Esclusi (facts corto): %6d", n_short)
    log.info("Esclusi (no princip):  %6d", n_no_princ)
    log.info("Errori (file manc):    %6d", n_missing)
    log.info("Errori (validazione):  %6d", n_invalid)
    for (legal_area, division), count in sorted(unknown_combos.items()):
        log.warning(
            "Combinazione Cassazione esclusa: legalArea=%s division=%s (%d)",
            legal_area,
            division,
            count,
        )
    log.info("Output: %s", out_path)
    log.info("=" * 50)

    return n_written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Processa il corpus di sentenze.")
    parser.add_argument("--zip", type=Path, default=CORPUS_ZIP)
    parser.add_argument("--out", type=Path, default=CORPUS_JSONL)
    parser.add_argument("--limit", type=int, default=None, help="Limita il numero di provvedimenti.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    n = build_corpus(args.zip, args.out, args.limit)
    log.info("Completato. %d provvedimenti scritti.", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
