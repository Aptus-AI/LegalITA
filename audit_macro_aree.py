"""
Audit rapido delle macro-aree del corpus Cassazione.

Uso:
    python .\audit_macro_aree.py
    python .\audit_macro_aree.py --area diritto_civile --examples 10
    python .\audit_macro_aree.py --area civile_generale --examples 10

Output, solo se eseguito:
    audit_macro_aree_dettaglio.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from config import CORPUS_JSONL, MACRO_AREE
from taxonomy import classify_cassazione_macro_area, normalize_macro_area


AREE_FOCUS = {
    "diritto_civile",
    "diritto_lavoro",
    "diritto_penale",
    "diritto_tributario",
}


def preview(value: object, length: int = 180) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:length] + ("..." if len(text) > length else "")


def load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Corpus non trovato: {path}\n"
            "Esegui prima: python .\\build_corpus.py"
        )

    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise ValueError("Il corpus.jsonl e vuoto.")

    df = pd.DataFrame(records)

    for column in ["ecli_id", "macro_area", "division", "legal_area", "doc_type", "date", "facts", "principles"]:
        if column not in df.columns:
            df[column] = ""

    df["macro_area"] = df["macro_area"].map(lambda value: normalize_macro_area(value, strict=False))
    df["ecli_suffix"] = df["ecli_id"].astype(str).str.extract(r"(CIV|PEN)$", expand=False).fillna("")
    df["legal_area"] = df["legal_area"].fillna("").astype(str).str.upper()
    df["classified_macro_area"] = df.apply(
        lambda row: classify_cassazione_macro_area(row["legal_area"], row["division"]) or "",
        axis=1,
    )

    df["facts_preview"] = df["facts"].map(preview)
    df["principles_preview"] = df["principles"].map(
        lambda value: preview(value[0] if isinstance(value, list) and value else "", 180)
    )
    return df


def print_mapping() -> None:
    print("\n=== MACRO-AREE CANONICHE ===")
    for slug, label in MACRO_AREE.items():
        print(f"{slug:<24} -> {label}")

    print("\n=== CLASSIFICAZIONE CASSAZIONE ===")
    print("PEN + Sez. 1/2/3/4/5/6/U -> diritto_penale")
    print("CIV + Sez. 1/2/3/U       -> diritto_civile")
    print("CIV + Sez. 5             -> diritto_tributario")
    print("CIV + Sez. L             -> diritto_lavoro")
    print("Sez. 7                   -> ESCLUSA")


def print_cross_tables(df: pd.DataFrame) -> None:
    print("\n=== DISTRIBUZIONE REALE: macro_area x legal_area ===")
    print(pd.crosstab(df["macro_area"], df["legal_area"], margins=True).to_string())

    print("\n=== CONTROLLO INCROCIATO: macro_area x suffisso ECLI (CIV/PEN) ===")
    print(pd.crosstab(df["macro_area"], df["ecli_suffix"], margins=True).to_string())

    print("\n=== DIVISION x legal_area ===")
    print(pd.crosstab(df["division"], df["legal_area"], margins=True).to_string())


def print_warnings(df: pd.DataFrame) -> None:
    print("\n=== ALERT MAPPATURA ===")
    mismatches = df[
        (df["classified_macro_area"] != "")
        & (df["macro_area"] != df["classified_macro_area"])
    ]

    if mismatches.empty:
        print("Nessuna divergenza tra macro_area salvata e classificazione corrente.")
        return

    print(f"[ATTENZIONE] Divergenze trovate: {len(mismatches)}")
    for _, row in mismatches.head(20).iterrows():
        print(
            f"- {row['ecli_id']}: salvata={row['macro_area']} "
            f"classificata={row['classified_macro_area']} "
            f"legal_area={row['legal_area']} division={row['division']}"
        )


def print_examples(df: pd.DataFrame, area: str | None, examples: int) -> None:
    if area:
        requested = normalize_macro_area(area, strict=True)
        areas = [requested]
    else:
        areas = [current for current in sorted(AREE_FOCUS) if current in set(df["macro_area"])]

    print("\n=== ESEMPI DI CONTENUTO PER AREA ===")
    for current_area in areas:
        subset = df[df["macro_area"] == current_area].copy()
        if subset.empty:
            print(f"\n--- {current_area}: nessun provvedimento ---")
            continue

        print(f"\n--- {current_area}: {len(subset)} provvedimenti ---")
        print("legal_area:", subset["legal_area"].value_counts(dropna=False).to_dict())

        for _, row in subset.head(examples).iterrows():
            print(
                f"\n{row['ecli_id']} | division={row['division']} | "
                f"legal_area={row['legal_area']} | date={row['date']}"
            )
            print(f"  fatti:     {row['facts_preview']}")
            print(f"  principio: {row['principles_preview']}")


def export_csv(df: pd.DataFrame, output_path: Path) -> None:
    columns = [
        "macro_area",
        "classified_macro_area",
        "division",
        "legal_area",
        "ecli_suffix",
        "date",
        "doc_type",
        "ecli_id",
        "facts_preview",
        "principles_preview",
    ]
    df[columns].sort_values(["macro_area", "legal_area", "division", "date", "ecli_id"]).to_csv(
        output_path, index=False, encoding="utf-8-sig"
    )
    print(f"\nCSV esportato: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlla la mappatura delle macro-aree del corpus Cassazione.")
    parser.add_argument("--area", type=str, default=None, help="Mostra esempi solo per una macro-area specifica.")
    parser.add_argument("--examples", type=int, default=5, help="Numero di esempi mostrati per area.")
    parser.add_argument("--all-areas", action="store_true", help="Analizza tutte le aree canoniche.")
    args = parser.parse_args()

    df = load_dataframe(CORPUS_JSONL)
    audited = df if args.all_areas else df[df["macro_area"].isin(AREE_FOCUS)].copy()

    print_mapping()
    print(f"\nCorpus letto: {CORPUS_JSONL}")
    print(f"Provvedimenti analizzati: {len(audited)} / {len(df)} complessivi")

    print_cross_tables(audited)
    print_warnings(audited)
    print_examples(audited, args.area, args.examples)
    export_csv(audited, Path("audit_macro_aree_dettaglio.csv"))


if __name__ == "__main__":
    main()
