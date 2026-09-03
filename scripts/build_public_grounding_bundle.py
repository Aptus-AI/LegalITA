"""Build a minimal, shareable grounding bundle from local source artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any


PROFILE_LISTS = (
    "retrieved_marginal",
    "retrieved_irrelevant",
    "retrieved_unassessed",
)


def build_bundle(*, registry: Path, profiles: Path, out_dir: Path) -> Path:
    if out_dir.exists():
        raise FileExistsError(f"La destinazione esiste già: {out_dir}")
    if not registry.is_file():
        raise FileNotFoundError(registry)
    if not profiles.is_dir():
        raise FileNotFoundError(profiles)

    registry_dir = out_dir / "registry"
    profiles_dir = out_dir / "question_profiles"
    registry_dir.mkdir(parents=True)
    profiles_dir.mkdir()

    public_registry = registry_dir / "ecli_registry_v1.sqlite"
    _copy_registry(registry, public_registry)
    profile_index = _copy_profiles(profiles, profiles_dir)
    registry_info = _registry_info(public_registry)

    manifest = {
        "schema_version": "legalita-grounding-bundle-1",
        "built_at": registry_info.get("built_at"),
        "registry_rows": registry_info.get("row_count"),
        "namespaces": registry_info.get("namespaces"),
        "question_profiles": profile_index["task_count"],
        "files": {
            "registry/ecli_registry_v1.sqlite": _sha256(public_registry),
            "question_profiles/index.json": _sha256(profiles_dir / "index.json"),
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return out_dir


def _copy_registry(source: Path, destination: Path) -> None:
    source_connection = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    destination_connection = sqlite3.connect(destination)
    try:
        source_connection.backup(destination_connection)
        destination_connection.execute(
            "INSERT INTO meta(key, value) VALUES('source', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            ("LegalITA public ECLI registry snapshot",),
        )
        destination_connection.commit()
    finally:
        destination_connection.close()
        source_connection.close()


def _copy_profiles(source: Path, destination: Path) -> dict[str, Any]:
    source_index_path = source / "index.json"
    source_index = (
        json.loads(source_index_path.read_text(encoding="utf-8"))
        if source_index_path.exists()
        else {}
    )
    task_entries: dict[str, Any] = {}
    for path in sorted(source.glob("*.json")):
        if path.name == "index.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        task_id = str(payload.get("task_id") or "").strip()
        if not task_id:
            continue
        resolving = [
            {
                "ecli": item["ecli"],
                "evidence_tiers": list(item.get("evidence_tiers") or ()),
            }
            for item in payload.get("resolving_rulings") or []
            if item.get("ecli")
        ]
        public_profile: dict[str, Any] = {
            "schema_version": "legalita-question-profile-public-1",
            "task_id": task_id,
            "macro_area": payload.get("macro_area"),
            "question": payload.get("question"),
            "resolving_rulings": resolving,
        }
        for key in PROFILE_LISTS:
            public_profile[key] = [
                {"ecli": item["ecli"]}
                for item in payload.get(key) or []
                if item.get("ecli")
            ]
        output_path = destination / f"{task_id.replace('/', '_')}.json"
        output_path.write_text(
            json.dumps(public_profile, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        task_entries[task_id] = {
            "resolving": len(resolving),
            **{key: len(public_profile[key]) for key in PROFILE_LISTS},
        }

    index = {
        "schema_version": "legalita-question-profiles-public-index-1",
        "generated_at": source_index.get("generated_at"),
        "task_count": len(task_entries),
        "tasks": task_entries,
    }
    (destination / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return index


def _registry_info(path: Path) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        meta = dict(connection.execute("SELECT key, value FROM meta"))
        row_count = connection.execute("SELECT COUNT(*) FROM registry").fetchone()[0]
        namespaces = dict(
            connection.execute(
                "SELECT namespace, COUNT(*) FROM registry GROUP BY namespace ORDER BY namespace"
            )
        )
    finally:
        connection.close()
    return {
        "built_at": meta.get("built_at"),
        "row_count": row_count,
        "namespaces": namespaces,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crea il bundle pubblico per il grounding locale.")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--zip", action="store_true", dest="make_zip")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = build_bundle(registry=args.registry, profiles=args.profiles, out_dir=args.out_dir)
    print(f"bundle={out_dir.resolve()}")
    if args.make_zip:
        archive = Path(shutil.make_archive(str(out_dir), "zip", root_dir=out_dir.parent, base_dir=out_dir.name))
        print(f"archive={archive.resolve()}")
        print(f"sha256={_sha256(archive)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
