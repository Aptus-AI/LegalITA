from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from evaluation.citation_grounding import QuestionProfile, _ground_citation, main
from evaluation.citations.local_registry import LocalRegistryIndex
from evaluation.citations.local_resolver import LocalCitationResolver
from evaluation.citations.models import Citation
from evaluation.citations.structured_urls import extract_structured_url_citations


ECLI = "ECLI:IT:CASS:2024:1234CIV"


def build_registry(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE registry (ecli TEXT PRIMARY KEY, namespace TEXT, court TEXT, "
        "year INTEGER, number INTEGER, legal_area TEXT, variant TEXT, doc_type TEXT, "
        "sector TEXT, publication_date TEXT)"
    )
    connection.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
    connection.execute(
        "INSERT INTO registry VALUES (?,?,?,?,?,?,?,?,?,?)",
        (ECLI, "CASS", "CASS", 2024, 1234, "CIV", None, "SENTENZA", "1", "2024-05-02"),
    )
    connection.execute(
        "INSERT INTO registry VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("ECLI:IT:CASS:2024:1234PEN", "CASS", "CASS", 2024, 1234, "PEN", None, None, None, None),
    )
    connection.executemany(
        "INSERT INTO meta VALUES (?,?)",
        [("built_at", "2026-09-03T08:11:21+00:00"), ("namespaces", "CASS")],
    )
    connection.commit()
    connection.close()


def build_profiles(path: Path) -> None:
    path.mkdir()
    (path / "index.json").write_text(
        json.dumps({"schema_version": "profiles-1", "generated_at": "2026-09-03"}),
        encoding="utf-8",
    )
    (path / "diritto_civile_0001.json").write_text(
        json.dumps(
            {
                "task_id": "diritto_civile/0001",
                "question": "Domanda di prova",
                "resolving_rulings": [{"ecli": ECLI, "evidence_tiers": ["A_criteria_slot"]}],
                "retrieved_marginal": [],
                "retrieved_irrelevant": [],
                "retrieved_unassessed": [],
            }
        ),
        encoding="utf-8",
    )


class LocalResolverTest(unittest.TestCase):
    def test_resolves_an_explicit_ecli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry.sqlite"
            build_registry(registry)
            with LocalRegistryIndex(registry) as index:
                result = LocalCitationResolver(index=index).resolve(
                    Citation(text=ECLI, span=(0, len(ECLI)), ecli=ECLI)
                )
        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.matched_ecli, ECLI)
        self.assertEqual(result.existence_source, "local_registry")

    def test_number_year_homonyms_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry.sqlite"
            build_registry(registry)
            with LocalRegistryIndex(registry) as index:
                result = LocalCitationResolver(index=index).resolve(
                    Citation(
                        text="Cass. n. 1234/2024",
                        span=(0, 19),
                        suggested_namespace="CASS",
                        number="1234",
                        year=2024,
                    )
                )
        self.assertEqual(result.status, "ambiguous")
        self.assertEqual(result.identity_status, "ambiguous")

    def test_ecli_slug_is_extracted_from_any_case_law_url(self) -> None:
        citations = extract_structured_url_citations(
            "[decisione](https://example.test/case-law/ECLI_IT_CASS_2024_1234CIV)"
        )
        self.assertEqual([citation.ecli for citation in citations], [ECLI])


class GroundingCliTest(unittest.TestCase):
    def test_metadata_conflict_never_receives_grounding_credit(self) -> None:
        profile = QuestionProfile(
            task_id="diritto_civile/0001",
            path=Path("profile.json"),
            question="",
            resolving=frozenset({ECLI}),
            retrieved=frozenset({ECLI}),
            tiers={ECLI: ("A_criteria_slot",)},
        )
        grounded = _ground_citation(
            {
                "final_status": "resolved_local_registry_metadata_mismatch",
                "matched_ecli": ECLI,
                "citation": {"text": ECLI, "ecli": ECLI},
            },
            profile=profile,
            coverage={},
        )

        self.assertEqual(grounded["gold_v3_relation"], "unresolved")
        self.assertEqual(grounded["issue_profile_match"], [])

    def test_local_run_writes_reports_and_prints_only_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = root / "registry.sqlite"
            profiles = root / "profiles"
            results = root / "outputs.json"
            out_dir = root / "report"
            build_registry(registry)
            build_profiles(profiles)
            results.write_text(
                json.dumps(
                    [
                        {
                            "task_id": "diritto_civile/0001",
                            "model": "test-model",
                            "model_output": f"Si veda {ECLI}.",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--results",
                        str(results),
                        "--citation-registry",
                        str(registry),
                        "--question-profiles",
                        str(profiles),
                        "--out-dir",
                        str(out_dir),
                        "--n-tasks",
                        "1",
                        "--fast-path",
                    ]
                )

            printed = stdout.getvalue()
            report = json.loads((out_dir / "citation_grounding_v3.json").read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertIn("GOG=100.0%  Coverage=100.0%  Tasks=1", printed)
        self.assertIn("backend=local  registry=2026-09-03", printed)
        self.assertIn("results=", printed)
        self.assertNotIn("citations_total", printed)
        self.assertEqual(report["summary"]["gog"], 1.0)
        self.assertEqual(report["tasks"][0]["citations"][0]["gold_v3_relation"], "issue_aligned")


if __name__ == "__main__":
    unittest.main()
