from __future__ import annotations

import io
import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from legal_ita.cli import benchmark
from legal_ita.grounding.service import QuestionProfile, _ground_citation, main
from evaluation.citations.local_registry import LocalRegistryIndex
from evaluation.citations.local_resolver import LocalCitationResolver
from evaluation.citations.models import Citation
from evaluation.citations.structured_urls import extract_structured_url_citations
from scripts.build_public_grounding_bundle import build_bundle


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


class BenchmarkGroundingIntegrationTest(unittest.TestCase):
    """Il grounding offline e' parte di legalita-benchmark, salvo --skip-citation-grounding."""

    def test_missing_bundle_stops_before_any_api_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                "LEGALITA_CITATION_REGISTRY_PATH": str(Path(tmp) / "missing.sqlite"),
                "LEGALITA_QUESTION_PROFILES_DIR": str(Path(tmp) / "missing_profiles"),
            }
            previous = {key: os.environ.get(key) for key in env}
            os.environ.update(env)
            try:
                with self.assertRaises(SystemExit) as raised:
                    benchmark.run(models=["gpt-4o"], limit=1, skip_citation_grounding=False)
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

        message = str(raised.exception)
        self.assertIn("Registry locale non trovato", message)
        self.assertIn("Question profiles non trovati", message)
        self.assertIn("--skip-citation-grounding", message)

    def test_ground_run_writes_report_inside_run_dir_and_summary_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = root / "registry.sqlite"
            profiles = root / "profiles"
            run_dir = root / "results" / "test-model" / "run"
            run_dir.mkdir(parents=True)
            build_registry(registry)
            build_profiles(profiles)
            (run_dir / "scores.json").write_text(
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
            previous = os.environ.get("CITATION_EXTRACTOR_FAST_PATH")
            os.environ["CITATION_EXTRACTOR_FAST_PATH"] = "1"
            try:
                fields = benchmark.ground_run(run_dir, (registry, profiles), n_tasks=1)
            finally:
                if previous is None:
                    os.environ.pop("CITATION_EXTRACTOR_FAST_PATH", None)
                else:
                    os.environ["CITATION_EXTRACTOR_FAST_PATH"] = previous

            self.assertTrue((run_dir / "citation_grounding_v3.json").exists())
            self.assertTrue((run_dir / "citation_grounding_v3.md").exists())

        self.assertEqual(fields["citation_grounding_status"], "complete")
        self.assertEqual(fields["gog"], 1.0)
        self.assertEqual(fields["coverage"], 1.0)
        self.assertEqual(fields["gog_backend"], "local")
        self.assertEqual(fields["registry_built_at"], "2026-09-03T08:11:21+00:00")

    def test_ground_run_failure_does_not_invalidate_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            fields = benchmark.ground_run(
                run_dir, (run_dir / "missing.sqlite", run_dir / "missing"), n_tasks=1
            )
        self.assertEqual(fields["citation_grounding_status"], "error")
        self.assertIn("citation_grounding_error", fields)


class PublicBundleTest(unittest.TestCase):
    def test_builder_keeps_only_runtime_profile_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = root / "registry.sqlite"
            profiles = root / "profiles"
            bundle = root / "bundle"
            build_registry(registry)
            build_profiles(profiles)
            source_profile = profiles / "diritto_civile_0001.json"
            payload = json.loads(source_profile.read_text(encoding="utf-8"))
            payload["internal_notes"] = {"run": "company-only"}
            source_profile.write_text(json.dumps(payload), encoding="utf-8")

            build_bundle(registry=registry, profiles=profiles, out_dir=bundle)
            public_profile = json.loads(
                (bundle / "question_profiles/diritto_civile_0001.json").read_text(encoding="utf-8")
            )
            connection = sqlite3.connect(bundle / "registry/ecli_registry_v1.sqlite")
            source = connection.execute("SELECT value FROM meta WHERE key='source'").fetchone()[0]
            connection.close()

        self.assertNotIn("internal_notes", public_profile)
        self.assertEqual(source, "LegalITA public ECLI registry snapshot")


if __name__ == "__main__":
    unittest.main()
