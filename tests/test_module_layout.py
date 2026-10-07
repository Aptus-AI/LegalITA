from __future__ import annotations

import tomllib
import unittest
from pathlib import Path

from legal_ita import config
from legal_ita.cli import benchmark


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONSOLE_SCRIPTS = {
    "legalita-audit-macro-aree",
    "legalita-benchmark",
    "legalita-build-corpus",
    "legalita-build-tasks",
    "legalita-charts",
    "legalita-grounding",
    "legalita-mdd",
    "legalita-score-csv",
    "legalita-score-mdd-csv",
}


def _pyproject() -> dict:
    return tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


class ModuleLayoutTest(unittest.TestCase):
    def test_config_paths_still_resolve_from_project_root(self) -> None:
        self.assertEqual(config.ROOT_DIR, PROJECT_ROOT)
        self.assertEqual(config.TASKS_DIR, PROJECT_ROOT / "tasks")
        self.assertEqual(config.RESULTS_DIR, PROJECT_ROOT / "results")

    def test_no_python_modules_in_project_root(self) -> None:
        self.assertEqual(sorted(path.name for path in PROJECT_ROOT.glob("*.py")), [])
        self.assertNotIn("py-modules", _pyproject().get("tool", {}).get("setuptools", {}))

    def test_all_console_entry_points_target_cli_package(self) -> None:
        scripts = _pyproject()["project"]["scripts"]
        self.assertEqual(set(scripts), CONSOLE_SCRIPTS)
        self.assertTrue(all(target.startswith("legal_ita.cli.") for target in scripts.values()))

    def test_cli_parsers_can_be_built_without_running_a_pipeline(self) -> None:
        from legal_ita.cli.audit_macro_aree import build_parser as audit_parser
        from legal_ita.cli.build_corpus import main as corpus_main
        from legal_ita.cli.build_tasks import build_parser as tasks_parser
        from legal_ita.cli.grounding import build_parser as grounding_parser
        from legal_ita.cli.mdd import build_parser as mdd_parser
        from legal_ita.cli.score_external_csv import build_parser as score_csv_parser
        from legal_ita.cli.score_external_mdd import build_parser as mdd_csv_parser
        from benchmark.corpus_builder import build_parser as corpus_parser
        from evaluation.reporting.cli import build_parser as reporting_parser

        self.assertTrue(callable(corpus_main))
        factories = (
            benchmark.build_parser,
            corpus_parser,
            tasks_parser,
            reporting_parser,
            audit_parser,
            mdd_parser,
            mdd_csv_parser,
            score_csv_parser,
            grounding_parser,
        )
        for factory in factories:
            with self.subTest(factory=factory.__module__):
                self.assertIsNotNone(factory().format_help())


class MDDLegacyCompatibilityTest(unittest.TestCase):
    """I dati prodotti prima della rinomina bullshit -> MDD restano leggibili."""

    def test_legacy_task_type_is_normalized_on_load(self) -> None:
        from evaluation.mdd_judge import MDDTask

        task = MDDTask.model_validate(
            {
                "task_id": "bullshit/0041",
                "task_type": "bullshit",
                "macro_area": "Diritto del lavoro",
                "difficulty": "D4",
                "query": "Domanda",
                "criteria": [{"id": "C-001", "title": "t", "match_criteria": "m"}],
            }
        )
        self.assertEqual(task.task_type, "mdd")
        self.assertEqual(task.task_id, "bullshit/0041")

    def test_mdd_tasks_are_recognized_by_type_or_id_prefix(self) -> None:
        from evaluation.scoring import is_citation_scoring_applicable
        from legal_ita.schemas import is_mdd_task

        for task_type, task_id in (("mdd", "x"), ("bullshit", "x"), (None, "bullshit/0041"), (None, "mdd/0041")):
            with self.subTest(task_type=task_type, task_id=task_id):
                self.assertTrue(is_mdd_task(task_type, task_id))
                self.assertFalse(is_citation_scoring_applicable({"task_type": task_type, "task_id": task_id}))
        self.assertFalse(is_mdd_task(None, "diritto_civile/0001"))
        self.assertTrue(is_citation_scoring_applicable({"task_type": "standard", "task_id": "bullshit/0041"}))

    def test_reporting_reads_legacy_and_new_results_dirs(self) -> None:
        from evaluation.reporting.cli import _path_family, _path_model

        results = Path("/r")
        for subdir in ("bullshit", "mdd"):
            path = results / subdir / "gpt-4o" / "20260101-000000" / "scores.json"
            with self.subTest(subdir=subdir):
                self.assertEqual(_path_model(path, results), "gpt-4o")
                self.assertEqual(_path_family(path, results), "mdd")


if __name__ == "__main__":
    unittest.main()
