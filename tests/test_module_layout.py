from __future__ import annotations

import unittest
from pathlib import Path

import config as legacy_config
import run_benchmark as legacy_benchmark
from evaluation.scoring import score_batch
from evaluation.scoring.summary import summarize_batch_scores
from evaluation.scoring.tasks import score_batch as score_batch_from_tasks
from legal_ita import config
from legal_ita.cli import benchmark


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ModuleLayoutTest(unittest.TestCase):
    def test_config_paths_still_resolve_from_project_root(self) -> None:
        self.assertEqual(config.ROOT_DIR, PROJECT_ROOT)
        self.assertEqual(config.TASKS_DIR, PROJECT_ROOT / "tasks")
        self.assertEqual(legacy_config.RESULTS_DIR, PROJECT_ROOT / "results")

    def test_legacy_benchmark_wrapper_reexports_package_api(self) -> None:
        self.assertIs(legacy_benchmark.load_tasks, benchmark.load_tasks)

        from evaluation import citation_grounding as legacy_grounding
        from legal_ita.grounding import service as grounding

        self.assertIs(legacy_grounding.main, grounding.main)

    def test_scoring_facades_reexport_one_implementation(self) -> None:
        self.assertIs(score_batch, score_batch_from_tasks)
        self.assertEqual(summarize_batch_scores.__module__, "evaluation.scoring.service")

    def test_all_console_entry_points_target_package_modules(self) -> None:
        project = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        scripts_block = project.split("[project.scripts]", 1)[1].split("\n[", 1)[0]
        scripts = {
            key.strip(): value.strip().strip('"')
            for line in scripts_block.splitlines()
            if "=" in line
            for key, value in [line.split("=", 1)]
        }
        self.assertEqual(
            set(scripts),
            {
                "legalita-audit-macro-aree",
                "legalita-benchmark",
                "legalita-build-corpus",
                "legalita-bullshit",
                "legalita-charts",
                "legalita-grounding",
                "legalita-score-bullshit-csv",
                "legalita-score-csv",
            },
        )
        self.assertTrue(all(target.startswith(("legal_ita.", "benchmark.", "evaluation.")) for target in scripts.values()))

    def test_cli_parsers_can_be_built_without_running_a_pipeline(self) -> None:
        from benchmark.corpus_builder import build_parser as corpus_parser
        from evaluation.reporting.cli import build_parser as reporting_parser
        from legal_ita.cli.audit_macro_aree import build_parser as audit_parser
        from legal_ita.cli.bullshit import build_parser as bullshit_parser
        from legal_ita.cli.score_external_bullshit import build_parser as bullshit_csv_parser
        from legal_ita.cli.score_external_csv import build_parser as score_csv_parser
        from legal_ita.grounding.service import build_parser as grounding_parser

        factories = (
            benchmark.build_parser,
            corpus_parser,
            reporting_parser,
            audit_parser,
            bullshit_parser,
            bullshit_csv_parser,
            score_csv_parser,
            grounding_parser,
        )
        for factory in factories:
            with self.subTest(factory=factory.__module__):
                self.assertIsNotNone(factory().format_help())


if __name__ == "__main__":
    unittest.main()
