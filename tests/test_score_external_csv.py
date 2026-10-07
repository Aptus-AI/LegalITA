from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from legal_ita.cli.score_external_csv import load_external_outputs


class ExternalCsvLoaderTest(unittest.TestCase):
    def test_loads_semicolon_export_with_multiline_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "answers.csv"
            path.write_text(
                'Task ID;Domanda;Risposte;Nota\n'
                'task-1;"Una domanda, con virgola";"Prima riga\nSeconda riga";ok\n',
                encoding="utf-8",
            )

            rows = load_external_outputs(path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows.iloc[0]["Domanda"], "Una domanda, con virgola")
        self.assertEqual(rows.iloc[0]["answer_clean"], "Prima riga\nSeconda riga")

    def test_loads_comma_separated_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "answers.csv"
            path.write_text(
                'Task ID,Domanda,Risposte\n'
                'task-1,"Una domanda, con virgola",Una risposta\n',
                encoding="utf-8",
            )

            rows = load_external_outputs(path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows.iloc[0]["answer_clean"], "Una risposta")


if __name__ == "__main__":
    unittest.main()
