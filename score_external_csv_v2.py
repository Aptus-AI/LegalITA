"""Compatibility CLI; prefer the ``legalita-score-csv`` command."""

from legal_ita.cli.score_external_csv import *  # noqa: F401,F403
from legal_ita.cli.score_external_csv import main


if __name__ == "__main__":
    raise SystemExit(main())
