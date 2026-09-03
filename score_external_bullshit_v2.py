"""Compatibility CLI; prefer the ``legalita-score-bullshit-csv`` command."""

from legal_ita.cli.score_external_bullshit import *  # noqa: F401,F403
from legal_ita.cli.score_external_bullshit import main


if __name__ == "__main__":
    raise SystemExit(main())
