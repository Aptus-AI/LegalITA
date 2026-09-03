"""Compatibility CLI; prefer the ``legalita-benchmark`` command."""

from legal_ita.cli.benchmark import *  # noqa: F401,F403
from legal_ita.cli.benchmark import main


if __name__ == "__main__":
    raise SystemExit(main())
