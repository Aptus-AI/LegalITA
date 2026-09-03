"""Compatibility CLI; prefer the ``legalita-charts`` command."""

from evaluation.reporting.cli import *  # noqa: F401,F403
from evaluation.reporting.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
