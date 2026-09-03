"""Compatibility CLI; prefer the ``legalita-bullshit`` command."""

from legal_ita.cli.bullshit import *  # noqa: F401,F403
from legal_ita.cli.bullshit import main


if __name__ == "__main__":
    raise SystemExit(main())
