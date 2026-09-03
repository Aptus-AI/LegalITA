"""Compatibility import; use :mod:`legal_ita.grounding.service` in new code."""

from legal_ita.grounding.service import *  # noqa: F401,F403
from legal_ita.grounding.service import main


if __name__ == "__main__":
    raise SystemExit(main())
