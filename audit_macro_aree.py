"""Compatibility CLI; prefer the ``legalita-audit-macro-aree`` command."""

from legal_ita.cli.audit_macro_aree import *  # noqa: F401,F403
from legal_ita.cli.audit_macro_aree import main


if __name__ == "__main__":
    raise SystemExit(main())
