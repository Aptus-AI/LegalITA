"""CLI entry point for local citation grounding."""

from legal_ita.grounding.service import build_parser, main

__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
