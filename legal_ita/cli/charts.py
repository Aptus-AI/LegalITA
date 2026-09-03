"""CLI entry point for reports and charts."""

from evaluation.reporting.cli import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
