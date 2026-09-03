"""CLI entry point for corpus preprocessing."""

from benchmark.corpus_builder import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
