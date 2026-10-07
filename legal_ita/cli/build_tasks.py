"""CLI entry point for task generation."""

from benchmark.task_builder import build_parser, main

__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
