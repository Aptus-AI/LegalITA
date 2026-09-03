"""Compatibility CLI; prefer the ``legalita-build-corpus`` command."""

from benchmark.corpus_builder import *  # noqa: F401,F403
from benchmark.corpus_builder import main


if __name__ == "__main__":
    raise SystemExit(main())
