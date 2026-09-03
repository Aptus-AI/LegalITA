"""Citation extraction and local registry resolution."""

from .local_registry import LocalRegistryIndex
from .local_resolver import LocalCitationResolver
from .models import Citation, RegistryCandidate, ResolutionResult
from .parser import CitationParser, parse_citations

__all__ = [
    "Citation",
    "CitationParser",
    "LocalCitationResolver",
    "LocalRegistryIndex",
    "RegistryCandidate",
    "ResolutionResult",
    "parse_citations",
]
