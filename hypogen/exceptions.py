"""HypoGen exception hierarchy.

All project-specific exceptions derive from ``HypoGenError`` so callers can
catch the whole family with a single ``except HypoGenError`` clause.
"""

from __future__ import annotations


class HypoGenError(Exception):
    """Base class for all HypoGen exceptions."""


class ProviderError(HypoGenError):
    """Raised when an LLM provider is unreachable or all retry attempts fail.

    Example
    -------
    ::

        raise ProviderError(
            "Ollama not running — start with: ollama serve",
            provider="ollama",
            model="llama3.1",
        )
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        detail = f" (provider={provider}, model={model})" if provider else ""
        super().__init__(f"{message}{detail}")


class CorpusLoadError(HypoGenError):
    """Raised when a paper or corpus directory cannot be read.

    Example
    -------
    ::

        raise CorpusLoadError("No .txt files found", path="/data/papers/")
    """

    def __init__(self, message: str, *, path: str | None = None) -> None:
        self.path = path
        detail = f" (path={path})" if path else ""
        super().__init__(f"{message}{detail}")


class ExtractionError(HypoGenError):
    """Raised when causal claim extraction from an LLM response fails.

    This covers both JSON parse failures and Pydantic validation errors
    after the fallback retry has also failed.

    Example
    -------
    ::

        raise ExtractionError(
            "Failed to parse LLM response after retry",
            paper="paper1.txt",
            raw_output="...",
        )
    """

    def __init__(
        self,
        message: str,
        *,
        paper: str | None = None,
        raw_output: str | None = None,
    ) -> None:
        self.paper = paper
        self.raw_output = raw_output
        detail = f" (paper={paper})" if paper else ""
        super().__init__(f"{message}{detail}")


class NoGapsFoundError(HypoGenError):
    """Raised when the gap detector finds no structural gaps in the graph.

    This typically means the knowledge graph is too sparse (too few edges)
    to have meaningful structural gaps, or all nodes are well-connected.

    Example
    -------
    ::

        raise NoGapsFoundError(
            "Graph has only 2 nodes — add more papers to enable gap detection",
            node_count=2,
            edge_count=1,
        )
    """

    def __init__(
        self,
        message: str,
        *,
        node_count: int | None = None,
        edge_count: int | None = None,
    ) -> None:
        self.node_count = node_count
        self.edge_count = edge_count
        detail = ""
        if node_count is not None:
            detail = f" (nodes={node_count}, edges={edge_count})"
        super().__init__(f"{message}{detail}")
