"""Knowledge graph builder — constructs a NetworkX DiGraph from causal claims.

Responsibilities:
- Normalise node names (lowercase, strip punctuation) for deduplication.
- Merge claims by (cause, effect) pair via the merger module.
- Filter edges below confidence threshold or paper count threshold.
- Validate DAG structure (detect cycles and warn).
- Return both the NetworkX graph and the list of KnowledgeEdge objects.
"""

from __future__ import annotations

import logging
import re
import string

import networkx as nx

from hypogen.config import HypoGenSettings, get_settings
from hypogen.data.schema import CausalClaim, KnowledgeEdge
from hypogen.graph.merger import merge_claims

logger = logging.getLogger("hypogen.graph")


def normalize_node_name(name: str) -> str:
    """Normalise a variable name for consistent deduplication.

    Transformations:
    - Lowercase
    - Strip leading/trailing whitespace
    - Collapse internal whitespace to single space
    - Remove punctuation that typically comes from tokenisation artefacts
      (but preserve hyphens in compound terms like "hdl-cholesterol")

    Parameters
    ----------
    name:
        Raw variable name from LLM output.

    Returns
    -------
    str
        Normalised name.
    """
    name = name.lower().strip()
    # Remove trailing/leading punctuation except hyphens (compound terms)
    name = name.strip(string.punctuation.replace("-", ""))
    # Collapse whitespace
    name = re.sub(r"\s+", " ", name)
    return name


def build_knowledge_graph(
    claims: list[CausalClaim],
    min_confidence: float = 0.5,
    min_papers: int = 1,
) -> tuple[nx.DiGraph, list[KnowledgeEdge]]:
    """Build a cumulative knowledge graph from causal claims.

    Algorithm:
    1. Normalise all cause/effect node names.
    2. Merge claims with the same (cause, effect) pair:
       - avg_confidence = arithmetic mean of individual confidences
       - paper_count = number of distinct source papers
       - mechanisms and evidence sentences aggregated
    3. Filter merged edges by ``min_confidence`` and ``min_papers``.
    4. Build NetworkX DiGraph with edge attributes.
    5. Detect and report cycles (warn but do not raise — remove cycle-forming
       edges rather than crashing, as LLMs sometimes hallucinate cycles).

    Parameters
    ----------
    claims:
        All causal claims from all papers.
    min_confidence:
        Minimum average confidence for an edge to be included.
    min_papers:
        Minimum number of papers that must support an edge.

    Returns
    -------
    tuple[nx.DiGraph, list[KnowledgeEdge]]
        ``(graph, edges)`` where ``graph`` is the NetworkX DiGraph and
        ``edges`` is the list of ``KnowledgeEdge`` objects that passed
        filtering.
    """
    if not claims:
        logger.warning("No claims provided to build_knowledge_graph — returning empty graph.")
        return nx.DiGraph(), []

    # Step 1: normalise node names in claims
    normalised_claims = [
        CausalClaim(
            cause=normalize_node_name(c.cause),
            effect=normalize_node_name(c.effect),
            confidence=c.confidence,
            mechanism=c.mechanism,
            source_paper=c.source_paper,
            evidence=c.evidence,
        )
        for c in claims
        if normalize_node_name(c.cause) != normalize_node_name(c.effect)  # skip self-loops
    ]

    # Step 2: merge by (cause, effect) pair
    merged_edges = merge_claims(normalised_claims)
    logger.info(
        "Merged %d claims into %d unique edges.", len(normalised_claims), len(merged_edges)
    )

    # Step 3: filter
    filtered_edges = [
        e
        for e in merged_edges
        if e.avg_confidence >= min_confidence and e.paper_count >= min_papers
    ]
    logger.info(
        "After filtering (min_confidence=%.2f, min_papers=%d): %d edges retained.",
        min_confidence,
        min_papers,
        len(filtered_edges),
    )

    # Step 4: build graph
    graph = nx.DiGraph()
    for edge in filtered_edges:
        graph.add_edge(
            edge.cause,
            edge.effect,
            avg_confidence=edge.avg_confidence,
            paper_count=edge.paper_count,
            mechanisms=edge.mechanisms,
        )

    # Step 5: detect and handle cycles
    filtered_edges = _remove_cycles(graph, filtered_edges)

    logger.info(
        "Knowledge graph built: %d nodes, %d edges.",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    return graph, filtered_edges


def _remove_cycles(
    graph: nx.DiGraph, edges: list[KnowledgeEdge]
) -> list[KnowledgeEdge]:
    """Detect cycles in the graph and remove lowest-confidence edges to break them.

    LLMs occasionally extract contradictory directions for the same relationship
    (e.g., both "A → B" and "B → A"), creating spurious cycles. We remove the
    lower-confidence edge from each cycle.
    """
    if nx.is_directed_acyclic_graph(graph):
        return edges

    cycles = list(nx.simple_cycles(graph))
    logger.warning(
        "Cycle(s) detected in knowledge graph: %d cycle(s). "
        "Removing lowest-confidence edges to break cycles.",
        len(cycles),
    )

    edges_to_remove: set[tuple[str, str]] = set()
    for cycle in cycles:
        # Find the edge in this cycle with the lowest confidence
        cycle_edges = list(zip(cycle, cycle[1:] + [cycle[0]]))
        min_conf = float("inf")
        weakest = None
        for u, v in cycle_edges:
            conf = graph[u][v].get("avg_confidence", 0.0)
            if conf < min_conf:
                min_conf = conf
                weakest = (u, v)
        if weakest:
            edges_to_remove.add(weakest)

    for u, v in edges_to_remove:
        graph.remove_edge(u, v)
        logger.warning("Removed cycle-forming edge: %s → %s", u, v)

    # Filter edge list to match
    removed_pairs = edges_to_remove
    remaining = [e for e in edges if (e.cause, e.effect) not in removed_pairs]
    return remaining
