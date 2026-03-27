"""Gap scorer — assigns an importance score to a detected knowledge gap.

The score reflects how valuable it would be to fill a particular gap:
- Gaps in highly connected regions of the graph are more important (filling
  them would affect many downstream conclusions).
- Gaps involving variables that appear across many papers are more important
  (they are central to the field, not a niche detail).
- The base importance also depends on gap type (MISSING_CAUSE in a hub node
  is more important than ISOLATED_NODE on a peripheral variable).

Score is in [0.0, 1.0]. The raw value is clipped to this range by the caller
(``GapDetector._make_gap``).
"""

from __future__ import annotations

import logging
import math

import networkx as nx

from hypogen.data.schema import GapNode

logger = logging.getLogger("hypogen.gaps.scorer")

# Base importance weight per gap type.
# These reflect the typical scientific value of resolving each type of gap.
_GAP_TYPE_WEIGHTS: dict[str, float] = {
    "MISSING_CAUSE": 0.75,
    "MISSING_EFFECT": 0.65,
    "ISOLATED_NODE": 0.45,
    "LOW_CONFIDENCE_CHAIN": 0.60,
    "BROKEN_CHAIN": 0.70,
}


def score_gap(gap_node: GapNode, graph: nx.DiGraph) -> float:
    """Compute an importance score for a structural gap in the knowledge graph.

    The score is a weighted combination of three signals:

    1. **Neighbour connectivity** (40%): How many total edges do the neighbours
       of the gap variable have? Highly connected neighbours mean this gap sits
       in a dense, important region of the knowledge graph.

    2. **Cross-paper frequency** (30%): How many papers mention edges involving
       this node? A variable cited across many papers is scientifically central.
       Measured via the ``paper_count`` edge attribute.

    3. **Gap-type base weight** (30%): Different gap types have intrinsically
       different scientific value (see ``_GAP_TYPE_WEIGHTS``).

    Parameters
    ----------
    gap_node:
        The ``GapNode`` to score. Its ``neighboring_nodes`` field is used to
        look up graph connectivity.
    graph:
        The full NetworkX DiGraph with edge attributes.

    Returns
    -------
    float
        A raw score in roughly [0.0, 1.5] that is then clipped to [0.0, 1.0]
        by the caller. Higher is more important.
    """
    variable = gap_node.variable
    gap_type = gap_node.gap_type
    neighbors = gap_node.neighboring_nodes

    # ── Signal 1: neighbour connectivity ──────────────────────────────────
    # Sum the degree of all neighbouring nodes (excluding the gap node itself).
    # Normalise by dividing by the expected degree in a moderately connected graph.
    neighbour_degree_sum = 0
    for neighbour in neighbors:
        if graph.has_node(neighbour):
            neighbour_degree_sum += graph.degree(neighbour)

    # Normalise: use a log scale to dampen outliers. log(1 + x) / log(1 + 20) maps
    # 0 neighbours→0, ~20 total neighbour edges→1.0, >20→slightly above 1.
    connectivity_score = math.log1p(neighbour_degree_sum) / math.log1p(20.0)

    # ── Signal 2: cross-paper frequency ───────────────────────────────────
    # Count total paper_count for all edges touching this node.
    total_paper_mentions = 0
    if graph.has_node(variable):
        for _, _, attrs in graph.out_edges(variable, data=True):
            total_paper_mentions += attrs.get("paper_count", 1)
        for _, _, attrs in graph.in_edges(variable, data=True):
            total_paper_mentions += attrs.get("paper_count", 1)

    # Normalise: log(1 + x) / log(1 + 10) maps 0→0, ~10 papers→1.0.
    paper_score = math.log1p(total_paper_mentions) / math.log1p(10.0)

    # ── Signal 3: gap-type base weight ────────────────────────────────────
    type_score = _GAP_TYPE_WEIGHTS.get(gap_type, 0.5)

    # ── Weighted combination ───────────────────────────────────────────────
    raw_score = 0.40 * connectivity_score + 0.30 * paper_score + 0.30 * type_score

    logger.debug(
        "Gap score for '%s' (%s): connectivity=%.3f, papers=%.3f, type=%.3f → raw=%.3f",
        variable,
        gap_type,
        connectivity_score,
        paper_score,
        type_score,
        raw_score,
    )

    return raw_score
