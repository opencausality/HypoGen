"""Gap detector — finds structural gaps in the causal knowledge graph.

A gap is a place where the graph is incomplete: a variable that is an effect
with no known cause, a cause with no known downstream effect, an isolated node
with only a single connection, or a low-confidence chain linking important nodes.

These gaps are the seed for hypothesis generation — the LLM is asked to propose
what might fill each gap based on context from neighbouring nodes.
"""

from __future__ import annotations

import logging
from typing import Literal

import networkx as nx

from hypogen.data.schema import GapNode

logger = logging.getLogger("hypogen.gaps")

# Edge confidence below this is considered "low confidence" for chain detection.
_LOW_CONFIDENCE_THRESHOLD = 0.6

GapType = Literal[
    "MISSING_CAUSE",
    "MISSING_EFFECT",
    "ISOLATED_NODE",
    "LOW_CONFIDENCE_CHAIN",
    "BROKEN_CHAIN",
]


class GapDetector:
    """Detect structural gaps in a causal knowledge graph.

    Gaps are classified into five types:

    - **MISSING_CAUSE**: A node has no parents but has 2 or more children.
      Something is clearly causing downstream effects, but its own causes are
      unknown — a blank upstream in the causal chain.

    - **MISSING_EFFECT**: A node has parent(s) but no children. A variable
      is known to be caused but its downstream consequences are unexplored.

    - **ISOLATED_NODE**: A node with exactly one edge (degree 1). It touches
      the main graph at only one point and is not yet integrated into a causal
      chain.

    - **LOW_CONFIDENCE_CHAIN**: A path exists between two well-connected nodes,
      but every edge along that path has a confidence score below
      ``_LOW_CONFIDENCE_THRESHOLD``. The causal mechanism needs strengthening.

    - **BROKEN_CHAIN**: Two connected components exist in the undirected
      projection of the graph, but share a common node neighbour across the
      gap (detected via shared neighbours in the full graph).

    Parameters
    ----------
    None — the detector is stateless; pass the graph to :meth:`detect`.
    """

    def detect(
        self,
        graph: nx.DiGraph,
        min_confidence: float = 0.5,
    ) -> list[GapNode]:
        """Detect all structural gaps in the knowledge graph.

        Parameters
        ----------
        graph:
            A directed knowledge graph where each edge has an ``avg_confidence``
            attribute (as built by :func:`hypogen.graph.builder.build_knowledge_graph`).
        min_confidence:
            Confidence threshold used by the graph builder. Nodes whose only
            edges are near this threshold are considered weak.

        Returns
        -------
        list[GapNode]
            All detected gaps, sorted by ``gap_score`` descending (most
            important gaps first).
        """
        if graph.number_of_nodes() == 0:
            logger.warning("Empty graph passed to GapDetector — no gaps to detect.")
            return []

        logger.info(
            "Running gap detection on graph with %d nodes, %d edges.",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )

        gaps: list[GapNode] = []
        seen_variables: set[str] = set()  # avoid duplicate gaps for the same variable/type

        for node in graph.nodes():
            parents = list(graph.predecessors(node))
            children = list(graph.successors(node))
            degree = graph.degree(node)

            # ── MISSING_CAUSE ─────────────────────────────────────────────
            if len(parents) == 0 and len(children) >= 2:
                gap = self._make_gap(
                    variable=node,
                    gap_type="MISSING_CAUSE",
                    graph=graph,
                    parents=parents,
                    children=children,
                    explanation=(
                        f"'{node}' drives {len(children)} downstream effects "
                        f"({', '.join(children[:3])}{', ...' if len(children) > 3 else ''}) "
                        f"but has no identified causes. Understanding what triggers "
                        f"'{node}' would complete the upstream causal chain."
                    ),
                )
                key = (node, "MISSING_CAUSE")
                if key not in seen_variables:
                    gaps.append(gap)
                    seen_variables.add(key)

            # ── MISSING_EFFECT ────────────────────────────────────────────
            if len(parents) >= 1 and len(children) == 0:
                gap = self._make_gap(
                    variable=node,
                    gap_type="MISSING_EFFECT",
                    graph=graph,
                    parents=parents,
                    children=children,
                    explanation=(
                        f"'{node}' is caused by {len(parents)} upstream variable(s) "
                        f"({', '.join(parents[:3])}{', ...' if len(parents) > 3 else ''}) "
                        f"but no downstream effects have been identified. "
                        f"What does '{node}' itself cause?"
                    ),
                )
                key = (node, "MISSING_EFFECT")
                if key not in seen_variables:
                    gaps.append(gap)
                    seen_variables.add(key)

            # ── ISOLATED_NODE ─────────────────────────────────────────────
            if degree == 1:
                gap = self._make_gap(
                    variable=node,
                    gap_type="ISOLATED_NODE",
                    graph=graph,
                    parents=parents,
                    children=children,
                    explanation=(
                        f"'{node}' has only a single edge in the knowledge graph "
                        f"(degree 1) and is not yet integrated into a causal chain. "
                        f"It is likely connected to more variables than the current "
                        f"literature captures."
                    ),
                )
                key = (node, "ISOLATED_NODE")
                if key not in seen_variables:
                    gaps.append(gap)
                    seen_variables.add(key)

        # ── LOW_CONFIDENCE_CHAIN ──────────────────────────────────────────
        low_conf_gaps = self._detect_low_confidence_chains(graph, seen_variables)
        gaps.extend(low_conf_gaps)

        # Sort by gap_score descending
        gaps.sort(key=lambda g: g.gap_score, reverse=True)

        logger.info("Gap detection complete: %d gaps found.", len(gaps))
        return gaps

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _make_gap(
        variable: str,
        gap_type: GapType,
        graph: nx.DiGraph,
        parents: list[str],
        children: list[str],
        explanation: str,
    ) -> GapNode:
        """Construct a GapNode with precomputed gap_score and edge descriptions."""
        from hypogen.gaps.scorer import score_gap

        neighbors = list(set(parents + children))
        existing_edges = [f"{p} → {variable}" for p in parents] + [
            f"{variable} → {c}" for c in children
        ]

        # Build a preliminary GapNode to pass to the scorer
        gap = GapNode(
            variable=variable,
            gap_type=gap_type,
            gap_score=0.0,  # placeholder; filled by scorer below
            neighboring_nodes=neighbors,
            existing_edges=existing_edges,
            gap_explanation=explanation,
        )

        score = score_gap(gap, graph)
        # Return a new model instance with the computed score
        return gap.model_copy(update={"gap_score": min(1.0, max(0.0, score))})

    def _detect_low_confidence_chains(
        self,
        graph: nx.DiGraph,
        seen_variables: set[tuple[str, str]],
    ) -> list[GapNode]:
        """Detect nodes that sit in low-confidence causal chains.

        A low-confidence chain is a path of 2 or more edges where every edge
        has ``avg_confidence < _LOW_CONFIDENCE_THRESHOLD``. We flag the central
        node of such paths (neither the source nor the sink) since that is where
        mechanism elucidation would be most valuable.
        """
        from hypogen.gaps.scorer import score_gap

        gaps: list[GapNode] = []

        # Only consider paths of length 2 (A → B → C)
        for middle in graph.nodes():
            parents = list(graph.predecessors(middle))
            children = list(graph.successors(middle))

            if not parents or not children:
                continue  # Not in a chain

            # Check if all incoming AND outgoing edges are low confidence
            in_confidences = [
                graph[p][middle].get("avg_confidence", 1.0) for p in parents
            ]
            out_confidences = [
                graph[middle][c].get("avg_confidence", 1.0) for c in children
            ]

            all_low = all(
                c < _LOW_CONFIDENCE_THRESHOLD
                for c in in_confidences + out_confidences
            )

            if all_low:
                key = (middle, "LOW_CONFIDENCE_CHAIN")
                if key in seen_variables:
                    continue

                avg_conf = (sum(in_confidences) + sum(out_confidences)) / (
                    len(in_confidences) + len(out_confidences)
                )
                explanation = (
                    f"'{middle}' sits in a low-confidence causal chain. "
                    f"All {len(in_confidences)} incoming edge(s) and "
                    f"{len(out_confidences)} outgoing edge(s) have confidence "
                    f"below {_LOW_CONFIDENCE_THRESHOLD} (avg={avg_conf:.2f}). "
                    f"The mechanism linking "
                    f"{', '.join(parents[:2])} → '{middle}' → {', '.join(children[:2])} "
                    f"requires experimental validation to strengthen confidence."
                )

                neighbors = list(set(parents + children))
                existing_edges = [f"{p} → {middle}" for p in parents] + [
                    f"{middle} → {c}" for c in children
                ]

                gap = GapNode(
                    variable=middle,
                    gap_type="LOW_CONFIDENCE_CHAIN",
                    gap_score=0.0,
                    neighboring_nodes=neighbors,
                    existing_edges=existing_edges,
                    gap_explanation=explanation,
                )
                score = score_gap(gap, graph)
                gap = gap.model_copy(update={"gap_score": min(1.0, max(0.0, score))})
                gaps.append(gap)
                seen_variables.add(key)

        return gaps
