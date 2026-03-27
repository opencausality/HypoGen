"""Tests for the GapDetector."""

from __future__ import annotations

import pytest

from hypogen.data.schema import GapNode, KnowledgeEdge, KnowledgeGraph
from hypogen.knowledge.gap_detector import GapDetector


def _make_graph(edges: list[tuple[str, str, float, int]]) -> KnowledgeGraph:
    """Build a minimal KnowledgeGraph from (cause, effect, confidence, paper_count) tuples."""
    from datetime import datetime, timezone

    knowledge_edges = [
        KnowledgeEdge(
            cause=c,
            effect=e,
            avg_confidence=conf,
            paper_count=papers,
        )
        for c, e, conf, papers in edges
    ]

    nodes = list({n for c, e, _, _ in edges for n in (c, e)})

    return KnowledgeGraph(
        nodes=nodes,
        edges=knowledge_edges,
        papers_analyzed=len({e[0] for e in edges}),
        created_at=datetime.now(tz=timezone.utc).isoformat(),
    )


class TestGapDetector:
    def test_detects_missing_cause(self) -> None:
        """Nodes with effects but no causes should be detected as MISSING_CAUSE gaps."""
        graph = _make_graph([("a", "b", 0.9, 1)])
        detector = GapDetector()
        gaps = detector.detect(graph)

        # "a" has no incoming edges → MISSING_CAUSE candidate
        missing_cause_vars = [g.variable for g in gaps if g.gap_type == "MISSING_CAUSE"]
        assert "a" in missing_cause_vars

    def test_detects_missing_effect(self) -> None:
        """Leaf nodes with causes but no effects are MISSING_EFFECT gaps."""
        graph = _make_graph([("a", "b", 0.9, 1)])
        detector = GapDetector()
        gaps = detector.detect(graph)

        missing_effect_vars = [g.variable for g in gaps if g.gap_type == "MISSING_EFFECT"]
        assert "b" in missing_effect_vars

    def test_detects_low_confidence_chain(self) -> None:
        """Edges with confidence below threshold produce LOW_CONFIDENCE_CHAIN gaps."""
        graph = _make_graph([("x", "y", 0.3, 1)])  # very low confidence
        detector = GapDetector()
        gaps = detector.detect(graph)

        low_conf = [g for g in gaps if g.gap_type == "LOW_CONFIDENCE_CHAIN"]
        assert len(low_conf) >= 1

    def test_returns_list_of_gap_nodes(self, knowledge_graph_fixture: KnowledgeGraph) -> None:
        """detect() always returns a list of GapNode instances."""
        detector = GapDetector()
        gaps = detector.detect(knowledge_graph_fixture)
        assert isinstance(gaps, list)
        for gap in gaps:
            assert isinstance(gap, GapNode)

    def test_gap_scores_in_range(self, knowledge_graph_fixture: KnowledgeGraph) -> None:
        """All gap scores must be in [0, 1]."""
        detector = GapDetector()
        gaps = detector.detect(knowledge_graph_fixture)
        for gap in gaps:
            assert 0.0 <= gap.gap_score <= 1.0

    def test_gaps_sorted_descending(self, knowledge_graph_fixture: KnowledgeGraph) -> None:
        """Gaps are returned sorted by gap_score descending."""
        detector = GapDetector()
        gaps = detector.detect(knowledge_graph_fixture)
        if len(gaps) > 1:
            scores = [g.gap_score for g in gaps]
            assert scores == sorted(scores, reverse=True)

    def test_empty_graph_returns_empty_gaps(self) -> None:
        """Empty graph has no gaps."""
        from datetime import datetime, timezone

        graph = KnowledgeGraph(
            nodes=[],
            edges=[],
            papers_analyzed=0,
            created_at=datetime.now(tz=timezone.utc).isoformat(),
        )
        detector = GapDetector()
        gaps = detector.detect(graph)
        assert gaps == []
