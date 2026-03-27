"""Tests for the KnowledgeGraph builder."""

from __future__ import annotations

import pytest

from hypogen.data.schema import CausalClaim, KnowledgeEdge, KnowledgeGraph
from hypogen.graph.builder import GraphBuilder


def _make_claim(cause: str, effect: str, confidence: float, paper: str = "p.txt") -> CausalClaim:
    return CausalClaim(
        cause=cause,
        effect=effect,
        confidence=confidence,
        mechanism="test mechanism",
        source_paper=paper,
        evidence="test evidence",
    )


class TestGraphBuilder:
    def test_builds_graph_with_nodes_and_edges(self) -> None:
        """Builder produces a graph with the correct nodes and edges."""
        claims = [
            _make_claim("air pollution", "inflammation", 0.9),
            _make_claim("inflammation", "disease", 0.8),
        ]
        builder = GraphBuilder()
        graph = builder.build(claims)

        assert "air pollution" in graph.nodes
        assert "inflammation" in graph.nodes
        assert "disease" in graph.nodes
        assert len(graph.edges) == 2

    def test_duplicate_claims_merged(self) -> None:
        """Claims for the same (cause, effect) pair are merged into one edge."""
        claims = [
            _make_claim("smoking", "cancer", 0.9, "paper1.txt"),
            _make_claim("smoking", "cancer", 0.7, "paper2.txt"),
        ]
        builder = GraphBuilder()
        graph = builder.build(claims)

        # Should produce exactly one edge
        edges = [e for e in graph.edges if e.cause == "smoking" and e.effect == "cancer"]
        assert len(edges) == 1
        # Average of 0.9 and 0.7 = 0.8
        assert abs(edges[0].avg_confidence - 0.8) < 1e-6
        assert edges[0].paper_count == 2

    def test_papers_analyzed_count(self) -> None:
        """Builder counts distinct paper names correctly."""
        claims = [
            _make_claim("a", "b", 0.8, "p1.txt"),
            _make_claim("c", "d", 0.8, "p1.txt"),
            _make_claim("e", "f", 0.8, "p2.txt"),
        ]
        builder = GraphBuilder()
        graph = builder.build(claims)
        assert graph.papers_analyzed == 2

    def test_empty_claims_returns_empty_graph(self) -> None:
        """No claims → no nodes or edges."""
        builder = GraphBuilder()
        graph = builder.build([])
        assert graph.nodes == []
        assert graph.edges == []
        assert graph.papers_analyzed == 0

    def test_nodes_are_normalized_lowercase(self) -> None:
        """Node names are lowercased for deduplication."""
        claims = [
            _make_claim("Air Pollution", "Inflammation", 0.85),
        ]
        builder = GraphBuilder()
        graph = builder.build(claims)

        assert "air pollution" in graph.nodes
        assert "inflammation" in graph.nodes

    def test_below_threshold_claims_excluded(self) -> None:
        """Claims below min_confidence threshold are excluded."""
        from hypogen.config import HypoGenSettings

        settings = HypoGenSettings(min_confidence=0.7)
        claims = [
            _make_claim("a", "b", 0.9),
            _make_claim("c", "d", 0.5),  # below threshold
        ]
        builder = GraphBuilder(settings=settings)
        graph = builder.build(claims)

        edges = {(e.cause, e.effect) for e in graph.edges}
        assert ("a", "b") in edges
        assert ("c", "d") not in edges
