"""Tests for the HypothesisGenerator."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from hypogen.data.schema import GapNode, Hypothesis, KnowledgeGraph
from hypogen.hypotheses.generator import HypothesisGenerator
from hypogen.hypotheses.ranker import composite_score, rank_hypotheses


def _make_gap(variable: str, gap_type: str = "MISSING_CAUSE", score: float = 0.8) -> GapNode:
    return GapNode(
        variable=variable,
        gap_type=gap_type,
        gap_score=score,
        neighboring_nodes=["inflammation"],
        existing_edges=["inflammation → disease"],
        gap_explanation=f"Test gap for {variable}",
    )


def _make_generator(response_json: str) -> HypothesisGenerator:
    adapter = MagicMock()
    adapter.complete.return_value = response_json
    return HypothesisGenerator(adapter=adapter)


class TestHypothesisGenerator:
    def test_returns_list_of_hypotheses(self, knowledge_graph_fixture: KnowledgeGraph) -> None:
        """Generator returns a list of Hypothesis instances."""
        payload = json.dumps(
            [
                {
                    "id": "H001",
                    "hypothesis_text": "We hypothesize that X causes Y via Z.",
                    "predicted_cause": "X",
                    "predicted_effect": "Y",
                    "predicted_mechanism": "Z pathway",
                    "testability_score": 0.85,
                    "novelty_score": 0.75,
                    "suggested_experiment": "RCT measuring Y after X intervention.",
                    "supporting_context": "air pollution → inflammation",
                }
            ]
        )
        generator = _make_generator(payload)
        hypotheses = generator.generate(
            knowledge_graph_fixture.gaps,
            knowledge_graph_fixture,
            max_hypotheses=5,
        )
        assert len(hypotheses) >= 1
        assert isinstance(hypotheses[0], Hypothesis)

    def test_hypothesis_has_gap_reference(self, knowledge_graph_fixture: KnowledgeGraph) -> None:
        """Each returned hypothesis references a known gap."""
        payload = json.dumps(
            [
                {
                    "id": "H001",
                    "hypothesis_text": "X causes Y.",
                    "predicted_cause": "X",
                    "predicted_effect": "Y",
                    "predicted_mechanism": "mechanism",
                    "testability_score": 0.8,
                    "novelty_score": 0.7,
                    "suggested_experiment": "experiment",
                    "supporting_context": "context",
                }
            ]
        )
        generator = _make_generator(payload)
        hypotheses = generator.generate(
            knowledge_graph_fixture.gaps,
            knowledge_graph_fixture,
            max_hypotheses=5,
        )
        for h in hypotheses:
            assert h.gap_addressed is not None
            assert isinstance(h.gap_addressed, GapNode)

    def test_respects_max_hypotheses_limit(self) -> None:
        """Generator never returns more hypotheses than max_hypotheses."""
        from datetime import datetime, timezone

        gaps = [_make_gap(f"var_{i}") for i in range(10)]
        # Return many from LLM
        many = [
            {
                "id": f"H{i:03d}",
                "hypothesis_text": f"H{i}",
                "predicted_cause": "x",
                "predicted_effect": "y",
                "predicted_mechanism": "m",
                "testability_score": 0.8,
                "novelty_score": 0.7,
                "suggested_experiment": "exp",
                "supporting_context": "ctx",
            }
            for i in range(10)
        ]
        adapter = MagicMock()
        adapter.complete.return_value = json.dumps(many)
        generator = HypothesisGenerator(adapter=adapter)

        graph = KnowledgeGraph(
            nodes=[],
            edges=[],
            papers_analyzed=0,
            gaps=gaps,
            created_at=datetime.now(tz=timezone.utc).isoformat(),
        )
        hypotheses = generator.generate(gaps, graph, max_hypotheses=3)
        assert len(hypotheses) <= 3

    def test_empty_gaps_returns_empty_list(self, knowledge_graph_fixture: KnowledgeGraph) -> None:
        """No gaps → no hypotheses."""
        generator = _make_generator("[]")
        hypotheses = generator.generate([], knowledge_graph_fixture, max_hypotheses=5)
        assert hypotheses == []


class TestCompositeScore:
    def test_score_uses_all_components(self, sample_hypothesis: Hypothesis) -> None:
        """composite_score combines testability, novelty, and gap_score."""
        score = composite_score(sample_hypothesis)
        expected = (
            0.4 * sample_hypothesis.testability_score
            + 0.4 * sample_hypothesis.novelty_score
            + 0.2 * sample_hypothesis.gap_addressed.gap_score
        )
        assert abs(score - expected) < 1e-6

    def test_rank_hypotheses_sorted_descending(self) -> None:
        """rank_hypotheses returns highest composite score first."""
        low_gap = _make_gap("low_var", score=0.1)
        high_gap = _make_gap("high_var", score=0.9)

        low_h = Hypothesis(
            id="H_LOW",
            gap_addressed=low_gap,
            hypothesis_text="low",
            predicted_cause="x",
            predicted_effect="y",
            predicted_mechanism="m",
            testability_score=0.2,
            novelty_score=0.2,
            suggested_experiment="exp",
            supporting_context="ctx",
        )
        high_h = Hypothesis(
            id="H_HIGH",
            gap_addressed=high_gap,
            hypothesis_text="high",
            predicted_cause="x",
            predicted_effect="y",
            predicted_mechanism="m",
            testability_score=0.9,
            novelty_score=0.9,
            suggested_experiment="exp",
            supporting_context="ctx",
        )

        ranked = rank_hypotheses([low_h, high_h])
        assert ranked[0].id == "H_HIGH"
        assert ranked[1].id == "H_LOW"

    def test_rank_empty_list(self) -> None:
        assert rank_hypotheses([]) == []
