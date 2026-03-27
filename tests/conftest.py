"""Shared pytest fixtures for HypoGen tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from hypogen.data.schema import (
    CausalClaim,
    GapNode,
    Hypothesis,
    KnowledgeEdge,
    KnowledgeGraph,
)

# ── Directory helpers ─────────────────────────────────────────────────────────

FIXTURES_DIR = Path(__file__).parent / "fixtures"
PAPERS_DIR = FIXTURES_DIR / "papers"


@pytest.fixture
def fixture_paper_path() -> Path:
    """Return path to the respiratory disease paper fixture."""
    return PAPERS_DIR / "respiratory_disease.txt"


@pytest.fixture
def all_paper_paths() -> list[Path]:
    """Return all four paper fixture paths."""
    return list(PAPERS_DIR.glob("*.txt"))


# ── CausalClaim fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def sample_claims() -> list[CausalClaim]:
    """Three causal claims about respiratory health."""
    return [
        CausalClaim(
            cause="air pollution",
            effect="inflammation",
            confidence=0.9,
            mechanism="Particulate matter triggers oxidative stress in lung epithelium",
            source_paper="respiratory_disease.txt",
            evidence="PM2.5 exposure significantly elevated IL-6 and TNF-α markers in exposed cohorts.",
        ),
        CausalClaim(
            cause="inflammation",
            effect="respiratory disease",
            confidence=0.85,
            mechanism="Chronic inflammation damages alveolar tissue over time",
            source_paper="respiratory_disease.txt",
            evidence="Persistent inflammatory markers correlated strongly with COPD onset.",
        ),
        CausalClaim(
            cause="air pollution",
            effect="respiratory disease",
            confidence=0.8,
            mechanism="Direct mucosal damage from toxic compounds",
            source_paper="air_pollution_cardiovascular.txt",
            evidence="Long-term air pollution exposure doubles the risk of chronic bronchitis.",
        ),
    ]


# ── KnowledgeGraph fixtures ───────────────────────────────────────────────────


@pytest.fixture
def sample_edges() -> list[KnowledgeEdge]:
    """Pre-built edges for the knowledge graph."""
    return [
        KnowledgeEdge(
            cause="air pollution",
            effect="inflammation",
            avg_confidence=0.88,
            paper_count=2,
            mechanisms=["oxidative stress pathway"],
        ),
        KnowledgeEdge(
            cause="inflammation",
            effect="respiratory disease",
            avg_confidence=0.85,
            paper_count=1,
            mechanisms=["alveolar tissue damage"],
        ),
    ]


@pytest.fixture
def knowledge_graph_fixture(sample_edges: list[KnowledgeEdge]) -> KnowledgeGraph:
    """A knowledge graph with edges, nodes, and pre-detected gaps."""
    from datetime import datetime, timezone

    gap = GapNode(
        variable="exercise",
        gap_type="MISSING_CAUSE",
        gap_score=0.75,
        neighboring_nodes=["inflammation", "respiratory disease"],
        existing_edges=["air pollution → inflammation"],
        gap_explanation="Exercise has no incoming causal edges despite being linked to inflammation.",
    )

    return KnowledgeGraph(
        nodes=["air pollution", "inflammation", "respiratory disease", "exercise"],
        edges=sample_edges,
        papers_analyzed=2,
        gaps=[gap],
        hypotheses=[],
        created_at=datetime.now(tz=timezone.utc).isoformat(),
        model_used="ollama/llama3.1",
    )


# ── GapNode fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def sample_gap() -> GapNode:
    """A single MISSING_CAUSE gap node."""
    return GapNode(
        variable="exercise",
        gap_type="MISSING_CAUSE",
        gap_score=0.8,
        neighboring_nodes=["inflammation"],
        existing_edges=["inflammation → respiratory disease"],
        gap_explanation="Exercise lacks causal predecessors despite clinical relevance.",
    )


@pytest.fixture
def sample_hypothesis(sample_gap: GapNode) -> Hypothesis:
    """A hypothesis addressing the exercise gap."""
    return Hypothesis(
        id="H001",
        gap_addressed=sample_gap,
        hypothesis_text=(
            "We hypothesize that regular aerobic exercise reduces chronic inflammation "
            "via increased anti-inflammatory cytokine production, thereby lowering the "
            "risk of respiratory disease."
        ),
        predicted_cause="aerobic exercise",
        predicted_effect="inflammation reduction",
        predicted_mechanism="IL-10 upregulation and TNF-α suppression through physical activity",
        testability_score=0.9,
        novelty_score=0.7,
        suggested_experiment=(
            "Randomized controlled trial: 12-week aerobic exercise program vs. sedentary control, "
            "measuring IL-6, TNF-α, and IL-10 at baseline and follow-up."
        ),
        supporting_context="Existing edges: air pollution → inflammation → respiratory disease",
    )


# ── LLM mock ─────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_llm_adapter() -> MagicMock:
    """A mock LLMAdapter that returns a plausible claims JSON response."""
    adapter = MagicMock()

    claims_json = """[
        {
            "cause": "air pollution",
            "effect": "inflammation",
            "confidence": 0.85,
            "mechanism": "Oxidative stress from particulate matter",
            "source_paper": "test_paper.txt",
            "evidence": "PM2.5 exposure elevated inflammatory markers."
        }
    ]"""

    adapter.complete.return_value = claims_json
    return adapter
