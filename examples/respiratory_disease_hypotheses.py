"""Example: Generate novel hypotheses from respiratory disease literature.

Demonstrates the full HypoGen pipeline:
1. Extract causal claims from multiple research papers
2. Build and inspect the causal knowledge graph
3. Detect structural gaps in the knowledge
4. Generate novel, testable hypotheses

Run:
    python examples/respiratory_disease_hypotheses.py
"""

from __future__ import annotations

import logging
from pathlib import Path

from hypogen.config import Settings, configure_logging
from hypogen.graph.builder import GraphBuilder
from hypogen.hypotheses.generator import HypothesisGenerator
from hypogen.hypotheses.ranker import composite_score, rank_hypotheses
from hypogen.knowledge.extractor import CausalClaimsExtractor
from hypogen.knowledge.gap_detector import GapDetector
from hypogen.llm.adapter import LLMAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
configure_logging()

PAPERS_DIR = Path(__file__).parent.parent / "tests" / "fixtures" / "papers"


def main() -> None:
    """Run the full HypoGen pipeline on respiratory disease literature."""
    settings = Settings()

    # ── Load papers ────────────────────────────────────────────────────────────
    papers = list(PAPERS_DIR.glob("*.txt"))
    print(f"Loading {len(papers)} papers from {PAPERS_DIR.name}/")

    # ── Extract causal claims ─────────────────────────────────────────────────
    adapter = LLMAdapter(settings=settings)
    extractor = CausalClaimsExtractor(adapter=adapter, settings=settings)

    all_claims = []
    for paper in papers:
        text = paper.read_text(encoding="utf-8")
        claims = extractor.extract(text, source_paper=paper.name)
        print(f"  {paper.name} → {len(claims)} causal claims")
        all_claims.extend(claims)

    print(f"\nTotal claims extracted: {len(all_claims)}")

    # ── Build knowledge graph ─────────────────────────────────────────────────
    builder = GraphBuilder(settings=settings)
    graph = builder.build(all_claims)

    print(f"\nKnowledge Graph:")
    print(f"  Nodes: {len(graph.nodes)}")
    print(f"  Edges: {len(graph.edges)}")
    print(f"  Papers: {graph.papers_analyzed}")

    if graph.edges:
        print("\n  Sample edges:")
        for edge in graph.edges[:5]:
            print(f"    {edge.cause} → {edge.effect} (conf={edge.avg_confidence:.2f}, n={edge.paper_count})")

    # ── Detect knowledge gaps ─────────────────────────────────────────────────
    detector = GapDetector()
    gaps = detector.detect(graph)
    graph.gaps = gaps

    print(f"\nKnowledge Gaps Detected: {len(gaps)}")
    print("\n  Top gaps by importance:")
    for gap in gaps[:5]:
        print(f"  [{gap.gap_type}] {gap.variable} (score={gap.gap_score:.2f})")
        print(f"    → {gap.gap_explanation}")

    if not gaps:
        print("  No significant gaps detected. Try adding more papers.")
        return

    # ── Generate hypotheses ───────────────────────────────────────────────────
    generator = HypothesisGenerator(adapter=adapter, settings=settings)
    hypotheses = generator.generate(gaps, graph, max_hypotheses=settings.max_hypotheses)
    graph.hypotheses = hypotheses

    ranked = rank_hypotheses(hypotheses)

    print(f"\nHypotheses Generated: {len(ranked)}")
    print("\n" + "═" * 70)
    print("NOVEL TESTABLE HYPOTHESES")
    print("═" * 70)

    for i, h in enumerate(ranked, 1):
        score = composite_score(h)
        print(f"\n{i}. {h.id}  (score={score:.2f})")
        print(f"   Hypothesis: {h.hypothesis_text}")
        print(f"   Cause → Effect: {h.predicted_cause} → {h.predicted_effect}")
        print(f"   Mechanism: {h.predicted_mechanism}")
        print(f"   Testability: {h.testability_score:.2f}  |  Novelty: {h.novelty_score:.2f}")
        print(f"   Experiment: {h.suggested_experiment}")
        print(f"   Addressing gap: [{h.gap_addressed.gap_type}] {h.gap_addressed.variable}")

    if ranked:
        top = ranked[0]
        print(f"\n{'═' * 70}")
        print(f"TOP RECOMMENDATION: {top.hypothesis_text}")
        print(f"Priority experiment: {top.suggested_experiment}")

    # ── Save graph ────────────────────────────────────────────────────────────
    output = Path("respiratory_knowledge_graph.json")
    output.write_text(graph.model_dump_json(indent=2))
    print(f"\nKnowledge graph (with hypotheses) saved → {output}")


if __name__ == "__main__":
    main()
