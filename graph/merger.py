"""Claim merger — aggregates CausalClaim objects into KnowledgeEdge objects.

When multiple papers describe the same causal relationship, we want to:
- Average the confidence scores (more papers = stronger signal if consistent).
- Count distinct papers (paper_count is a quality signal for gap scoring).
- Collect all mechanisms (useful context for hypothesis generation).
- Collect all evidence sentences (useful for traceability).
"""

from __future__ import annotations

import logging
from collections import defaultdict

from hypogen.data.schema import CausalClaim, KnowledgeEdge

logger = logging.getLogger("hypogen.graph.merger")


def merge_claims(claims: list[CausalClaim]) -> list[KnowledgeEdge]:
    """Merge causal claims by (cause, effect) pair into KnowledgeEdge objects.

    For claims with the same (cause, effect) pair:
    - ``avg_confidence`` = arithmetic mean of all individual confidence scores.
    - ``paper_count`` = number of DISTINCT source papers mentioning this edge.
    - ``mechanisms`` = deduplicated list of all mechanism strings.
    - ``evidence_sentences`` = all evidence sentences, deduplicated.

    Parameters
    ----------
    claims:
        List of ``CausalClaim`` objects (typically already normalised).

    Returns
    -------
    list[KnowledgeEdge]
        One ``KnowledgeEdge`` per unique (cause, effect) pair.
        Sorted by ``avg_confidence`` descending for consistent ordering.
    """
    if not claims:
        return []

    # Group by (cause, effect) pair
    grouped: dict[tuple[str, str], list[CausalClaim]] = defaultdict(list)
    for claim in claims:
        key = (claim.cause, claim.effect)
        grouped[key].append(claim)

    edges: list[KnowledgeEdge] = []

    for (cause, effect), group_claims in grouped.items():
        # Average confidence
        avg_conf = sum(c.confidence for c in group_claims) / len(group_claims)

        # Count distinct papers
        distinct_papers = {c.source_paper for c in group_claims}
        paper_count = len(distinct_papers)

        # Collect deduplicated mechanisms (skip generic placeholders)
        mechanisms: list[str] = []
        seen_mechs: set[str] = set()
        for c in group_claims:
            mech = c.mechanism.strip()
            mech_lower = mech.lower()
            if (
                mech
                and mech_lower not in seen_mechs
                and "not specified" not in mech_lower
                and "unknown" not in mech_lower
            ):
                mechanisms.append(mech)
                seen_mechs.add(mech_lower)

        # Collect deduplicated evidence sentences
        evidence_sentences: list[str] = []
        seen_evidence: set[str] = set()
        for c in group_claims:
            ev = c.evidence.strip()
            if ev and ev not in seen_evidence:
                evidence_sentences.append(ev)
                seen_evidence.add(ev)

        edge = KnowledgeEdge(
            cause=cause,
            effect=effect,
            avg_confidence=round(avg_conf, 4),
            paper_count=paper_count,
            mechanisms=mechanisms,
            evidence_sentences=evidence_sentences,
        )
        edges.append(edge)
        logger.debug(
            "Merged edge: %s → %s (avg_conf=%.2f, papers=%d)",
            cause,
            effect,
            avg_conf,
            paper_count,
        )

    # Sort by avg_confidence descending for predictable ordering
    edges.sort(key=lambda e: e.avg_confidence, reverse=True)

    logger.info("Merged %d claims into %d unique edges.", len(claims), len(edges))
    return edges
