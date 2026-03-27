"""REST API routes for HypoGen."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from hypogen.config import get_settings
from hypogen.data.schema import KnowledgeGraph
from hypogen.graph.builder import GraphBuilder
from hypogen.hypotheses.generator import HypothesisGenerator
from hypogen.knowledge.extractor import CausalClaimsExtractor
from hypogen.knowledge.gap_detector import GapDetector
from hypogen.llm.adapter import LLMAdapter

logger = logging.getLogger("hypogen.api")

router = APIRouter()


class AnalyzeRequest(BaseModel):
    """Request body for /analyze."""

    texts: list[str]
    paper_names: Optional[list[str]] = None


class HypothesizeRequest(BaseModel):
    """Request body for /hypothesize."""

    graph: KnowledgeGraph
    max_hypotheses: Optional[int] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    provider: str
    model: str


@router.post("/analyze", response_model=KnowledgeGraph, summary="Extract causal knowledge graph from texts")
def analyze(request: AnalyzeRequest) -> KnowledgeGraph:
    """Extract causal claims and build a knowledge graph from a list of texts.

    Each element of ``texts`` is treated as one paper. Optionally provide
    matching ``paper_names`` for attribution.
    """
    settings = get_settings()
    adapter = LLMAdapter(settings=settings)
    extractor = CausalClaimsExtractor(adapter=adapter, settings=settings)
    builder = GraphBuilder(settings=settings)
    detector = GapDetector()

    names = request.paper_names or [f"paper_{i + 1}" for i in range(len(request.texts))]
    if len(names) != len(request.texts):
        raise HTTPException(
            status_code=422,
            detail="paper_names length must match texts length",
        )

    all_claims = []
    for text, name in zip(request.texts, names):
        try:
            claims = extractor.extract(text, source_paper=name)
            all_claims.extend(claims)
        except Exception as exc:
            logger.warning("Failed to extract from '%s': %s", name, exc)

    graph = builder.build(all_claims)
    graph.gaps = detector.detect(graph)
    return graph


@router.post("/hypothesize", response_model=KnowledgeGraph, summary="Generate hypotheses for detected gaps")
def hypothesize(request: HypothesizeRequest) -> KnowledgeGraph:
    """Generate novel hypotheses for the gaps in an existing knowledge graph."""
    settings = get_settings()
    max_h = request.max_hypotheses or settings.max_hypotheses

    if not request.graph.gaps:
        raise HTTPException(status_code=422, detail="Knowledge graph has no gaps to address")

    adapter = LLMAdapter(settings=settings)
    generator = HypothesisGenerator(adapter=adapter, settings=settings)

    graph = request.graph
    graph.hypotheses = generator.generate(graph.gaps, graph, max_hypotheses=max_h)
    return graph


@router.get("/health", response_model=HealthResponse, summary="Health check")
def health() -> HealthResponse:
    """Return service health and current LLM configuration."""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        provider=settings.llm_provider.value,
        model=settings.litellm_model,
    )
