"""Hypothesis generator — uses the LLM to propose novel testable hypotheses.

For each gap node, the generator:
1. Gathers all causal claims involving the gap variable and its neighbours.
2. Formats the known causal context as human-readable text.
3. Calls the LLM with the hypothesis generation prompt.
4. Parses the response into a validated ``Hypothesis`` object.
5. If parsing fails, retries once with a stricter prompt.

Design rules:
- Ground every hypothesis in the graph — no free-form hallucination.
- Raise ``ExtractionError`` if both LLM attempts fail.
- Log which model was used and the gap that prompted each hypothesis.
"""

from __future__ import annotations

import logging
import uuid

import networkx as nx

from hypogen.config import HypoGenSettings, get_settings
from hypogen.data.schema import CausalClaim, GapNode, Hypothesis
from hypogen.exceptions import ExtractionError
from hypogen.llm.adapter import LLMAdapter
from hypogen.llm.parsers import parse_hypothesis
from hypogen.llm.prompts import (
    HYPOTHESIS_GENERATION_PROMPT,
    HYPOTHESIS_SYSTEM_PROMPT,
    RETRY_PROMPT,
)

logger = logging.getLogger("hypogen.hypotheses")

# Schema hint shown in the retry prompt.
_HYPOTHESIS_SCHEMA = (
    '{"hypothesis_text": "We hypothesize that ...", '
    '"predicted_cause": "...", '
    '"predicted_effect": "...", '
    '"predicted_mechanism": "...", '
    '"testability_score": 0.8, '
    '"novelty_score": 0.9, '
    '"suggested_experiment": "...", '
    '"supporting_context": "..."}'
)

# Maximum number of neighbouring claims to include in context (keeps prompt size
# manageable for smaller Ollama models).
_MAX_CONTEXT_CLAIMS = 12


class HypothesisGenerator:
    """Generate novel testable hypotheses by asking the LLM to fill graph gaps.

    Parameters
    ----------
    llm_adapter:
        Pre-configured LLM adapter. Creates a default adapter from settings
        if omitted.
    settings:
        Application settings. Uses the global singleton if omitted.
    """

    def __init__(
        self,
        llm_adapter: LLMAdapter | None = None,
        settings: HypoGenSettings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._adapter = llm_adapter or LLMAdapter(settings=self._settings)

    # ── Public API ────────────────────────────────────────────────────────

    def generate(
        self,
        gap: GapNode,
        graph: nx.DiGraph,
        claims: list[CausalClaim],
    ) -> Hypothesis:
        """Generate a single hypothesis grounded in the gap and its graph context.

        Parameters
        ----------
        gap:
            The structural gap to address.
        graph:
            The full knowledge graph (used to look up neighbouring context).
        claims:
            All raw claims extracted from the corpus, used to enrich the
            context with mechanism details.

        Returns
        -------
        Hypothesis
            A validated, LLM-generated hypothesis.

        Raises
        ------
        ExtractionError
            If both the primary and retry LLM calls fail to produce valid JSON.
        """
        hypothesis_id = f"H{str(uuid.uuid4())[:8].upper()}"
        logger.info(
            "Generating hypothesis %s for gap: '%s' (%s)",
            hypothesis_id,
            gap.variable,
            gap.gap_type,
        )

        # Build context from graph + claims
        known_edges_text, n_papers = self._build_edge_context(graph, claims)
        existing_edges_text = self._build_existing_edges_text(gap, graph, claims)

        prompt = HYPOTHESIS_GENERATION_PROMPT.format(
            n_papers=n_papers,
            known_edges=known_edges_text,
            gap_variable=gap.variable,
            gap_type=gap.gap_type,
            gap_explanation=gap.gap_explanation,
            neighbors=", ".join(gap.neighboring_nodes) or "none identified",
            existing_edges=existing_edges_text or "No direct connections yet documented.",
        )

        # First attempt
        try:
            raw = self._adapter.complete(
                prompt,
                system=HYPOTHESIS_SYSTEM_PROMPT,
                temperature=0.4,  # slightly higher than extraction for creativity
                max_tokens=1024,
            )
            return parse_hypothesis(raw, gap, hypothesis_id=hypothesis_id)
        except ExtractionError as first_exc:
            logger.warning(
                "First hypothesis parse failed for gap '%s': %s — retrying.",
                gap.variable,
                first_exc,
            )

        # Retry with stricter prompt
        retry_prompt = RETRY_PROMPT.format(
            schema=_HYPOTHESIS_SCHEMA,
            original_prompt=prompt,
        )
        try:
            raw = self._adapter.complete(
                retry_prompt,
                system=HYPOTHESIS_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=1024,
            )
            return parse_hypothesis(raw, gap, hypothesis_id=hypothesis_id)
        except ExtractionError as second_exc:
            raise ExtractionError(
                f"Both hypothesis generation attempts failed for gap "
                f"'{gap.variable}' ({gap.gap_type}): {second_exc}",
                paper=gap.variable,
            ) from second_exc

    # ── Internals ─────────────────────────────────────────────────────────

    @staticmethod
    def _build_edge_context(
        graph: nx.DiGraph,
        claims: list[CausalClaim],
    ) -> tuple[str, int]:
        """Build a compact text summary of all known causal edges.

        Returns a tuple of (formatted_text, number_of_distinct_papers).
        """
        # Build a lookup from (cause, effect) → mechanism strings
        mech_lookup: dict[tuple[str, str], list[str]] = {}
        for claim in claims:
            key = (claim.cause, claim.effect)
            mech_lookup.setdefault(key, [])
            if (
                claim.mechanism
                and "not specified" not in claim.mechanism.lower()
                and claim.mechanism not in mech_lookup[key]
            ):
                mech_lookup[key].append(claim.mechanism)

        # Count distinct papers
        distinct_papers = {c.source_paper for c in claims}

        lines: list[str] = []
        for u, v, data in sorted(graph.edges(data=True), key=lambda e: -e[2].get("avg_confidence", 0)):
            conf = data.get("avg_confidence", 0.0)
            paper_count = data.get("paper_count", 1)
            mechs = mech_lookup.get((u, v), [])
            mech_str = mechs[0] if mechs else "mechanism not specified"
            lines.append(
                f"  {u} → {v} "
                f"(confidence={conf:.2f}, papers={paper_count}, "
                f"mechanism: {mech_str})"
            )
            if len(lines) >= _MAX_CONTEXT_CLAIMS:
                lines.append(f"  ... and {graph.number_of_edges() - len(lines)} more edges")
                break

        edge_text = "\n".join(lines) if lines else "  (no edges yet)"
        return edge_text, len(distinct_papers)

    @staticmethod
    def _build_existing_edges_text(
        gap: GapNode,
        graph: nx.DiGraph,
        claims: list[CausalClaim],
    ) -> str:
        """Build a description of edges directly touching the gap variable."""
        variable = gap.variable

        # Gather in/out edges with mechanisms from claims
        mech_lookup: dict[tuple[str, str], str] = {}
        for claim in claims:
            key = (claim.cause, claim.effect)
            if key not in mech_lookup and "not specified" not in claim.mechanism.lower():
                mech_lookup[key] = claim.mechanism

        lines: list[str] = []

        if graph.has_node(variable):
            for parent in graph.predecessors(variable):
                conf = graph[parent][variable].get("avg_confidence", 0.0)
                mech = mech_lookup.get((parent, variable), "mechanism not specified")
                lines.append(
                    f"  {parent} → {variable} "
                    f"(confidence={conf:.2f}, mechanism: {mech})"
                )
            for child in graph.successors(variable):
                conf = graph[variable][child].get("avg_confidence", 0.0)
                mech = mech_lookup.get((variable, child), "mechanism not specified")
                lines.append(
                    f"  {variable} → {child} "
                    f"(confidence={conf:.2f}, mechanism: {mech})"
                )

        return "\n".join(lines)
