"""Causal claims extractor — orchestrates LLM calls for a full corpus.

The extractor:
1. Splits each paper into paragraphs (to stay within LLM context limits).
2. Sends each paragraph to the LLM with the claim extraction prompt.
3. Parses and validates the response into ``CausalClaim`` objects.
4. Deduplicates claims by (cause, effect) pair within the same paper
   (keeping the highest-confidence version if duplicates exist).
5. If parsing fails on the first attempt, retries with a stricter prompt.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from hypogen.config import HypoGenSettings, get_settings
from hypogen.data.schema import CausalClaim
from hypogen.exceptions import ExtractionError
from hypogen.ingestion.splitter import split_into_paragraphs
from hypogen.llm.adapter import LLMAdapter
from hypogen.llm.parsers import parse_claims
from hypogen.llm.prompts import (
    CLAIM_EXTRACTION_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
    RETRY_PROMPT,
)

logger = logging.getLogger("hypogen.extraction")

# Schema shown to the LLM in the retry prompt so it understands the target format.
_CLAIM_SCHEMA = (
    '{"claims": [{"cause": "string", "effect": "string", '
    '"confidence": 0.0, "mechanism": "string", "evidence": "string"}]}'
)


class ClaimsExtractor:
    """Extracts causal claims from research papers using an LLM.

    Parameters
    ----------
    adapter:
        LLM adapter to use for inference. Creates a default adapter from
        settings if omitted.
    settings:
        Application settings. Uses the global singleton if omitted.
    """

    def __init__(
        self,
        adapter: LLMAdapter | None = None,
        settings: HypoGenSettings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._adapter = adapter or LLMAdapter(settings=self._settings)

    # ── Public API ────────────────────────────────────────────────────────

    def extract_from_paper(
        self, paper_name: str, paper_text: str
    ) -> list[CausalClaim]:
        """Use the LLM to extract causal claims from a single paper.

        Splits the paper into paragraphs, extracts claims from each paragraph,
        then deduplicates by (cause, effect) pair — keeping the highest-
        confidence claim when duplicates exist within the same paper.

        Parameters
        ----------
        paper_name:
            Display name / filename of the paper (used in claim metadata).
        paper_text:
            Full text of the paper.

        Returns
        -------
        list[CausalClaim]
            Deduplicated causal claims from this paper.
        """
        logger.info("Extracting claims from: %s", paper_name)

        paragraphs = split_into_paragraphs(paper_text)
        if not paragraphs:
            logger.warning("No paragraphs found in %s — skipping.", paper_name)
            return []

        logger.debug("Processing %d paragraphs from %s", len(paragraphs), paper_name)

        all_claims: list[CausalClaim] = []

        for i, paragraph in enumerate(paragraphs):
            claims = self._extract_from_paragraph(paper_name, paragraph, chunk_idx=i)
            all_claims.extend(claims)

        deduplicated = self._deduplicate_within_paper(all_claims)
        logger.info(
            "Extracted %d unique claims from %s (total before dedup: %d)",
            len(deduplicated),
            paper_name,
            len(all_claims),
        )
        return deduplicated

    def extract_from_corpus(
        self, corpus: list[tuple[str, str]]
    ) -> list[CausalClaim]:
        """Extract causal claims from all papers in a corpus.

        Parameters
        ----------
        corpus:
            List of ``(paper_name, paper_text)`` tuples.

        Returns
        -------
        list[CausalClaim]
            All claims extracted from all papers (not deduplicated across papers —
            that happens in the graph builder via merging).
        """
        if not corpus:
            logger.warning("Empty corpus passed to extract_from_corpus.")
            return []

        logger.info("Starting corpus extraction: %d papers", len(corpus))

        all_claims: list[CausalClaim] = []
        for paper_name, paper_text in corpus:
            try:
                claims = self.extract_from_paper(paper_name, paper_text)
                all_claims.extend(claims)
            except ExtractionError as exc:
                logger.error(
                    "Extraction failed for %s, skipping: %s", paper_name, exc
                )

        logger.info(
            "Corpus extraction complete: %d total claims from %d papers",
            len(all_claims),
            len(corpus),
        )
        return all_claims

    # ── Internals ─────────────────────────────────────────────────────────

    def _extract_from_paragraph(
        self, paper_name: str, paragraph: str, chunk_idx: int = 0
    ) -> list[CausalClaim]:
        """Extract claims from a single paragraph with one retry on failure."""
        prompt = CLAIM_EXTRACTION_PROMPT.format(
            paper_name=paper_name,
            text=paragraph,
        )

        # First attempt
        try:
            raw = self._adapter.complete(
                prompt, system=EXTRACTION_SYSTEM_PROMPT, temperature=0.1
            )
            return parse_claims(raw, paper_name)
        except ExtractionError as first_exc:
            logger.warning(
                "First parse attempt failed for %s chunk %d: %s",
                paper_name,
                chunk_idx,
                first_exc,
            )

        # Retry with stricter formatting prompt
        logger.debug("Retrying extraction for %s chunk %d with strict prompt", paper_name, chunk_idx)
        retry_prompt = RETRY_PROMPT.format(
            schema=_CLAIM_SCHEMA,
            original_prompt=prompt,
        )
        try:
            raw = self._adapter.complete(
                retry_prompt, system=EXTRACTION_SYSTEM_PROMPT, temperature=0.0
            )
            return parse_claims(raw, paper_name)
        except ExtractionError as second_exc:
            logger.error(
                "Retry also failed for %s chunk %d: %s — skipping chunk.",
                paper_name,
                chunk_idx,
                second_exc,
            )
            return []

    @staticmethod
    def _deduplicate_within_paper(claims: list[CausalClaim]) -> list[CausalClaim]:
        """Deduplicate claims by (cause, effect) pair within a single paper.

        When the same relationship appears in multiple paragraphs of the same
        paper, keep only the highest-confidence version.
        """
        best: dict[tuple[str, str], CausalClaim] = {}
        for claim in claims:
            key = (claim.cause, claim.effect)
            if key not in best or claim.confidence > best[key].confidence:
                best[key] = claim
        return list(best.values())
