"""Text splitting utilities for LLM chunking.

Long research papers exceed LLM context windows and produce noisy extractions
when processed as a single block. This module splits text into meaningful
paragraphs so each LLM call handles a focused, coherent chunk.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger("hypogen.ingestion")

# ── Constants ─────────────────────────────────────────────────────────────────

# Minimum paragraph length to be worth sending to the LLM.
# Very short fragments (headers, single citations) rarely contain causal claims.
MIN_PARAGRAPH_CHARS = 80

# Maximum paragraph length. Paragraphs above this are split further at
# sentence boundaries to keep LLM prompts manageable.
MAX_PARAGRAPH_CHARS = 2000


def split_into_paragraphs(text: str) -> list[str]:
    """Split a research paper into paragraphs suitable for LLM processing.

    Strategy:
    1. Split on double-newline boundaries (standard paragraph separator).
    2. Filter out very short fragments (headers, reference lines, figure captions).
    3. Merge consecutive short paragraphs that may have been split artificially.
    4. Split very long paragraphs at sentence boundaries.

    Parameters
    ----------
    text:
        Full paper text as a string.

    Returns
    -------
    list[str]
        List of paragraph strings, each cleaned and stripped of excess whitespace.
        Paragraphs are suitable for direct insertion into LLM prompts.
    """
    if not text or not text.strip():
        return []

    # Normalise line endings and split on paragraph boundaries.
    # Consider both double-newline and single-newline-after-sentence-end.
    normalised = text.replace("\r\n", "\n").replace("\r", "\n")

    # Split on double (or more) newlines — standard paragraph boundary.
    raw_paragraphs = re.split(r"\n{2,}", normalised)

    # Clean each paragraph: collapse internal whitespace, strip edges.
    cleaned: list[str] = []
    for para in raw_paragraphs:
        # Collapse runs of spaces/tabs but preserve single newlines within para
        para = re.sub(r"[ \t]+", " ", para).strip()
        # Replace single newlines within a paragraph with a space
        para = re.sub(r"\n", " ", para)
        if len(para) >= MIN_PARAGRAPH_CHARS:
            cleaned.append(para)

    # Split paragraphs that are too long at sentence boundaries.
    result: list[str] = []
    for para in cleaned:
        if len(para) <= MAX_PARAGRAPH_CHARS:
            result.append(para)
        else:
            result.extend(_split_at_sentences(para, MAX_PARAGRAPH_CHARS))

    logger.debug(
        "Split text (%d chars) into %d paragraphs", len(text), len(result)
    )
    return result


def _split_at_sentences(text: str, max_chars: int) -> list[str]:
    """Split a long paragraph at sentence boundaries.

    Uses a simple sentence-ending pattern (period/exclamation/question mark
    followed by whitespace and a capital letter) rather than a full NLP
    tokeniser to avoid extra dependencies.

    Parameters
    ----------
    text:
        Text to split.
    max_chars:
        Target maximum character count per chunk. May be exceeded if a
        single sentence is longer.

    Returns
    -------
    list[str]
        List of text chunks, each approximately ``max_chars`` or shorter.
    """
    # Sentence-ending pattern: punctuation followed by space + uppercase letter.
    sentence_end_pattern = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
    sentences = sentence_end_pattern.split(text)

    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= max_chars:
            current = current + " " + sentence
        else:
            if current:
                chunks.append(current.strip())
            current = sentence

    if current:
        chunks.append(current.strip())

    return [c for c in chunks if len(c) >= MIN_PARAGRAPH_CHARS]
