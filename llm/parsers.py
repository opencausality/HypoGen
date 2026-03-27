"""LLM output parsers — convert raw LLM text into validated Pydantic models.

Design rules:
- Use ``json.loads()`` as the primary parser.
- Fall back to a regex-based JSON stripper to handle markdown fences.
- Validate against Pydantic schemas before returning.
- If parsing fails, retry the LLM call once with a stricter prompt.
- Log raw LLM output at DEBUG level for troubleshooting.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

from hypogen.data.schema import CausalClaim, GapNode, Hypothesis
from hypogen.exceptions import ExtractionError

logger = logging.getLogger("hypogen.llm.parsers")

# ── JSON extraction helpers ───────────────────────────────────────────────────


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences that LLMs sometimes add despite instructions.

    Handles patterns like:
    - ```json ... ```
    - ``` ... ```
    - `{...}`
    """
    # Remove triple-backtick fences with optional language specifier
    text = re.sub(r"```(?:json|JSON)?\s*", "", text)
    text = re.sub(r"```", "", text)
    # Remove single-backtick wrapping
    text = re.sub(r"^`|`$", "", text.strip())
    return text.strip()


def _extract_json_object(text: str) -> str:
    """Extract the first complete JSON object or array from text.

    Some LLMs prepend or append prose around the JSON. This function uses
    a simple brace-matching approach to find the JSON boundaries.
    """
    # First try to find a JSON object {...}
    start = text.find("{")
    if start != -1:
        depth = 0
        for i, char in enumerate(text[start:], start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

    # Fall back to looking for a JSON array [...]
    start = text.find("[")
    if start != -1:
        depth = 0
        for i, char in enumerate(text[start:], start):
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

    return text  # Return as-is and let json.loads fail with a clear error


def _parse_json_robust(raw: str) -> Any:
    """Parse JSON from LLM output using multiple fallback strategies.

    Strategy:
    1. Try ``json.loads()`` directly.
    2. Strip markdown fences, try again.
    3. Extract first JSON object/array, try again.
    4. Raise ``ValueError`` with the cleaned text for error reporting.
    """
    logger.debug("Raw LLM output (%d chars): %.500s...", len(raw), raw)

    # Attempt 1: direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Attempt 2: strip markdown fences
    cleaned = _strip_markdown_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt 3: extract JSON object
    extracted = _extract_json_object(cleaned)
    try:
        return json.loads(extracted)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Cannot parse JSON from LLM output after all fallbacks. "
            f"Cleaned text: {cleaned[:300]!r}"
        ) from exc


# ── Claim parser ─────────────────────────────────────────────────────────────


def parse_claims(output: str, paper_name: str) -> list[CausalClaim]:
    """Parse LLM output into a list of validated ``CausalClaim`` objects.

    Expected JSON structure::

        {"claims": [
            {"cause": "...", "effect": "...", "confidence": 0.8,
             "mechanism": "...", "evidence": "..."}
        ]}

    Parameters
    ----------
    output:
        Raw text response from the LLM.
    paper_name:
        Name of the paper these claims were extracted from.
        Injected into each ``CausalClaim.source_paper``.

    Returns
    -------
    list[CausalClaim]
        Validated causal claims. Empty list if the LLM found no claims.

    Raises
    ------
    ExtractionError
        If the output cannot be parsed or fails schema validation.
    """
    try:
        data = _parse_json_robust(output)
    except ValueError as exc:
        raise ExtractionError(
            f"Failed to parse claims JSON: {exc}",
            paper=paper_name,
            raw_output=output[:500],
        ) from exc

    # Handle both {"claims": [...]} and bare [...]
    if isinstance(data, dict):
        raw_claims = data.get("claims", [])
    elif isinstance(data, list):
        raw_claims = data
    else:
        raise ExtractionError(
            "Expected JSON object or array from LLM, got something else.",
            paper=paper_name,
            raw_output=output[:500],
        )

    if not raw_claims:
        logger.debug("No claims extracted from %s", paper_name)
        return []

    claims: list[CausalClaim] = []
    for i, item in enumerate(raw_claims):
        if not isinstance(item, dict):
            logger.warning(
                "Skipping non-dict claim at index %d in %s", i, paper_name
            )
            continue
        try:
            claim = CausalClaim(
                cause=str(item.get("cause", "")).strip().lower(),
                effect=str(item.get("effect", "")).strip().lower(),
                confidence=float(item.get("confidence", 0.5)),
                mechanism=str(item.get("mechanism", "mechanism not specified")).strip(),
                source_paper=paper_name,
                evidence=str(item.get("evidence", "")).strip(),
            )
            claims.append(claim)
        except Exception as exc:
            logger.warning(
                "Skipping invalid claim at index %d in %s: %s", i, paper_name, exc
            )

    logger.debug(
        "Parsed %d valid claims from %s (out of %d raw)",
        len(claims),
        paper_name,
        len(raw_claims),
    )
    return claims


# ── Hypothesis parser ─────────────────────────────────────────────────────────


def parse_hypothesis(output: str, gap: GapNode, hypothesis_id: str | None = None) -> Hypothesis:
    """Parse LLM output into a validated ``Hypothesis`` object.

    Expected JSON structure::

        {
            "hypothesis_text": "We hypothesize that ...",
            "predicted_cause": "...",
            "predicted_effect": "...",
            "predicted_mechanism": "...",
            "testability_score": 0.8,
            "novelty_score": 0.9,
            "suggested_experiment": "...",
            "supporting_context": "..."
        }

    Parameters
    ----------
    output:
        Raw text response from the LLM.
    gap:
        The ``GapNode`` this hypothesis addresses.
    hypothesis_id:
        Optional ID to assign. Defaults to a UUID-based string.

    Returns
    -------
    Hypothesis
        Validated hypothesis object.

    Raises
    ------
    ExtractionError
        If the output cannot be parsed or fails schema validation.
    """
    if hypothesis_id is None:
        hypothesis_id = f"H{str(uuid.uuid4())[:8].upper()}"

    try:
        data = _parse_json_robust(output)
    except ValueError as exc:
        raise ExtractionError(
            f"Failed to parse hypothesis JSON: {exc}",
            paper=gap.variable,
            raw_output=output[:500],
        ) from exc

    if not isinstance(data, dict):
        raise ExtractionError(
            "Expected JSON object from LLM for hypothesis, got array or scalar.",
            paper=gap.variable,
            raw_output=output[:500],
        )

    try:
        hyp = Hypothesis(
            id=hypothesis_id,
            gap_addressed=gap,
            hypothesis_text=str(data.get("hypothesis_text", "")).strip(),
            predicted_cause=str(data.get("predicted_cause", "")).strip().lower(),
            predicted_effect=str(data.get("predicted_effect", "")).strip().lower(),
            predicted_mechanism=str(data.get("predicted_mechanism", "")).strip(),
            testability_score=float(data.get("testability_score", 0.5)),
            novelty_score=float(data.get("novelty_score", 0.5)),
            suggested_experiment=str(data.get("suggested_experiment", "")).strip(),
            supporting_context=str(data.get("supporting_context", "")).strip(),
        )
    except Exception as exc:
        raise ExtractionError(
            f"Hypothesis schema validation failed: {exc}",
            paper=gap.variable,
            raw_output=output[:500],
        ) from exc

    return hyp
