"""Hypothesis ranking utilities.

Ranks hypotheses by a composite score:
  ``0.4 × testability_score + 0.4 × novelty_score + 0.2 × gap_score``
"""

from __future__ import annotations

import logging

from hypogen.data.schema import Hypothesis

logger = logging.getLogger("hypogen.hypotheses.ranker")

_TESTABILITY_WEIGHT = 0.4
_NOVELTY_WEIGHT = 0.4
_GAP_WEIGHT = 0.2


def composite_score(hypothesis: Hypothesis) -> float:
    """Return the composite ranking score for a hypothesis.

    Score = ``0.4 × testability + 0.4 × novelty + 0.2 × gap_score``

    Parameters
    ----------
    hypothesis:
        A generated hypothesis with filled testability and novelty scores.

    Returns
    -------
    float
        Value in [0.0, 1.0].
    """
    return (
        _TESTABILITY_WEIGHT * hypothesis.testability_score
        + _NOVELTY_WEIGHT * hypothesis.novelty_score
        + _GAP_WEIGHT * hypothesis.gap_addressed.gap_score
    )


def rank_hypotheses(hypotheses: list[Hypothesis]) -> list[Hypothesis]:
    """Sort hypotheses by composite score, highest first.

    Parameters
    ----------
    hypotheses:
        Unranked list of generated hypotheses.

    Returns
    -------
    list[Hypothesis]
        Sorted descending by composite score.
    """
    ranked = sorted(hypotheses, key=composite_score, reverse=True)

    if ranked:
        logger.debug(
            "Ranked %d hypotheses. Top: '%s' (score=%.3f)",
            len(ranked),
            ranked[0].hypothesis_text[:60],
            composite_score(ranked[0]),
        )

    return ranked
