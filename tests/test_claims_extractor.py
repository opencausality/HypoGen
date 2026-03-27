"""Tests for the CausalClaimsExtractor."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from hypogen.data.schema import CausalClaim
from hypogen.knowledge.extractor import CausalClaimsExtractor


def _make_extractor(response_json: str) -> CausalClaimsExtractor:
    """Return an extractor backed by a mock adapter returning ``response_json``."""
    adapter = MagicMock()
    adapter.complete.return_value = response_json
    return CausalClaimsExtractor(adapter=adapter)


class TestCausalClaimsExtractor:
    def test_returns_list_of_causal_claims(self) -> None:
        """Extractor parses valid JSON into CausalClaim instances."""
        payload = json.dumps(
            [
                {
                    "cause": "smoking",
                    "effect": "lung cancer",
                    "confidence": 0.95,
                    "mechanism": "Carcinogen-induced DNA damage",
                    "source_paper": "test.txt",
                    "evidence": "Smokers show 10x higher lung cancer incidence.",
                }
            ]
        )
        extractor = _make_extractor(payload)
        claims = extractor.extract("Smoking causes lung cancer.", source_paper="test.txt")

        assert len(claims) == 1
        assert isinstance(claims[0], CausalClaim)
        assert claims[0].cause == "smoking"
        assert claims[0].effect == "lung cancer"

    def test_source_paper_propagated(self) -> None:
        """source_paper argument is passed through to claims."""
        payload = json.dumps(
            [
                {
                    "cause": "exercise",
                    "effect": "inflammation",
                    "confidence": 0.7,
                    "mechanism": "Anti-inflammatory cytokines",
                    "source_paper": "override_me.txt",
                    "evidence": "Exercise reduces CRP levels.",
                }
            ]
        )
        extractor = _make_extractor(payload)
        claims = extractor.extract("Exercise reduces inflammation.", source_paper="correct.txt")

        # source_paper should match the argument, not the LLM-returned value
        assert all(c.source_paper == "correct.txt" for c in claims)

    def test_empty_text_returns_empty_list(self, mock_llm_adapter: MagicMock) -> None:
        """Empty input text returns empty claims list."""
        mock_llm_adapter.complete.return_value = "[]"
        extractor = CausalClaimsExtractor(adapter=mock_llm_adapter)
        claims = extractor.extract("", source_paper="empty.txt")
        assert claims == []

    def test_below_confidence_threshold_filtered(self) -> None:
        """Claims below min_confidence are excluded."""
        from hypogen.config import HypoGenSettings

        settings = HypoGenSettings(min_confidence=0.6)
        payload = json.dumps(
            [
                {
                    "cause": "stress",
                    "effect": "headache",
                    "confidence": 0.4,
                    "mechanism": "Cortisol spike",
                    "source_paper": "test.txt",
                    "evidence": "Stressed individuals report headaches.",
                }
            ]
        )
        extractor = _make_extractor(payload)
        extractor._settings = settings
        claims = extractor.extract("Stress causes headaches.", source_paper="test.txt")
        assert claims == []

    def test_invalid_json_triggers_retry(self, mock_llm_adapter: MagicMock) -> None:
        """If the first response is invalid JSON, extractor retries once."""
        valid_payload = json.dumps(
            [
                {
                    "cause": "obesity",
                    "effect": "diabetes",
                    "confidence": 0.88,
                    "mechanism": "Insulin resistance",
                    "source_paper": "test.txt",
                    "evidence": "BMI > 30 associated with T2D.",
                }
            ]
        )
        # First call returns junk, second returns valid JSON
        mock_llm_adapter.complete.side_effect = ["not json at all }{", valid_payload]
        extractor = CausalClaimsExtractor(adapter=mock_llm_adapter)
        claims = extractor.extract("Obesity causes diabetes.", source_paper="test.txt")
        assert len(claims) >= 1
