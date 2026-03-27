"""CLI tests for HypoGen."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from hypogen.cli import app

runner = CliRunner()


class TestProvidersCommand:
    def test_lists_all_providers(self) -> None:
        """providers command shows all supported LLM providers."""
        result = runner.invoke(app, ["providers"])
        assert result.exit_code == 0
        assert "ollama" in result.output.lower()
        assert "openai" in result.output.lower()
        assert "anthropic" in result.output.lower()


class TestAnalyzeCommand:
    def test_analyze_with_fixture_paper(self, fixture_paper_path: Path) -> None:
        """analyze command reads a paper file and produces a summary."""
        with patch("hypogen.cli.LLMAdapter") as mock_cls:
            adapter = MagicMock()
            adapter.complete.return_value = "[]"
            mock_cls.return_value = adapter

            result = runner.invoke(app, ["analyze", str(fixture_paper_path)])

        assert result.exit_code == 0
        assert "Papers analyzed" in result.output or "Knowledge Graph" in result.output

    def test_analyze_saves_output(self, fixture_paper_path: Path, tmp_path: Path) -> None:
        """--output flag saves the knowledge graph JSON."""
        output = tmp_path / "graph.json"

        with patch("hypogen.cli.LLMAdapter") as mock_cls:
            adapter = MagicMock()
            adapter.complete.return_value = "[]"
            mock_cls.return_value = adapter

            result = runner.invoke(app, ["analyze", str(fixture_paper_path), "--output", str(output)])

        assert result.exit_code == 0
        assert output.exists()

    def test_analyze_missing_file_exits_nonzero(self, tmp_path: Path) -> None:
        """analyze with a non-existent file exits with code 1."""
        missing = tmp_path / "doesnotexist.txt"
        result = runner.invoke(app, ["analyze", str(missing)])
        assert result.exit_code != 0


class TestGapsCommand:
    def test_gaps_with_valid_graph(self, tmp_path: Path, knowledge_graph_fixture) -> None:
        """gaps command displays the gap table for a valid graph JSON."""
        graph_path = tmp_path / "graph.json"
        graph_path.write_text(knowledge_graph_fixture.model_dump_json())

        result = runner.invoke(app, ["gaps", str(graph_path)])

        assert result.exit_code == 0
        # Should show at least one gap
        assert "MISSING" in result.output or "Gap" in result.output or "exercise" in result.output

    def test_gaps_missing_file_exits_nonzero(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope.json"
        result = runner.invoke(app, ["gaps", str(missing)])
        assert result.exit_code != 0


class TestHypothesizeCommand:
    def test_hypothesize_generates_hypotheses(
        self, tmp_path: Path, knowledge_graph_fixture
    ) -> None:
        """hypothesize command generates hypotheses from graph gaps."""
        import json

        graph_path = tmp_path / "graph.json"
        graph_path.write_text(knowledge_graph_fixture.model_dump_json())
        output = tmp_path / "with_hyps.json"

        hyp_payload = json.dumps(
            [
                {
                    "id": "H001",
                    "hypothesis_text": "Test hypothesis.",
                    "predicted_cause": "exercise",
                    "predicted_effect": "inflammation",
                    "predicted_mechanism": "IL-10 pathway",
                    "testability_score": 0.85,
                    "novelty_score": 0.75,
                    "suggested_experiment": "RCT with 12-week exercise intervention.",
                    "supporting_context": "air pollution → inflammation",
                }
            ]
        )

        with patch("hypogen.cli.LLMAdapter") as mock_cls:
            adapter = MagicMock()
            adapter.complete.return_value = hyp_payload
            mock_cls.return_value = adapter

            result = runner.invoke(
                app,
                ["hypothesize", str(graph_path), "--output", str(output)],
            )

        assert result.exit_code == 0
        assert output.exists()
