"""Command-line interface for HypoGen."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hypogen.config import configure_logging, get_settings

app = typer.Typer(
    name="hypogen",
    help="Generate novel scientific hypotheses by mining causal gaps in research literature.",
    no_args_is_help=True,
)
console = Console()
logger = logging.getLogger("hypogen.cli")


def _load_graph(graph_path: Path):
    """Load a knowledge graph from JSON."""
    from hypogen.data.schema import KnowledgeGraph

    with graph_path.open() as f:
        return KnowledgeGraph.model_validate_json(f.read())


# ── analyze ───────────────────────────────────────────────────────────────────


@app.command()
def analyze(
    papers: list[Path] = typer.Argument(..., help="One or more research paper text files."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save knowledge graph to JSON file."),
    min_confidence: Optional[float] = typer.Option(None, "--min-confidence", help="Minimum claim confidence (0-1)."),
) -> None:
    """Extract causal claims from research papers and build a knowledge graph."""
    configure_logging()
    settings = get_settings()

    for p in papers:
        if not p.exists():
            console.print(f"[red]File not found: {p}[/red]")
            raise typer.Exit(1)

    if min_confidence is not None:
        import os
        os.environ["HYPOGEN_MIN_CONFIDENCE"] = str(min_confidence)

    from hypogen.graph.builder import GraphBuilder
    from hypogen.knowledge.extractor import CausalClaimsExtractor
    from hypogen.knowledge.gap_detector import GapDetector
    from hypogen.llm.adapter import LLMAdapter

    adapter = LLMAdapter(settings=settings)
    extractor = CausalClaimsExtractor(adapter=adapter, settings=settings)
    builder = GraphBuilder(settings=settings)
    detector = GapDetector()

    all_claims = []
    with console.status("[bold green]Extracting causal claims..."):
        for paper_path in papers:
            text = paper_path.read_text(encoding="utf-8")
            claims = extractor.extract(text, source_paper=paper_path.name)
            all_claims.extend(claims)
            console.print(f"  [dim]{paper_path.name}[/dim] → {len(claims)} claims")

    graph = builder.build(all_claims)
    graph.gaps = detector.detect(graph)

    # Summary panel
    console.print(
        Panel(
            f"[bold]Papers analyzed:[/bold] {graph.papers_analyzed}\n"
            f"[bold]Causal claims:[/bold] {len(graph.claims)}\n"
            f"[bold]Graph nodes:[/bold] {len(graph.nodes)}\n"
            f"[bold]Graph edges:[/bold] {len(graph.edges)}\n"
            f"[bold]Knowledge gaps:[/bold] {len(graph.gaps)}",
            title="[bold cyan]Knowledge Graph Summary[/bold cyan]",
        )
    )

    if graph.gaps:
        table = Table(title="Top Knowledge Gaps", show_lines=True)
        table.add_column("Variable", style="cyan")
        table.add_column("Gap Type", style="yellow")
        table.add_column("Score", justify="right")
        table.add_column("Explanation")

        for gap in sorted(graph.gaps, key=lambda g: g.gap_score, reverse=True)[:5]:
            table.add_row(
                gap.variable,
                gap.gap_type,
                f"{gap.gap_score:.2f}",
                gap.gap_explanation[:80],
            )
        console.print(table)

    if output:
        output.write_text(graph.model_dump_json(indent=2))
        console.print(f"\n[green]Knowledge graph saved → {output}[/green]")


# ── gaps ──────────────────────────────────────────────────────────────────────


@app.command()
def gaps(
    graph_file: Path = typer.Argument(..., help="Knowledge graph JSON file."),
    top: int = typer.Option(10, "--top", help="Number of top gaps to show."),
) -> None:
    """Show the top knowledge gaps in an existing knowledge graph."""
    configure_logging()

    if not graph_file.exists():
        console.print(f"[red]File not found: {graph_file}[/red]")
        raise typer.Exit(1)

    graph = _load_graph(graph_file)

    if not graph.gaps:
        console.print("[yellow]No gaps detected in this knowledge graph.[/yellow]")
        return

    table = Table(title=f"Knowledge Gaps — {graph_file.name}", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Variable", style="cyan")
    table.add_column("Gap Type", style="yellow")
    table.add_column("Score", justify="right")
    table.add_column("Neighbors")
    table.add_column("Explanation")

    for i, gap in enumerate(
        sorted(graph.gaps, key=lambda g: g.gap_score, reverse=True)[:top], 1
    ):
        table.add_row(
            str(i),
            gap.variable,
            gap.gap_type,
            f"{gap.gap_score:.2f}",
            ", ".join(gap.neighboring_nodes[:3]),
            gap.gap_explanation[:100],
        )
    console.print(table)


# ── hypothesize ───────────────────────────────────────────────────────────────


@app.command()
def hypothesize(
    graph_file: Path = typer.Argument(..., help="Knowledge graph JSON file (from 'analyze' command)."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save updated graph to JSON file."),
    top: int = typer.Option(5, "--top", help="Number of top hypotheses to display."),
    max_hypotheses: Optional[int] = typer.Option(None, "--max", help="Maximum hypotheses to generate."),
) -> None:
    """Generate novel testable hypotheses from knowledge graph gaps."""
    configure_logging()
    settings = get_settings()

    if not graph_file.exists():
        console.print(f"[red]File not found: {graph_file}[/red]")
        raise typer.Exit(1)

    graph = _load_graph(graph_file)

    if not graph.gaps:
        console.print("[yellow]No gaps in graph — run 'analyze' first.[/yellow]")
        raise typer.Exit(1)

    max_h = max_hypotheses or settings.max_hypotheses

    from hypogen.hypotheses.generator import HypothesisGenerator
    from hypogen.llm.adapter import LLMAdapter

    adapter = LLMAdapter(settings=settings)
    generator = HypothesisGenerator(adapter=adapter, settings=settings)

    with console.status("[bold green]Generating hypotheses..."):
        graph.hypotheses = generator.generate(graph.gaps, graph, max_hypotheses=max_h)

    console.print(
        Panel(
            f"[bold]Hypotheses generated:[/bold] {len(graph.hypotheses)}",
            title="[bold cyan]Hypothesis Generation Complete[/bold cyan]",
        )
    )

    if graph.hypotheses:
        table = Table(title=f"Top {min(top, len(graph.hypotheses))} Hypotheses", show_lines=True)
        table.add_column("ID", style="dim", width=6)
        table.add_column("Hypothesis", style="cyan")
        table.add_column("Testability", justify="right")
        table.add_column("Novelty", justify="right")
        table.add_column("Score", justify="right")

        from hypogen.hypotheses.ranker import composite_score

        for h in graph.hypotheses[:top]:
            table.add_row(
                h.id,
                h.hypothesis_text[:100],
                f"{h.testability_score:.2f}",
                f"{h.novelty_score:.2f}",
                f"{composite_score(h):.2f}",
            )
        console.print(table)

    if output:
        output.write_text(graph.model_dump_json(indent=2))
        console.print(f"\n[green]Updated graph saved → {output}[/green]")


# ── show ──────────────────────────────────────────────────────────────────────


@app.command()
def show(
    graph_file: Path = typer.Argument(..., help="Knowledge graph JSON file."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save visualization HTML."),
) -> None:
    """Visualize the causal knowledge graph as an interactive HTML file."""
    configure_logging()

    if not graph_file.exists():
        console.print(f"[red]File not found: {graph_file}[/red]")
        raise typer.Exit(1)

    graph = _load_graph(graph_file)

    from hypogen.graph.visualizer import visualize_knowledge_graph

    out_path = output or graph_file.with_suffix(".html")
    visualize_knowledge_graph(graph, out_path)
    console.print(f"[green]Graph visualization saved → {out_path}[/green]")


# ── providers ─────────────────────────────────────────────────────────────────


@app.command()
def providers() -> None:
    """List configured LLM providers and their connection status."""
    configure_logging()
    settings = get_settings()

    from hypogen.config import DEFAULT_MODELS, LLMProvider

    table = Table(title="LLM Providers", show_lines=True)
    table.add_column("Provider", style="cyan")
    table.add_column("Default Model")
    table.add_column("Status")

    current = settings.llm_provider

    for provider in LLMProvider:
        model = DEFAULT_MODELS[provider]
        if provider == current:
            status = "[bold green]✓ active[/bold green]"
            model_display = f"[bold]{settings.litellm_model}[/bold]"
        else:
            status = "[dim]–[/dim]"
            model_display = model

        table.add_row(provider.value, model_display, status)

    console.print(table)
    console.print(
        f"\nSet [bold]HYPOGEN_LLM_PROVIDER[/bold] to change provider.\n"
        f"Set [bold]HYPOGEN_LLM_MODEL[/bold] to override the model."
    )


# ── serve ─────────────────────────────────────────────────────────────────────


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address."),
    port: int = typer.Option(8000, "--port", "-p", help="Listen port."),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes (dev only)."),
) -> None:
    """Start the HypoGen REST API server."""
    configure_logging()

    try:
        import uvicorn
    except ImportError:
        console.print("[red]uvicorn is required for the API server: pip install uvicorn[/red]")
        raise typer.Exit(1)

    from hypogen.api.server import create_app

    console.print(
        Panel(
            f"[bold]Address:[/bold] http://{host}:{port}\n"
            f"[bold]Docs:[/bold] http://{host}:{port}/docs",
            title="[bold cyan]HypoGen API Server[/bold cyan]",
        )
    )
    uvicorn.run(create_app(), host=host, port=port, reload=reload)
