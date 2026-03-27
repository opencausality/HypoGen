<div align="center">

# HypoGen

**Causal hypothesis generation for scientific discovery.**

Find the gaps in the literature. Propose the experiments that haven't been run yet.

[![CI](https://github.com/opencausality/hypogen/actions/workflows/ci.yml/badge.svg)](https://github.com/opencausality/hypogen/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## What is HypoGen?

HypoGen reads a corpus of research papers, extracts causal claims into a cumulative knowledge
graph, detects **structural gaps** in that graph — variables with unexplained causal connections,
broken chains, isolated nodes — and generates **novel testable hypotheses** to fill those gaps.

```
Research papers (corpus/)
        │
        ▼ LLM extraction
  Causal claims per paper
        │
        ▼ aggregation
  Knowledge graph (causal DAG)
        │
        ▼ gap detection
  Structural gaps: missing causes, broken chains
        │
        ▼ hypothesis generation
  Novel hypotheses — grounded in graph, not hallucinated
        │
        ▼ ranking
  Top N hypotheses by testability + novelty
```

### Key Features

- 📖 **Corpus ingestion** — reads any collection of .txt research papers or abstracts
- 🧠 **Causal extraction** — LLM extracts causal claims with confidence scores and mechanisms
- 🕸️ **Knowledge graph** — cumulative DAG aggregated from all papers
- 🔍 **Gap detection** — finds missing causes, missing effects, isolated nodes, broken chains
- 💡 **Grounded hypotheses** — generated from graph gaps, not from LLM pattern matching
- 📊 **Ranked output** — sorted by testability × novelty
- 🏠 **Local-first** — Ollama default, no API key required

---

## Why Not Just Ask an LLM for Hypotheses?

When you ask an LLM *"what are some interesting research hypotheses in respiratory disease?"*,
it generates plausible-sounding hypotheses from its training data. The problem: it can't tell
you which ones *haven't already been tested*, it hallucates mechanisms, and it ignores the
*specific structure of what is and isn't known* in your particular literature subset.

**HypoGen generates hypotheses grounded in the structure of what you've given it.**

| | Ask an LLM | HypoGen |
|---|---|---|
| **Grounding** | Training data (opaque) | Your specific paper corpus |
| **Novelty** | Unknown — may duplicate published work | Detected from graph gaps |
| **Mechanism** | Hallucinated or paraphrased | Anchored to known graph neighbors |
| **Traceability** | "I believe…" | "This gap exists because node X has no known cause in these 12 papers" |
| **Iteration** | New chat = different hypotheses | Same corpus = reproducible gaps |
| **Specificity** | Generic domain hypotheses | Specific to your exact literature subset |
| **False positives** | Common (well-known results) | Rare (graph already encodes known relationships) |

### Concrete Example

**Input corpus**: 4 papers on respiratory and cardiovascular health.

**Known causal graph** (extracted):
```
smoking → lung_cancer
smoking → COPD
pm2.5 → airway_inflammation → reduced_lung_function
pm2.5 → oxidative_stress
exercise → reduced_blood_pressure
exercise → improved_hdl_cholesterol
obesity → insulin_resistance → type2_diabetes
obesity → chronic_inflammation → cardiovascular_disease
```

**Gap detected**: `oxidative_stress` — appears as an effect of PM2.5, but has no outgoing edges.
It's a node with known causes but no effects documented in the corpus.

**Generated hypothesis (grounded)**:
```
Hypothesis H3: Oxidative stress mediates the causal pathway from ambient air
pollution to cardiovascular disease.

Predicted mechanism: PM2.5-induced oxidative stress triggers endothelial
dysfunction, which activates inflammatory cascades similar to those observed
in obesity-related cardiovascular disease (as documented in the corpus).

Testability: 0.87 — can be tested with biomarker studies measuring ROS
alongside cardiovascular outcomes in air pollution exposure cohorts.

Novelty: 0.79 — the PM2.5 → oxidative_stress link is known, but no paper
in the corpus connects oxidative_stress to cardiovascular outcomes.

Suggested experiment: Longitudinal cohort study measuring serum oxidative
stress markers (8-OHdG, MDA) alongside PM2.5 exposure and echocardiographic
outcomes. Expected N=400, follow-up 24 months.

Supporting context: pm2.5 → oxidative_stress (0.82 confidence, 2 papers),
obesity → chronic_inflammation → cardiovascular_disease (0.88 confidence).
Proposed hypothesis bridges these known subgraphs.
```

The hypothesis is not invented — it's derived from a specific structural gap in a specific graph.

---

## Installation

```bash
pip install hypogen
# or
uv add hypogen
```

**Requirements**: Python 3.10+. Local Ollama (recommended) or any LiteLLM-supported provider.

---

## Quick Start

### 1. Analyze a corpus of papers

```bash
hypogen analyze --corpus papers/ --hypotheses 5 --output knowledge.json
```

Output:
```
HypoGen — Corpus Analysis
══════════════════════════

Corpus: papers/ (4 files)
Papers analyzed: 4

Extracting causal claims...
  paper1.txt → 8 claims extracted
  paper2.txt → 6 claims extracted
  paper3.txt → 7 claims extracted
  paper4.txt → 9 claims extracted

Building knowledge graph...
  Nodes: 22
  Edges: 30 (after deduplication and confidence filtering)
  Cross-paper agreements: 8 edges confirmed by 2+ papers

Detecting structural gaps...
  MISSING_EFFECT    oxidative_stress          (gap score: 0.84)
  MISSING_CAUSE     cardiovascular_disease    (gap score: 0.71)
  LOW_CONF_CHAIN    pm2.5 → reduced_lung_function (min confidence: 0.52)
  ISOLATED_NODE     endothelial_dysfunction   (gap score: 0.63)

Generating hypotheses for top 5 gaps...
  H1: Oxidative stress → cardiovascular disease (testability: 0.87, novelty: 0.79)
  H2: Exercise reduces chronic inflammation (testability: 0.91, novelty: 0.72)
  H3: COPD increases cardiovascular risk via shared inflammatory pathway
  H4: PM2.5 accelerates insulin resistance progression
  H5: Endothelial dysfunction links air pollution and metabolic syndrome

Knowledge graph saved → knowledge.json
```

### 2. View gaps in an existing graph

```bash
hypogen gaps --graph knowledge.json --top 5
```

### 3. Generate more hypotheses

```bash
hypogen hypothesize --graph knowledge.json --n 10
```

### 4. Visualize the knowledge graph

```bash
hypogen show --graph knowledge.json
```

Opens an interactive browser visualization with gap nodes highlighted in red.

---

## CLI Reference

```bash
# Full analysis: extract claims, build graph, find gaps, generate hypotheses
hypogen analyze --corpus papers/ [--hypotheses 5] [--show] [--output knowledge.json]
hypogen analyze --corpus papers/ --min-confidence 0.6 --hypotheses 10

# Operate on existing knowledge graph
hypogen gaps --graph knowledge.json [--top 10]
hypogen hypothesize --graph knowledge.json [--n 5]
hypogen show --graph knowledge.json

# Check LLM providers
hypogen providers

# REST API server
hypogen serve --port 8000
```

---

## Architecture

```
papers/*.txt
     │
     ▼
┌──────────────────┐
│  Corpus Loader   │  ← Load .txt files, split into paragraphs
└──────────────────┘
     │
     ▼
┌──────────────────┐
│ Claims Extractor │  ← LLM: extract (cause, effect, confidence,
│                  │    mechanism, evidence) per paper
└──────────────────┘
     │ list[CausalClaim]
     ▼
┌──────────────────┐
│  Graph Builder   │  ← Aggregate claims by (cause, effect) pair
│                  │    Average confidence, collect mechanisms
└──────────────────┘
     │ KnowledgeGraph
     ▼
┌──────────────────┐
│  Gap Detector    │  ← Find: MISSING_CAUSE, MISSING_EFFECT,
│                  │    ISOLATED_NODE, LOW_CONF_CHAIN
└──────────────────┘
     │ list[GapNode]
     ▼
┌──────────────────┐
│ Hypo Generator   │  ← LLM: generate hypothesis for each gap,
│                  │    grounded in neighboring graph context
└──────────────────┘
     │ list[Hypothesis]
     ▼
┌──────────────────┐
│    Ranker        │  ← 0.4 × testability + 0.4 × novelty + 0.2 × gap_score
└──────────────────┘
```

---

## Gap Detection Types

| Gap Type | Meaning | Example |
|----------|---------|---------|
| `MISSING_CAUSE` | Node is an effect in 2+ papers but has no documented cause | `insulin_resistance` has no parent in the graph |
| `MISSING_EFFECT` | Node is well-connected on the input side but leads nowhere | `oxidative_stress` has causes but no downstream effects |
| `ISOLATED_NODE` | Node connected by only 1 edge — poorly integrated | `endothelial_dysfunction` mentioned once |
| `LOW_CONFIDENCE_CHAIN` | A causal chain exists but all links are low confidence | `pm2.5 → (0.52) → lung_function` |
| `BROKEN_CHAIN` | Two clusters that domain knowledge suggests should connect | Inflammation subgraph disconnected from cardiovascular subgraph |

---

## The Causal Scratchpad Pattern

HypoGen implements the *causal scratchpad* pattern from **CausalEvolve (ICLR 2026 Workshop)**:
rather than asking an LLM to generate hypotheses from memory, we first build an explicit
causal graph from the literature, then use the graph's structure to *identify* where hypotheses
are needed, then use the LLM to generate mechanistic explanations for those specific gaps.

The graph acts as a scratchpad that constrains and grounds the LLM's output.

---

## Input Format

HypoGen accepts any plain text files — research paper PDFs converted to text, abstracts
copied from PubMed, or any domain-specific documents.

```
papers/
├── respiratory_disease_cohort_study.txt
├── air_pollution_cardiovascular_risk.txt
├── exercise_intervention_rct.txt
└── obesity_metabolic_outcomes.txt
```

Each file is treated as one paper. The filename is used as the `source_paper` label
in extracted claims.

---

## Configuration

```env
# LLM for extraction + hypothesis generation
HYPOGEN_LLM_PROVIDER=ollama
HYPOGEN_LLM_MODEL=ollama/llama3.1

# Cloud providers
HYPOGEN_LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Graph settings
HYPOGEN_MIN_CONFIDENCE=0.5     # minimum confidence to include edge
HYPOGEN_MIN_PAPERS_FOR_EDGE=1  # edge included if in at least N papers
HYPOGEN_MAX_HYPOTHESES=10      # max hypotheses to generate per run

HYPOGEN_LOG_LEVEL=INFO
```

---

## Data Model

```python
@dataclass
class CausalClaim:
    cause: str
    effect: str
    confidence: float
    mechanism: str
    source_paper: str
    evidence: str           # exact sentence from the paper

@dataclass
class GapNode:
    variable: str
    gap_type: str           # "MISSING_CAUSE", "MISSING_EFFECT", etc.
    gap_score: float        # 0–1 importance
    neighboring_nodes: list[str]
    gap_explanation: str

@dataclass
class Hypothesis:
    hypothesis_text: str    # "We hypothesize that X causes Y via Z"
    predicted_cause: str
    predicted_effect: str
    predicted_mechanism: str
    testability_score: float
    novelty_score: float
    suggested_experiment: str
    supporting_context: str
```

---

## Philosophy

HypoGen is built on the principle that **good hypotheses come from structure, not speculation**.

- 🏠 **Local-first**: Ollama default — your research papers never leave your machine
- 🔓 **Open source**: All gap detection and graph logic is MIT licensed
- 🚫 **No telemetry**: Zero data collection
- 🧠 **Grounded, not hallucinated**: Every hypothesis traces back to a specific graph gap

---

## Contributing

HypoGen is free for academic and research use.
If you publish results from hypotheses generated with HypoGen, consider citing the tool
and contributing your domain-specific corpora as fixtures for others.

*"The most valuable scientific question is the one nobody has thought to ask yet."*
