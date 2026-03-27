"""LLM prompt templates for causal claim extraction and hypothesis generation.

Prompt design principles:
1. Ask the LLM to return ONLY JSON — no markdown fences, no prose.
2. Distinguish correlation from causation explicitly in wording.
3. Include few-shot examples for consistent output structure.
4. Be conservative: only extract relationships where a MECHANISM is stated.
5. For hypothesis generation: ground in the graph, require specificity.
"""

from __future__ import annotations

# ── System prompts ────────────────────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = (
    "You are HypoGen — an expert scientific literature analyst specialising in "
    "causal reasoning. Your task is to extract causal relationships from research "
    "paper text. You MUST respond with valid JSON only. No markdown code fences, "
    "no backticks, no prose, no explanation outside the JSON structure. "
    "Distinguish causation (implies a mechanism) from mere correlation (co-occurrence). "
    "Be conservative: only extract relationships where the text implies a mechanism, "
    "not just co-occurrence. Use confidence < 0.5 for ambiguous cases rather than "
    "dropping them entirely."
)

HYPOTHESIS_SYSTEM_PROMPT = (
    "You are HypoGen — a scientific hypothesis generator with expertise in causal "
    "reasoning and experimental design. You generate novel, testable hypotheses "
    "that are GROUNDED in existing causal knowledge graphs. You MUST respond with "
    "valid JSON only. No markdown fences, no backticks, no prose outside the JSON. "
    "Do not invent hypotheses that are already known relationships in the graph. "
    "Every hypothesis must propose a specific mechanism, not just 'X affects Y'."
)

# ── Claim extraction prompt ───────────────────────────────────────────────────

CLAIM_EXTRACTION_PROMPT = """\
You are a scientific literature analyst with expertise in causal reasoning.

Read this research paper excerpt and extract all CAUSAL claims — relationships \
where one thing CAUSES another (not just correlates with another).

Paper: {paper_name}
Text:
{text}

For each causal claim, provide:
- cause: the causing variable/factor (lowercase, concise noun phrase)
- effect: the caused variable/factor (lowercase, concise noun phrase)
- confidence: how confident you are this is truly CAUSAL, not just correlational (0.0-1.0)
- mechanism: brief explanation of the biological/physical/chemical mechanism stated in the text
- evidence: the exact sentence or phrase from the text supporting this claim

Confidence scoring guide:
- 0.9-1.0: Explicit mechanistic statement ("X causes Y by doing Z")
- 0.7-0.9: Strong causal language with implied mechanism ("X leads to Y")
- 0.5-0.7: Moderate causal claim ("X is associated with Y via...")
- 0.3-0.5: Weak or ambiguous ("X may contribute to Y")
- 0.0-0.3: Correlational language only (avoid extracting these)

Only extract relationships where the text implies a MECHANISM, not just co-occurrence.
If no causal claims are present, return an empty list.

### Few-shot examples

Text: "Cigarette smoke contains polycyclic aromatic hydrocarbons that form DNA adducts \
in bronchial epithelial cells, triggering mutations in tumor suppressor genes and causing \
uncontrolled cell proliferation leading to lung carcinoma."
Output:
{{"claims": [{{"cause": "cigarette smoke", "effect": "lung carcinoma", "confidence": 0.95, \
"mechanism": "PAH compounds form DNA adducts causing tumor suppressor gene mutations and \
uncontrolled cell proliferation", "evidence": "Cigarette smoke contains polycyclic aromatic \
hydrocarbons that form DNA adducts in bronchial epithelial cells, triggering mutations in \
tumor suppressor genes and causing uncontrolled cell proliferation leading to lung carcinoma."}}]}}

Text: "Both exercise frequency and cardiovascular disease rates were measured in the cohort, \
and individuals who exercised more had lower rates of heart disease."
Output:
{{"claims": [{{"cause": "exercise frequency", "effect": "cardiovascular disease", \
"confidence": 0.35, "mechanism": "mechanism not specified in this excerpt", \
"evidence": "individuals who exercised more had lower rates of heart disease"}}]}}

### Now extract from this text

Return ONLY valid JSON (no markdown fences, no backticks):
{{"claims": [{{"cause": "...", "effect": "...", "confidence": 0.8, \
"mechanism": "...", "evidence": "..."}}]}}"""

# ── Hypothesis generation prompt ──────────────────────────────────────────────

HYPOTHESIS_GENERATION_PROMPT = """\
You are a scientific hypothesis generator with expertise in causal reasoning.

Here is the current state of causal knowledge in a research domain:

Known causal relationships (from {n_papers} papers):
{known_edges}

I have identified a structural gap in this knowledge graph:
Variable: {gap_variable}
Gap type: {gap_type}
Gap explanation: {gap_explanation}
Neighboring nodes in the graph: {neighbors}

Existing connections to this node:
{existing_edges}

Generate a NOVEL, TESTABLE hypothesis to fill this gap. The hypothesis must:
1. Be grounded in the existing knowledge graph (reference specific known relationships)
2. Propose a specific biological/physical/chemical mechanism — not just "X affects Y"
3. Be testable with a realistic experiment (specify the experimental approach)
4. NOT duplicate any existing known relationship in the graph
5. Address the specific gap type identified above

Gap type guidance:
- MISSING_CAUSE: propose what causes this variable (it has effects but no known causes)
- MISSING_EFFECT: propose what this variable causes downstream (it has causes but no effects)
- ISOLATED_NODE: propose how this connects to the main cluster of the graph
- LOW_CONFIDENCE_CHAIN: propose the specific mechanism strengthening a weak link
- BROKEN_CHAIN: propose the bridge variable connecting two disconnected clusters

Return ONLY valid JSON (no markdown fences, no backticks):
{{"hypothesis_text": "We hypothesize that ...", \
"predicted_cause": "...", \
"predicted_effect": "...", \
"predicted_mechanism": "...", \
"testability_score": 0.8, \
"novelty_score": 0.9, \
"suggested_experiment": "...", \
"supporting_context": "..."}}"""

# ── Retry prompt (stricter formatting) ───────────────────────────────────────

RETRY_PROMPT = """\
Your previous response was not valid JSON. Please try again.

Return ONLY a raw JSON object — no markdown code fences, no backticks, \
no commentary. The JSON must match this exact schema:

{schema}

Original request:
{original_prompt}
"""
