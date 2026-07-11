# Phase 2: Causal Modeling of Project Bottlenecks — Architecture

## Overview

Phase 2 turns meeting transcripts into a causal model of project bottlenecks: it extracts cause→effect events with an LLM, canonicalizes them into a shared directed acyclic graph (DAG) across meetings, fits a structural causal model (SCM) over that graph, and lets a manager simulate "what if we mitigated X" through a Gradio dashboard.

## System Architecture

```
                         ┌─────────────────────────┐
                         │       Entry Points       │
                         ├─────────────────────────┤
                         │  CLI (per-meeting extract)│
                         │  Intervention Dashboard  │
                         │  (Gradio, for managers)  │
                         └────────────┬────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                  │
           ┌────────▼──────┐  ┌──────▼───────┐  ┌───────▼────────┐
           │Variable Extract│  │ Causal Graph  │  │  Intervention   │
           │(extract_       │  │ Builder       │  │  Simulator      │
           │ variables.py)  │  │(build_causal_ │  │(simulate_       │
           │                │  │ graph.py)     │  │ intervention.py)│
           └────────┬───────┘  └──────┬───────┘  └────────┬────────┘
                    │                 │                    │
                    ▼                 ▼                    ▼
             causal events      DAG (nodes/edges      SCM (dowhy.gcm)
             JSON per meeting   + evidence) + plot     + do-calculus
                    │
                    ▼
           ┌────────────────────────┐
           │   Evaluation            │
           │ (LLM-judge P/R/F1 vs    │
           │  hand-labeled pairs)    │
           └────────────────────────┘
```

## Components

### 1. Variable Extraction (`02_causal_modeling/extract_variables.py`)

Turns a raw transcript into candidate cause→effect events.

| Responsibility | Function |
|---|---|
| Regex prefilter: tag each utterance | `enrich_transcript()` → `speaker_role`, `topic`, `utterance_type`, `emotion`, `decision`, `blocker`, `next_action` |
| Pick LLM-worthy windows around flagged utterances | `select_candidate_windows()` |
| Ask the LLM for cause/effect/timestamp | `build_causal_prompt()`, `extract_causal_events()` |
| Parse + de-duplicate the LLM's JSON response | `parse_causal_response()` |

**Key design decisions:**
- Cheap regex prefilter (`blocker`/`decision`/`next_action` flags) narrows which windows get an LLM call, keeping cost down
- Reuses `MeetingSummarizer` from Phase 1 (`01_summarization/summarizer.py`) for the LLM call and disk cache (`cache_causal/`)
- Low temperature (0.0) for reproducible extraction; deduped by `(cause, effect)` lowercased key

### 2. Causal Graph Builder (`02_causal_modeling/build_causal_graph.py`)

Merges causal events from many meetings into one DAG.

- `canonicalize_node()`: maps free-text cause/effect phrases onto a fixed vocabulary (`decision_stagnation`, `blocked_dependency`, `ambiguous_requirement`, `scope_change`, `missing_resource`, `qa_delay`, `integration_delay`, `missed_deadline`, or `other:<phrase>`) so the DAG doesn't fragment into one node per unique LLM phrasing
- `build_graph()`: builds a `networkx.DiGraph`, edge weight = number of times that cause→effect was observed, each edge carries its source evidence (transcript excerpts)
- `build_occurrence_table()`: one row per meeting, one binary column per DAG node — the table `dowhy` needs
- Outputs: graph JSON, GML, and a rendered PNG (`plot_graph()`)
- Optional: `run_dowhy_identification()` sanity-checks a treatment/outcome pair with `dowhy.CausalModel`

### 3. Intervention Simulator (`02_causal_modeling/simulate_intervention.py`)

Fits a structural causal model over the DAG and answers "what if we mitigated this cause?"

- `build_scm()`: root nodes get an empirical (observed-frequency) distribution; every downstream node gets a logistic-regression mechanism over its parents (`dowhy.gcm.ClassifierFCM`), so a hard intervention on a parent actually propagates through do-calculus
- `simulate_intervention()`: draws baseline samples vs. samples with the cause node forced to `"0"` (`gcm.interventional_samples`), reports baseline probability, intervened probability, absolute and relative reduction in the outcome node

### 4. Intervention Dashboard (`app/intervention_dashboard.py`)

Gradio web UI (manager-facing, port 7861) built on top of the two modules above.

- Loads the DAG + SCM once at startup from `data/sample_causal_events/*.json`
- Lets a manager pick a cause and an outcome, toggle mitigation, and run the simulation
- Renders a bar chart of baseline vs. mitigated probability
- Shows the shortest causal path and the real transcript quotes behind each hop (`_format_path_evidence()`), so the DAG's claim is traceable back to source text rather than opaque node names

### 5. Evaluation (`eval/evaluate_causal_extraction.py`, `eval/inspect_false_negatives.py`)

- Runs extraction against 10 hand-labeled synthetic meetings, matches predicted vs. gold cause/effect pairs with an LLM judge + maximum bipartite matching, computes per-meeting and pooled precision/recall/F1
- `inspect_false_negatives.py` takes every missed gold pair and asks a judge to find the closest predicted pair and classify the miss (`phrasing_mismatch` / `partial_match` / `genuine_miss`)

## Data Flow

```
Synthetic meeting transcripts (data/synthetic_transcripts/*.json)
       │
       ▼  extract_variables.py → enrich_transcript() + extract_causal_events()
Causal events JSON per meeting (output/causal_events/*.causal_events.json)
       │
       ▼  build_causal_graph.py → load_events() + build_graph()
Canonicalized DAG (output/causal_graph.json/.gml, output/causal_dag.png)
       │
       ▼  simulate_intervention.py → build_occurrence_data() + build_scm()
Structural causal model (dowhy.gcm.StructuralCausalModel)
       │
       ├──▼  app/intervention_dashboard.py — manager "what-if" UI
       │
       └──▼  eval/evaluate_causal_extraction.py — LLM-judge P/R/F1 vs. labels
                eval/inspect_false_negatives.py — miss classification
```

## Infrastructure

| Component | Technology |
|---|---|
| LLM (extraction + eval judge) | OpenAI API (`gpt-3.5-turbo` extraction, `gpt-4o-mini` judge) |
| Graph | `networkx` |
| Causal inference | `dowhy` (`CausalModel` + `gcm` structural causal models) |
| UI | Gradio |
| Plotting | `matplotlib` |
| Tabular data | `pandas` |
| Configuration | CLI args + `.env` for API keys |

## Directory Structure

```
02_causal_modeling/
├── extract_variables.py      # Regex prefilter + LLM causal-event extraction
├── build_causal_graph.py     # Canonicalization, DAG build, plot, DoWhy identification
├── simulate_intervention.py  # SCM fit + do-calculus intervention simulation
app/
├── intervention_dashboard.py # Gradio "what-if" dashboard for managers
eval/
├── evaluate_causal_extraction.py  # LLM-judge precision/recall/F1 vs. hand labels
├── inspect_false_negatives.py     # Classifies each missed gold pair
├── results/causal_extraction/     # Markdown report + raw JSON
data/
├── synthetic_transcripts/    # 10 labeled synthetic meetings (transcript + .labels.json)
├── sample_causal_events/     # Pre-extracted events used to seed the dashboard
output/
├── causal_events/            # Per-meeting predicted causal events
├── causal_graph.json/.gml, causal_dag.png
cache_causal/                 # LLM cache for causal-event extraction
cache_causal_eval/            # LLM cache for judge calls
```
