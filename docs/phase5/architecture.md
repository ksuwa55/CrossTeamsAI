# Phase 5: Integration + Open-Sourcing — Architecture

## Overview

Phase 5 does two things: (1) cross-links Phase 2's causal events and Phase 4's
KG triples — both already extracted from the same 10 synthetic transcripts,
but never connected to each other — and bundles all three existing dashboards
(Phase 1 summarizer, Phase 2 causal what-if, Phase 4 KG explorer) into one
app; (2) fixes packaging so the repo is reproducible/cloneable as open source
— see the "Open-Sourcing" section of the root `README.md` for the concrete
decisions: MIT license, pinned dependencies, and the regeneration commands
that reproduce `eval/results/`, `output/causal_events/`, and
`output/kg_triples/` from a fresh clone (these stay gitignored — see below).

## System Architecture

```
                         ┌──────────────────────────┐
                         │      app/dashboard.py     │
                         │  (single entry point,     │
                         │   port 7860)               │
                         ├──────────────────────────┤
                         │ Tab: Summarizer            │──▶ app/ui_runner.py (Phase 1, unmodified)
                         │ Tab: Causal What-If         │──▶ app/intervention_dashboard.py (Phase 2, unmodified)
                         │ Tab: Knowledge Graph        │──▶ app/kg_dashboard.py (Phase 4, unmodified)
                         │ Tab: Integrated View        │──▶ 05_integration/cross_link.py (new)
                         └──────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                     │
           output/causal_events/*.json          output/kg_triples/*.json
           (Phase 2, keyed by meeting_id)        (Phase 4, keyed by meeting_id)
                    │                                     │
                    └───────────────┬─────────────────────┘
                                     ▼
                       05_integration/cross_link.py
                    token-overlap + timestamp matching
                                     │
                                     ▼
                     cross-linked (causal_event, kg_triple)
                        pairs, shown in Integrated View
```

`05_integration/pipeline.py` is a separate CLI convenience for regenerating
everything (including a live Phase 1 summary) end-to-end for a transcript
directory; the dashboard's Integrated View loads `output/causal_events/` and
`output/kg_triples/` directly rather than requiring it to have been run.
Both directories are gitignored (LLM-derived, regenerable, not source) — on
a fresh clone, run `eval/evaluate_causal_extraction.py` and
`eval/evaluate_kg_extraction.py` first (see root `README.md`) to populate
them before the Integrated View has anything to show.

## Components

### 1. Cross-Linking (`05_integration/cross_link.py`)

The key enabler: `output/causal_events/*.causal_events.json` (Phase 2) and
`output/kg_triples/*.kg_triples.json` (Phase 4) were both extracted from the
same 10 files in `data/synthetic_transcripts/`, keyed by the same
`meeting_id` (the transcript filename stem). Neither extraction pipeline
knows about the other's output — this module is the one place that connects
them.

- `_token_set()` / `_jaccard()`: lowercased word-token overlap, no embeddings and no extra API calls, so cross-linking itself is fully deterministic and offline once `output/causal_events/`/`output/kg_triples/` exist locally.
- `link_causal_and_kg(causal_events, kg_triples, min_score=0.25)`: for every `(causal_event, kg_triple)` pair sharing a `meeting_id`, scores Jaccard overlap between `{cause, effect}` and `{subject_text, object_text, quote}`, with a bonus when `timestamp` matches exactly. Pairs above the threshold are kept, sorted by score.
- This is a heuristic, not a learned entity linker — see `docs/gaps_toward_academic_deliverable.md` for the residual gap (same category as Phase 4's substring-based person-alias resolution).

Verified against real data: across all 10 meetings, 69 links were found (e.g.
meeting_01: causal event `"Data-sharing agreement not signed off by Legal" ->
"Inability to push the customer records module"` links to KG triple
`"data-sharing agreement with the Berlin office" --blocks--> "push the
customer records module"`, sharing timestamp `00:00:45`).

### 2. Pipeline Orchestrator (`05_integration/pipeline.py`)

Reuses, unmodified: `enrich_transcript()` / `extract_causal_events()`
(`02_causal_modeling/extract_variables.py`), `extract_kg_triples()`
(`04_knowledgegraph_dashboard/extract_entities_relations.py`),
`MeetingSummarizer` (`01_summarization/summarizer.py`).

- `--skip-llm` (default): reuses existing `output/causal_events/`/`output/kg_triples/` per meeting, cross-links them, writes `output/integrated/<meeting_id>.integrated.json` — zero API calls.
- `--with-llm`: extracts fresh (each phase's own disk cache still avoids re-paying for already-seen prompts) and also generates a live Phase 1 general-mode summary.

### 3. Unified Dashboard (`app/dashboard.py`)

Imports the existing `demo` Gradio `Blocks` objects from `app/ui_runner.py`,
`app/intervention_dashboard.py`, `app/kg_dashboard.py` unmodified — each
already builds its Blocks object at module import time and only calls
`.launch()` inside `if __name__ == "__main__"`, so importing them is safe.
Composes all three with `gr.TabbedInterface`, plus a new **Integrated View**
tab: pick a meeting, see its cross-linked causal events + KG triples, and
optionally generate a live Phase 1 summary for that meeting (wrapped in
try/except with an offline-friendly error message, the same defensive
pattern `graph_search.semantic_search()` already uses in `kg_dashboard.py`).

Single port: **7860** (reuses the port `ui_runner.py`/`scripts/run_ui.sh`
already used, since the standalone Phase 1 dashboard is superseded by this
one for normal use).

## Directory Structure

```
05_integration/
├── cross_link.py     # Token-overlap + timestamp cross-linking between causal_events and kg_triples
├── pipeline.py        # CLI: run Phases 1/2/4 over a transcript dir, cross-link, write output/integrated/
app/
├── dashboard.py        # Unified entry point: Summarizer + Causal What-If + KG Explorer + Integrated View
scripts/
├── run_dashboard.sh    # Docker build + run for the unified dashboard (port 7860)
output/
├── causal_events/      # Per-meeting causal events (gitignored — regenerate via eval/evaluate_causal_extraction.py)
├── kg_triples/         # Per-meeting KG triples (gitignored — regenerate via eval/evaluate_kg_extraction.py)
├── integrated/         # Per-meeting integrated JSON (gitignored — regenerable via pipeline.py)
```

`output/causal_events/`, `output/kg_triples/`, and `eval/results/` all stay
gitignored — they're LLM-derived, regenerable artifacts, not source. See the
root `README.md`'s reproducibility notes for the exact commands to regenerate
them from a fresh clone.
