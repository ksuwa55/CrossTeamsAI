# Phase 4: Knowledge Graph + Dashboard — Architecture

## Overview

Phase 4 turns meeting transcripts into a knowledge graph connecting people, decisions, issues/blockers, tasks, and topics: it extracts `(subject, relation, object)` triples with an LLM, resolves entity mentions onto a shared graph across meetings, and exposes graph search, network visualization, and team-dynamics analytics through a Gradio dashboard.

## System Architecture

```
                         ┌─────────────────────────┐
                         │       Entry Points       │
                         ├─────────────────────────┤
                         │  CLI (per-meeting extract)│
                         │  Knowledge Graph Explorer │
                         │  (Gradio, port 7862)     │
                         └────────────┬────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                  │
           ┌────────▼──────┐  ┌──────▼───────┐  ┌───────▼────────┐
           │Entity/Relation │  │ Knowledge     │  │  Graph Search   │
           │Extraction      │  │ Graph Builder │  │  (graph_        │
           │(extract_       │  │(build_        │  │  search.py)     │
           │ entities_      │  │ knowledge_    │  │                 │
           │ relations.py)  │  │ graph.py)     │  │                 │
           └────────┬───────┘  └──────┬───────┘  └────────┬────────┘
                    │                 │                    │
                    ▼                 ▼                    ▼
              triples JSON      MultiDiGraph          keyword +
              per meeting       (nodes/edges +         semantic search,
                    │           evidence) + plot        path finding,
                    ▼                                    ego subgraphs
           ┌────────────────────────┐
           │   Evaluation            │
           │ (entity-linking P/R +   │
           │  LLM-judge triple P/R/F1│
           │  + graph coherence)     │
           └────────────────────────┘
```

## Components

### 1. Entity/Relation Extraction (`04_knowledgegraph_dashboard/extract_entities_relations.py`)

Turns a raw transcript into candidate knowledge-graph triples.

| Responsibility | Function |
|---|---|
| Regex prefilter + candidate windowing — **reused unmodified from Phase 2** | `enrich_transcript()`, `select_candidate_windows()` (imported from `02_causal_modeling/extract_variables.py`) |
| Ask the LLM for `(subject, relation, object)` triples with typed entities | `build_kg_prompt()`, `extract_kg_triples()` |
| Parse + validate against controlled vocabularies | `parse_kg_response()` |
| Entity resolution: normalize free text, collapse person aliases onto known speaker names | `normalize_entity()`, `resolve_person_alias()`, `node_key()` |
| Topic-drift timeline (zero extra LLM calls — reuses `enrich_transcript()`'s regex `topic` tag) | `extract_topic_timeline()` |

**Key design decisions:**
- Reuses Phase 2's prefilter/windowing so only utterances near a blocker/decision/next_action flag get an LLM call — the same cost-control rationale as `02_causal_modeling`.
- Reuses `MeetingSummarizer` from Phase 1 for the LLM call and disk cache (`cache_kg/`).
- Entity types are a controlled vocabulary (`person`, `decision`, `issue`, `task`, `topic`) and relations are a controlled vocabulary (`makes_decision`, `raises_issue`, `blocks`, `causes`, `assigned_to`, `owns`, `discusses`, `resolves`, `depends_on`, `related_to`) — the LLM is constrained to these so the graph doesn't fragment into one relation type per unique phrasing, mirroring the intent (though not the mechanism) of Phase 2's `canonicalize_node()`.
- Person entity resolution: `resolve_person_alias()` matches a mention against the transcript's known `speaker` list (case-insensitive substring), so "Aiko" and "Aiko - PM" collapse to one node. This is the concrete entity-linking step for otherwise open-domain, noisy text — the methodology the phase spec calls out as the main novelty.

### 2. Knowledge Graph Builder (`04_knowledgegraph_dashboard/build_knowledge_graph.py`)

Merges triples from many meetings into one graph.

- `node_key()` (imported from the extraction module): `f"{entity_type}:{normalize(text)}"` — node identity is normalized free text scoped by entity type, not a fixed keyword vocabulary, since entities here are open-domain (people, ad-hoc decision/issue phrases) unlike Phase 2's small fixed causal-category vocabulary.
- `build_graph()`: builds a `networkx.MultiDiGraph` (not a `DiGraph` like Phase 2's causal DAG) because two entities can be connected by more than one relation type (e.g. a person both `discusses` and is `assigned_to` the same task); edge weight = number of times that relation was observed, each edge keeps its source transcript evidence (quote + timestamp + meeting).
- `graph_coherence_metrics()`: density, connected-component count, greedy-modularity community count + modularity score, and average inter-community conductance — the unsupervised "graph coherence and clustering quality" metric from the phase spec, computable without gold labels.
- `plot_graph()`: matplotlib visualization colored by entity type (same `Agg` backend pattern as Phase 2, no new plotting dependency).

### 3. Graph Search (`04_knowledgegraph_dashboard/graph_search.py`)

- `keyword_search()`: substring match over node labels + evidence quotes, zero API calls.
- `semantic_search()`: OpenAI embeddings (`text-embedding-3-small`) over node label + evidence text, disk-cached in `cache_kg_embeddings/` exactly like `MeetingSummarizer`'s response cache — the "semantic search" half of the phase spec's novelty point (combined with network visualization and traceability). Requires live API credits for cache misses; the dashboard's Explore tab surfaces the error and points back to keyword search rather than failing silently.
- `find_path()` / `format_path_evidence()`: shortest path between two entities with the real transcript quotes behind each hop, generalizing `_format_path_evidence()` from `app/intervention_dashboard.py`.
- `ego_subgraph()`: nodes within N hops of a given entity (either direction), for the dashboard's "explore around this result" view.

### 4. Knowledge Graph Explorer (`app/kg_dashboard.py`)

Gradio web UI, port **7862** (7861 is Phase 2's intervention dashboard).

- Loads `data/sample_kg_triples/*.json` once at startup and builds the graph (same load-once pattern as `app/intervention_dashboard.py`).
- **Explore tab**: keyword/semantic search box → matching entities + an ego-subgraph plot of the top match.
- **Network tab**: full graph plot colored by entity type, coherence metrics, and a person-co-occurrence "discussion network" (who spoke in the same meeting) — the team-dynamics visual analytics from the phase spec.
- **Topic Drift tab**: per-meeting topic sequence, or a stacked bar of topic mix across all meetings ordered by meeting id, to show topic drift over the project timeline — built from `extract_topic_timeline()`, no extra LLM calls.

### 5. Evaluation (`eval/evaluate_kg_extraction.py`)

- Runs `extract_kg_triples()` unmodified against the 10 hand-labeled synthetic meetings.
- **Entity-linking precision/recall**: fraction of predicted/gold entity node keys that match.
- **Relation/fact extraction P/R/F1**: LLM-judge (`gpt-4o-mini`) + one-to-one maximum bipartite matching — reuses `parse_judge_response()` and `max_bipartite_matching()` from `eval/evaluate_causal_extraction.py` unmodified (both are schema-agnostic); only the judge prompt is triple-specific.
- **Graph coherence**: `graph_coherence_metrics()` on the graph built from all predicted triples pooled across meetings — no gold labels needed.

## Data Flow

```
Synthetic meeting transcripts (data/synthetic_transcripts/*.json)
       │
       ▼  extract_entities_relations.py → enrich_transcript() [Phase 2] + extract_kg_triples()
Triples JSON per meeting (output/kg_triples/*.kg_triples.json)
       │
       ▼  build_knowledge_graph.py → load_triples() + build_graph()
Entity-resolved knowledge graph (output/kg_graph.json, output/kg_graph.png)
       │
       ├──▼  graph_search.py — keyword/semantic search, path finding, ego subgraphs
       │
       ├──▼  app/kg_dashboard.py — Explore / Network / Topic Drift UI
       │
       └──▼  eval/evaluate_kg_extraction.py — entity-linking P/R, LLM-judge triple P/R/F1,
                graph coherence metrics
```

## Infrastructure

| Component | Technology |
|---|---|
| LLM (extraction + eval judge) | OpenAI API (`gpt-3.5-turbo` extraction, `gpt-4o-mini` judge) |
| Embeddings (semantic search) | OpenAI API (`text-embedding-3-small`) |
| Graph | `networkx` (`MultiDiGraph`) |
| Community detection | `networkx.algorithms.community` (greedy modularity) |
| UI | Gradio |
| Plotting | `matplotlib` |
| Configuration | CLI args + `.env` for API keys |

No new Python dependencies beyond what Phases 1–3 already require (see `requirements.txt`).

## Directory Structure

```
04_knowledgegraph_dashboard/
├── extract_entities_relations.py  # Prefilter reuse + LLM triple extraction + entity resolution
├── build_knowledge_graph.py       # Graph build, coherence metrics, plot
├── graph_search.py                # Keyword/semantic search, path finding, ego subgraphs
app/
├── kg_dashboard.py                # Gradio Explore/Network/Topic-Drift dashboard
eval/
├── evaluate_kg_extraction.py      # Entity-linking + triple P/R/F1 + graph coherence
├── results/kg_extraction/         # Markdown report + raw JSON
data/
├── synthetic_transcripts/         # Same 10 meetings as Phase 2, + new *.kg_labels.json
├── sample_kg_triples/             # Hand-authored seed triples that power the dashboard
output/
├── kg_triples/                    # Per-meeting predicted triples
├── kg_graph.json/.png
cache_kg/                          # LLM cache for triple extraction
cache_kg_embeddings/               # Embedding cache for semantic search
cache_kg_eval/                     # LLM cache for judge calls
```
