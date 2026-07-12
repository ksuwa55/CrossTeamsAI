# Phase 4: Pipeline Flow, Real Scenario & Evaluation Results

## Two Flows: Eval vs Real Use

The core extraction/graph/search logic is identical in both flows. Only the entry point and what happens with the output differs.

### Eval Flow (labeled synthetic meetings)

```
Synthetic transcript + .kg_labels.json (data/synthetic_transcripts/*.json)
  │
  ▼  eval/evaluate_kg_extraction.py → main()
  │   1. enrich_transcript()             — regex prefilter tags each utterance (Phase 2, reused)
  │   2. extract_kg_triples()            — LLM call per candidate window → typed (subject, relation, object) triples
  │   3. entity_linking_metrics()        — set overlap on normalized entity node keys
  │   4. judge_compatibility_matrix() + max_bipartite_matching() — LLM judge vs. gold triples
  │   5. evaluate_meeting()              — precision / recall / F1 per meeting
  │   6. build_graph() + graph_coherence_metrics() — pooled-graph structural metrics
```

### Real Scenario (Knowledge Graph Explorer)

```
Real (or sample) meeting transcripts → triples (data/sample_kg_triples/*.json)
  │
  ▼  app/kg_dashboard.py → _load_model()
  │   - load_triples() + build_graph()          — entity-resolved MultiDiGraph across all meetings
  │   - enrich_transcript() per transcript       — topic timelines + speaker co-occurrence
  │
  ▼  Team member searches or browses in the Gradio UI
  │   - keyword_search() / semantic_search() → matching entities + ego_subgraph() plot
  │   - Network tab: full graph + graph_coherence_metrics() + discussion network
  │   - Topic Drift tab: per-meeting or across-corpus topic mix
```

No gold labels exist in the real scenario — the dashboard's output goes straight to a user as a search/exploration aid, not a graded prediction.

## Key Files and Their Roles

| File | Role |
|---|---|
| `02_causal_modeling/extract_variables.py` → `enrich_transcript()` | Regex prefilter (Phase 2, reused unmodified): tags each utterance with role/topic/decision/blocker/next_action |
| `04_knowledgegraph_dashboard/extract_entities_relations.py` → `extract_kg_triples()` | LLM call per candidate window → typed `(subject, relation, object)` triples |
| `04_knowledgegraph_dashboard/extract_entities_relations.py` → `resolve_person_alias()` | Collapses person mentions onto the transcript's known speaker names |
| `04_knowledgegraph_dashboard/extract_entities_relations.py` → `node_key()` | Normalizes free-text entity mentions into a node identity |
| `04_knowledgegraph_dashboard/build_knowledge_graph.py` → `build_graph()` | Merges triples from all meetings into one weighted `MultiDiGraph` with evidence |
| `04_knowledgegraph_dashboard/build_knowledge_graph.py` → `graph_coherence_metrics()` | Density, connected components, modularity, conductance |
| `04_knowledgegraph_dashboard/graph_search.py` | Keyword/semantic search, path finding, ego subgraphs |
| `app/kg_dashboard.py` | Gradio Explore / Network / Topic Drift UI |
| `eval/evaluate_kg_extraction.py` | Entity-linking P/R + LLM-judge triple P/R/F1 + graph coherence vs. hand-labeled triples |

## Evaluation Results

Run: `python eval/evaluate_kg_extraction.py` (extraction model `gpt-3.5-turbo`, judge model `gpt-4o-mini`, unmodified pipeline). Full report: `eval/results/kg_extraction/kg_extraction_report.md` / `kg_extraction_raw.json`.

### Entity-Linking (Top-N precision/recall)

| Meeting | Pred Entities | Gold Entities | Matched | Precision | Recall |
|---|---|---|---|---|---|
| meeting_01_signoff_blocker | 16 | 5 | 0 | 0.0% | 0.0% |
| meeting_02_ambiguous_requirement | 15 | 6 | 2 | 13.3% | 33.3% |
| meeting_03_missing_resource | 4 | 6 | 1 | 25.0% | 16.7% |
| meeting_04_api_dependency | 10 | 6 | 3 | 30.0% | 50.0% |
| meeting_05_decision_stagnation | 16 | 6 | 3 | 18.8% | 50.0% |
| meeting_06_timezone_communication | 16 | 7 | 3 | 18.8% | 42.9% |
| meeting_07_scope_change | 9 | 6 | 1 | 11.1% | 16.7% |
| meeting_08_qa_delay_deadline | 6 | 6 | 1 | 16.7% | 16.7% |
| meeting_09_security_ambiguity | 5 | 6 | 1 | 20.0% | 16.7% |
| meeting_10_resource_reprioritization | 14 | 7 | 2 | 14.3% | 28.6% |
| **Aggregate (pooled)** | 111 | 61 | 17 | 15.3% | 27.9% |

### Relation / Fact Extraction (Precision / Recall / F1)

| Meeting | Predicted | Gold | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| meeting_01_signoff_blocker | 9 | 4 | 1 | 8 | 3 | 11.1% | 25.0% | 15.4% |
| meeting_02_ambiguous_requirement | 8 | 4 | 1 | 7 | 3 | 12.5% | 25.0% | 16.7% |
| meeting_03_missing_resource | 2 | 4 | 1 | 1 | 3 | 50.0% | 25.0% | 33.3% |
| meeting_04_api_dependency | 7 | 4 | 2 | 5 | 2 | 28.6% | 50.0% | 36.4% |
| meeting_05_decision_stagnation | 10 | 4 | 2 | 8 | 2 | 20.0% | 50.0% | 28.6% |
| meeting_06_timezone_communication | 12 | 4 | 1 | 11 | 3 | 8.3% | 25.0% | 12.5% |
| meeting_07_scope_change | 7 | 4 | 1 | 6 | 3 | 14.3% | 25.0% | 18.2% |
| meeting_08_qa_delay_deadline | 3 | 4 | 0 | 3 | 4 | 0.0% | 0.0% | 0.0% |
| meeting_09_security_ambiguity | 3 | 4 | 1 | 2 | 3 | 33.3% | 25.0% | 28.6% |
| meeting_10_resource_reprioritization | 10 | 4 | 2 | 8 | 2 | 20.0% | 50.0% | 28.6% |
| **Aggregate (pooled)** | 71 | 40 | 12 | 59 | 28 | 16.9% | 30.0% | 21.6% |

*Aggregate is pooled (micro-averaged): TP/FP/FN (or matched/pred/gold for entity linking) are summed across all meetings before computing precision/recall/F1.*

### Graph Coherence (unsupervised, no gold labels)

Built from all 71 predicted triples pooled across meetings:

| Metric | Value |
|---|---|
| Nodes | 110 |
| Edges | 71 |
| Density | 0.0059 |
| Connected components | 40 |
| Communities (greedy modularity) | 40 |
| Modularity | 0.958 |
| Avg. inter-community conductance | 0.0 |

## Entity-Linking Breakdown by Type

The pooled entity-linking numbers above hide a sharp split by entity type. Recomputing gold-entity recall separately per `entity_type` (matched gold entities / total gold entities of that type, across all 10 meetings):

| Entity type | Matched / Gold | Recall |
|---|---|---|
| `person` | 14 / 22 | 63.6% |
| `task` | 3 / 13 | 23.1% |
| `decision` | 0 / 10 | 0.0% |
| `issue` | 0 / 16 | 0.0% |

**Person entities link well; free-text entities essentially don't.** `resolve_person_alias()` deterministically rewrites a person mention to the transcript's exact speaker label (e.g. "Sofia" → "Sofia - PM"), so when the extractor identifies a person at all, its node key exact-matches gold's node key almost two-thirds of the time. `decision`/`issue`/`task` entities have no such anchor — `node_key()` only lowercases, strips punctuation, and drops a leading article, with no paraphrase tolerance — so a predicted issue phrase like *"data-sharing agreement with the Berlin office"* never exact-matches the gold phrase *"legal sign-off pending on data-sharing agreement"* even though a human (or the relation-level LLM judge, see below) would call them the same fact.

## Interpretation

- **Relation-extraction recall (30.0%) is higher than entity-linking recall (27.9%) even though a correct triple requires getting *two* entities right** — this is not a contradiction. Triple matching uses an LLM judge that tolerates paraphrasing and re-scoping (`judge_compatibility_matrix()`, same mechanism as Phase 2's causal-pair judge), while entity-linking is scored with strict normalized-string matching. The entity-linking numbers should be read as a lower bound on the extractor's real linking quality for `issue`/`decision`/`task` entities, the same caveat Phase 2's evaluation strategy raises for its own precision metric.
- **The extractor over-generates relative to gold, same pattern as Phase 2**: 71 predicted triples vs. 40 gold (1.8x), 111 predicted entities vs. 61 gold (1.8x) — comparable to Phase 2's causal extraction predicting 2.6x as many pairs as gold. It tends to split one gold fact into several finer-grained or differently-scoped predicted triples (e.g. `meeting_01_signoff_blocker`: gold's single "legal sign-off pending → blocks → customer records module" becomes four separate predicted triples about sign-off, test-case dependency, and sprint progress, none of which land on the same phrasing gold chose).
- **`meeting_01_signoff_blocker` never extracted a `person` entity** (0 of 9 predicted triples has a person subject or object), despite the transcript naming four speakers — an extraction miss specific to that meeting/window split, not a systemic failure of `resolve_person_alias()` (which worked correctly in the other 9 meetings; see the by-type breakdown above).
- **Graph coherence numbers look striking (modularity 0.958) but are an artifact of fragmentation, not a sign of rich cluster structure**: connected components (40) ≈ communities (40) ≈ more than a third of all nodes (110). Because free-text entities almost never merge across triples (see entity-linking breakdown), most nodes end up in tiny, mostly-disconnected islands (a handful of triples from one meeting), and greedy modularity trivially scores a near-fragmented graph as "highly modular." A graph coherence score is only a meaningful signal once entity linking is good enough to actually merge repeated mentions into shared nodes — right now it mostly reflects how *un*-merged the graph is.
- **Practical implication for the dashboard**: the Explore tab's keyword/semantic search and the Network tab's coherence metrics are only as good as entity resolution — with real extraction output (vs. the hand-authored `data/sample_kg_triples/` seed data, which was written to already be exact-match-friendly), users will see many near-duplicate nodes for the same underlying issue/decision, and searches will need to rely on evidence-quote substring matches rather than clean single-node hits. This is the main lever for improving Phase 4 beyond this iteration: either a semantic node-merging pass (e.g. cluster `issue`/`decision`/`task` nodes by embedding similarity before building the graph, rather than only normalizing person mentions) or a stricter/refined extraction prompt that reuses gold-label-style phrasing.
- Consistent with Phase 2's evaluation strategy, no window-size or prompt tuning was performed in this pass — the numbers above are read as directional, not final, given the 10-meeting/40-triple scale (see `evaluation-strategy.md` Known Limitations).
