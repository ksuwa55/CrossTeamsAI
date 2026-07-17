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
  │   3. entity_linking_metrics_by_type() — exact node_key() match (person/topic) or embedding-cluster match vs. gold (issue/decision/task)
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
| `04_knowledgegraph_dashboard/build_knowledge_graph.py` → `cluster_by_similarity()` | Shared low-level embedding-clustering primitive (single-linkage/union-find), used by both passes below |
| `04_knowledgegraph_dashboard/build_knowledge_graph.py` → `merge_similar_entities()` | Predicted-vs-predicted: merges `issue`/`decision`/`task` mentions, run before `build_graph()` (see "Semantic Entity Merging" below) |
| `04_knowledgegraph_dashboard/build_knowledge_graph.py` → `build_graph()` | Merges triples from all meetings into one weighted `MultiDiGraph` with evidence |
| `04_knowledgegraph_dashboard/build_knowledge_graph.py` → `graph_coherence_metrics()` | Density, connected components, modularity, conductance |
| `04_knowledgegraph_dashboard/graph_search.py` | Keyword/semantic search, path finding, ego subgraphs |
| `app/kg_dashboard.py` | Gradio Explore / Network / Topic Drift UI |
| `eval/evaluate_kg_extraction.py` → `entity_linking_metrics_by_type()` / `_semantic_match_counts()` | Predicted-vs-gold: matches `issue`/`decision`/`task` mentions against gold labels via embedding clustering (see "Semantic Entity Merging" below) |
| `eval/evaluate_kg_extraction.py` | Entity-linking P/R + LLM-judge triple P/R/F1 + graph coherence vs. hand-labeled triples |

## Evaluation Results (live, with semantic merge + semantic entity-linking enabled at default threshold 0.85)

Run: `python eval/evaluate_kg_extraction.py` (extraction model `gpt-3.5-turbo`, judge model `gpt-4o-mini`, default `--similarity-threshold 0.85`). Full report: `eval/results/kg_extraction/kg_extraction_report.md` / `kg_extraction_raw.json`.

Two embedding-based passes are enabled by default (see "Semantic Entity Merging" below for what each does and its own live results): `merge_similar_entities()` merges predicted `issue`/`decision`/`task` mentions with each other before the graph is built, and `entity_linking_metrics_by_type()` separately matches those same types' predicted mentions against **gold labels** via embedding clustering (`_semantic_match_counts()`), instead of requiring exact `node_key()` string equality. The pre-merge/pre-semantic-matching baseline (pure `node_key()` string matching, `--no-semantic-merge`) was 111 predicted / 61 gold / 17 matched entities, 15.3% precision / 27.9% recall — the table below is the live result with both passes enabled.

### Entity-Linking (Top-N precision/recall)

| Meeting | Pred Entities | Gold Entities | Matched (P) | Matched (R) | Precision | Recall |
|---|---|---|---|---|---|---|
| meeting_01_signoff_blocker | 16 | 5 | 0 | 0 | 0.0% | 0.0% |
| meeting_02_ambiguous_requirement | 15 | 6 | 2 | 2 | 13.3% | 33.3% |
| meeting_03_missing_resource | 4 | 6 | 2 | 2 | 50.0% | 33.3% |
| meeting_04_api_dependency | 10 | 6 | 3 | 3 | 30.0% | 50.0% |
| meeting_05_decision_stagnation | 16 | 6 | 3 | 3 | 18.8% | 50.0% |
| meeting_06_timezone_communication | 16 | 7 | 4 | 4 | 25.0% | 57.1% |
| meeting_07_scope_change | 9 | 6 | 1 | 1 | 11.1% | 16.7% |
| meeting_08_qa_delay_deadline | 6 | 6 | 1 | 1 | 16.7% | 16.7% |
| meeting_09_security_ambiguity | 5 | 6 | 1 | 1 | 20.0% | 16.7% |
| meeting_10_resource_reprioritization | 14 | 7 | 2 | 2 | 14.3% | 28.6% |
| **Aggregate (pooled)** | 111 | 61 | 19 | 19 | 17.1% | 31.1% |

Matched (P)/(R) are equal in every row here because none of the semantic matches in this run happen to pull multiple predicted variants onto one shared gold match within the same meeting — see "Semantic Entity Merging" for a case where they'd diverge.

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

Built from all 71 predicted triples pooled across meetings — identical whether `merge_similar_entities()` runs or not, since the default threshold produced zero merges on this run (see "Semantic Entity Merging" below):

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

The pooled entity-linking numbers above hide a sharp split by entity type. This breakdown is a first-class, live-run part of the pipeline (`entity_linking_metrics_by_type()` in `eval/evaluate_kg_extraction.py`, included automatically in every report), not a one-off hand computation — matched / total gold entities of that type, across all 10 meetings, with semantic merge + semantic entity-linking enabled at the default threshold:

| Entity type | Matched / Gold | Recall | Recall, `node_key()`-only baseline |
|---|---|---|---|
| `person` | 14 / 22 | 63.6% | 63.6% (unaffected — not in the semantic-matched types) |
| `task` | 3 / 13 | 23.1% | 23.1% |
| `decision` | 2 / 10 | 20.0% | 0.0% |
| `issue` | 0 / 16 | 0.0% | 0.0% |

**Person entities link well because of a deterministic anchor; free-text entities now partially benefit from a probabilistic one.** `resolve_person_alias()` rewrites a person mention onto the transcript's exact speaker label (e.g. "Sofia" → "Sofia - PM"), so when the extractor identifies a person at all, its node key exact-matches gold's node key almost two-thirds of the time. `decision`/`issue`/`task` entities have no such deterministic anchor, but now get a probabilistic one: `decision` recall moved from 0% to 20% because two predicted decision phrasings landed close enough (embedding cosine similarity ≥ 0.85) to their gold counterpart to cluster together. `issue` and `task` didn't move at the default threshold in this run — not because the approach doesn't apply to them, but because their closest predicted-vs-gold pairs (e.g. "accounting integration slipping by at least a week" vs. "accounting integration delayed" at 0.83) sit just under 0.85; see "Semantic Entity Linking vs. Gold Labels" below for the threshold sweep showing real recall gains for both once the threshold is relaxed.

## Interpretation

- **Relation-extraction recall (30.0%) is close to, but still slightly higher than, entity-linking recall (31.1%, up from 27.9% pre-semantic-matching) even though a correct triple requires getting *two* entities right** — no longer the wide gap the pre-merge baseline showed. Triple matching uses an LLM judge that tolerates paraphrasing (`judge_compatibility_matrix()`), while entity-linking now tolerates paraphrasing too for `issue`/`decision`/`task` via embedding clustering against gold (`_semantic_match_counts()`), closing most — not all — of the gap that existed when entity-linking was pure exact-string matching. The remaining gap is `issue` and `task`'s continued shortfall at the default threshold; see "Semantic Entity Linking vs. Gold Labels" below.
- **The extractor over-generates relative to gold, same pattern as Phase 2**: 71 predicted triples vs. 40 gold (1.8x), 111 predicted entities vs. 61 gold (1.8x) — comparable to Phase 2's causal extraction predicting 2.6x as many pairs as gold. It tends to split one gold fact into several finer-grained or differently-scoped predicted triples (e.g. `meeting_01_signoff_blocker`: gold's single "legal sign-off pending → blocks → customer records module" becomes four separate predicted triples about sign-off, test-case dependency, and sprint progress, none of which land on the same phrasing gold chose).
- **`meeting_01_signoff_blocker` never extracted a `person` entity** (0 of 9 predicted triples has a person subject or object), despite the transcript naming four speakers — an extraction miss specific to that meeting/window split, not a systemic failure of `resolve_person_alias()` (which worked correctly in the other 9 meetings; see the by-type breakdown above).
- **Graph coherence numbers look striking (modularity 0.958) but are an artifact of fragmentation, not a sign of rich cluster structure**: connected components (40) ≈ communities (40) ≈ more than a third of all nodes (110) — unchanged by this pass, since `merge_similar_entities()` (the predicted-vs-predicted merge that feeds the graph) is a no-op at threshold 0.85 on this dataset, distinct from the predicted-vs-gold matching that did move the entity-linking numbers above. See "Semantic Entity Merging" for why these are two different embedding-based passes with different live results.
- **Practical implication for the dashboard**: the Explore tab's keyword/semantic search and the Network tab's coherence metrics are only as good as entity resolution — with real extraction output (vs. the hand-authored `data/sample_kg_triples/` seed data, which was written to already be exact-match-friendly), users will see many near-duplicate nodes for the same underlying issue/decision, and searches will need to rely on evidence-quote substring matches rather than clean single-node hits. `merge_similar_entities()` is meant to address this directly but is a no-op on this dataset at the default threshold (see below); the entity-linking *eval metric* improvement documented above comes from a separate pass (predicted-vs-gold matching) that has no bearing on the dashboard's graph, since gold labels don't exist in real usage.
- Consistent with Phase 2's evaluation strategy, no window-size or prompt tuning was performed in this pass — the numbers above are read as directional, not final, given the 10-meeting/40-triple scale (see `evaluation-strategy.md` Known Limitations).

## Semantic Entity Merging

There are now **two separate embedding-based passes**, both live-evaluated, that address different problems and (as the live results below show) behave differently on this dataset:

1. **`merge_similar_entities()`** (predicted-vs-predicted) — merges predicted `issue`/`decision`/`task` mentions with each other before `build_graph()`, addressing graph-fragmentation (duplicate nodes for the same real-world thing) in the dashboard's real-usage graph, where no gold labels exist.
2. **Semantic entity-linking vs. gold labels** (predicted-vs-gold), added afterward — matches those same types' predicted mentions directly against gold labels for the eval metric, addressing the entity-linking recall/precision numbers themselves.

Both reuse the same low-level primitive, `build_knowledge_graph.cluster_by_similarity(embeddings, threshold)`: single-linkage/union-find clustering by cosine similarity, returning a cluster id per input position, pure and unit-testable with no API key (`04_knowledgegraph_dashboard/test_build_knowledge_graph.py`, `eval/test_evaluate_kg_extraction.py`). `graph_search.cosine_similarity()` (promoted from a private helper) and `graph_search.embed_texts()` (same `text-embedding-3-small` model + `cache_kg_embeddings/` disk cache as `semantic_search()`) are shared by both. Neither is folded into `node_key()`, which remains pure string normalization with no I/O.

### 1. `merge_similar_entities()` — predicted-vs-predicted, feeds the graph

`merge_similar_entities(triples, entity_types=("issue", "decision", "task"), threshold=0.85, ...)`, called before `build_graph()` (CLI entry point and `eval/evaluate_kg_extraction.py`'s `main()`, pooled across all meetings). For each type: embeds every unique predicted mention text, clusters via `cluster_by_similarity()`, rewrites every mention in a cluster onto the cluster's most-frequent phrasing (`cluster_mentions()`).

**Live result at the default threshold (0.85): a null result.** Zero mention pairs merge anywhere in this dataset — graph coherence is numerically identical to the `--no-semantic-merge` baseline (110 nodes, 71 edges, modularity 0.9576). The highest cosine similarities found among the 24 unique `issue`, 16 `decision`, and 39 `task` predicted mentions were:

| Type | Closest pair (predicted vs. predicted) | Cosine similarity |
|---|---|---|
| `task` | "integration with the accounting system" ↔ "accounting integration" | 0.8468 |
| `task` | "progress in the sprint" ↔ "sprint progress" | 0.8149 |
| `task` | "writing test cases" ↔ "writing test cases for that module" | 0.8078 |
| `decision` | "setting up an overlap window or an async comment thread" ↔ "async comment thread" | 0.7057 |
| `issue` | "ambiguity" ↔ "ambiguity in the spec update" | 0.6847 |

All below 0.85 — a genuine property of `gpt-3.5-turbo`'s extraction phrasing at this scale (10 meetings, 71 triples), not a bug. A threshold sweep down to 0.65 (reusing cached embeddings) does produce graph merges (node count 110 → 102) and semantically-correct-looking merges (e.g. "notifications service"/"notifications feature" at 0.7569), but **cannot affect entity-linking recall at any threshold** — this is structural, not a tuning gap: it only clusters predicted mentions with each other, so a cluster's canonical text is always drawn from the same predicted-mention pool, which (per the `node_key()`-only baseline) never matched gold's phrasing to begin with. This is exactly what motivated pass 2.

### 2. Semantic entity linking vs. gold labels — predicted-vs-gold, feeds the eval metric

`eval/evaluate_kg_extraction.py`'s `entity_linking_metrics_by_type()` now accepts `semantic_types` (default `SEMANTIC_MERGE_TYPES` = `issue`/`decision`/`task`): for those types, instead of requiring `node_key()` string equality against gold, it embeds *predicted and gold* representative mention texts **together** (per type, per meeting — never pooled across meetings, since gold labels are only valid within their own meeting) and clusters them with the same `cluster_by_similarity()`. A predicted entity counts toward precision if its cluster contains ≥1 gold entity; a gold entity counts toward recall if its cluster contains ≥1 predicted entity (`_semantic_match_counts()`). `person`/`topic` keep exact matching — `person` already has `resolve_person_alias()`'s deterministic anchor, and fuzzy-matching people risks conflating two different people merely discussed in similar terms.

**Live result at the default threshold (0.85): a real improvement.** Pooled entity-linking went from 15.3%/27.9% (P/R, `node_key()`-only) to **17.1%/31.1%** (see "Evaluation Results" above). By type: `decision` recall 0% → 20% (2/10) — two predicted decision phrasings landed at or above 0.85 cosine similarity with their gold counterpart. `issue` and `task` didn't move at 0.85, for the same "just under the bar" reason pass 1 saw — e.g. "accounting integration slipping by at least a week" vs. gold's "accounting integration delayed" scores 0.8305, just short of 0.85.

Unlike pass 1, this **is** genuinely threshold-sensitive (reusing the same cached embeddings, no extra cost):

| Threshold | `issue` recall | `decision` recall | `task` recall | Pooled recall (all types) |
|---|---|---|---|---|
| 0.85 (default) | 0.0% (0/16) | 20.0% (2/10) | 23.1% (3/13) | 31.1% (19/61) |
| 0.80 | 12.5% (2/16) | 20.0% (2/10) | 38.5% (5/13) | 37.7% (23/61) |
| 0.75 | 12.5% (2/16) | 30.0% (3/10) | 53.8% (7/13) | 42.6% (26/61) |
| 0.70 | 18.8% (3/16) | 40.0% (4/10) | 53.8% (7/13) | 45.9% (28/61) |
| 0.65 | 25.0% (4/16) | 40.0% (4/10) | 53.8% (7/13) | 47.5% (29/61) |

Inspected matches at 0.75–0.80 look correct, not degenerate (e.g. "reprioritize and demo the dashboard improvements" ↔ gold's "demo the dashboard improvements" at 0.76). **The default was kept at 0.85** — a deliberate choice to stay conservative rather than chase the larger jump available at 0.75, given the small (10-meeting, 40-triple) eval scale makes it hard to be confident the lower-threshold matches all generalize, and the phase spec suggested 0.85 as the starting point. 0.75–0.80 is recorded here as a considered-but-not-adopted alternative, revisit if a larger corpus is evaluated.

Note pass 2's improvement is **eval-only** — it has no effect on `merge_similar_entities()` or the real dashboard's graph, since gold labels don't exist outside this evaluation. The dashboard's graph-fragmentation problem (what pass 1 targets) remains open on this dataset at the default threshold. `--no-semantic-merge` disables both passes and reproduces the exact `node_key()`-only baseline for any future comparison.
