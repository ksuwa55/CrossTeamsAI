# Phase 4: Knowledge Graph + Dashboard — Evaluation Strategy

## Objective

Measure how well `04_knowledgegraph_dashboard/extract_entities_relations.py` recovers the true entities and relations in a meeting transcript, and characterize the structural quality of the resulting merged graph.

## Benchmark Dataset

**Synthetic labeled meetings** (`data/synthetic_transcripts/`) — the same 10 transcripts Phase 2 uses, with new gold labels:

- `meeting_XX_*.kg_labels.json`: hand-written gold `(subject, relation, object)` triples, one annotator (same annotator/caveats as Phase 2's `.labels.json` — see `docs/gaps_toward_academic_deliverable.md`).

### Dataset Statistics

| Category | Count |
|---|---|
| Meetings | 10 |
| Gold triples | 40 (4 per meeting) |
| Gold entity types covered | person, decision, issue, task |

Reused rather than re-collected so Phase 4's extraction quality can be read against the same source material as Phase 2's causal extraction (`docs/phase2/evaluation-strategy.md`).

## Metrics

### 1. Entity-Linking Precision/Recall

`entity_linking_metrics_by_type()` compares predicted entities against gold entities for a meeting, per `entity_type`, then `entity_linking_metrics()` pools across types:

- **`person`/`topic`**: exact match on `node_key(entity_type, text)` — `person` is anchored by `resolve_person_alias()` rewriting mentions onto the transcript's speaker label before `node_key()` sees them.
- **`issue`/`decision`/`task`** (`SEMANTIC_MERGE_TYPES`): predicted and gold mention texts are embedded together (`text-embedding-3-small`, `cache_kg_embeddings/`) and clustered by cosine similarity (`build_knowledge_graph.cluster_by_similarity()`, threshold `DEFAULT_SIMILARITY_THRESHOLD` = 0.75 by default, `--similarity-threshold`); a predicted (resp. gold) entity counts as matched if its cluster contains ≥1 gold (resp. predicted) entity (`_semantic_match_counts()`). This tolerates paraphrasing (e.g. predicted "accounting integration slipping by at least a week" vs. gold "accounting integration delayed") that plain `node_key()` string equality would miss.
- **Precision**: of the entities the model surfaced, how many correspond to a real gold entity (matched-for-precision / predicted).
- **Recall**: of the gold entities, how many the model surfaced (matched-for-recall / gold). These can differ per type, since one gold mention's cluster can contain several matching predicted variants.

This is the "Top-N precision for entity linking" metric from the phase spec, framed as set precision/recall rather than a ranked Top-N since the extractor doesn't produce a ranked entity list. `--no-semantic-merge` disables the embedding-cluster matching for `issue`/`decision`/`task`, falling back to exact `node_key()` matching for all types.

### 2. Relation / Fact Extraction: Precision / Recall / F1

Same matching philosophy as Phase 2's causal-event evaluation, generalized from pairs to triples:

- Gold and predicted phrases are short, paraphrased descriptions — embedding similarity conflates "same fact, different wording" with "related but distinct fact in the same topic," so an **LLM judge** (`gpt-4o-mini`, model-distinct from the `gpt-3.5-turbo` extractor) reads each predicted triple against all gold triples for that meeting and returns a compatibility vector, explicitly checking subject/object direction and semantic equivalence of the relation.
- Matching is one-to-one via **maximum-cardinality bipartite matching** (reuses `max_bipartite_matching()` from `eval/evaluate_causal_extraction.py` unmodified — the function is schema-agnostic, operating on a boolean compatibility matrix).
- `parse_judge_response()` is also reused unmodified from the causal evaluation for the same reason.
- Reported per-meeting and pooled (micro-averaged).

### 3. Graph Coherence and Clustering Quality (unsupervised)

`build_knowledge_graph.graph_coherence_metrics()`, computed on the graph built from all predicted triples pooled across meetings — no gold labels needed:

| Metric | What it measures |
|---|---|
| Density | How connected the graph is relative to a complete graph |
| Connected components | How fragmented the graph is |
| Modularity + community count | Whether the graph has meaningful cluster structure (greedy modularity communities) |
| Avg. inter-community conductance | How well-separated those clusters are |

## Evaluation Pipeline

```bash
python eval/evaluate_kg_extraction.py \
  --transcripts-dir data/synthetic_transcripts \
  --extract-model gpt-3.5-turbo \
  --judge-model gpt-4o-mini
```

Runs `enrich_transcript()` + `extract_kg_triples()` (unmodified pipeline) per meeting, judges predicted-vs-gold triples, computes entity-linking and relation-extraction metrics, builds the pooled graph and its coherence metrics, and writes:

- `output/kg_triples/<meeting_id>.kg_triples.json` — raw predictions
- `eval/results/kg_extraction/kg_extraction_report.md` — per-meeting + aggregate tables for both metrics, plus graph coherence, false positives, false negatives
- `eval/results/kg_extraction/kg_extraction_raw.json` — machine-readable matched/unmatched triples + all metrics

## Tunable Parameters

| Parameter | Default | Impact |
|---|---|---|
| `--extract-model` | `gpt-3.5-turbo` | Extraction quality vs. cost |
| `--judge-model` | `gpt-4o-mini` | Matching judge; needs reliable structured-JSON output |
| `--window-before` / `--window-after` | 2 / 1 | Context window size (inherited default from Phase 2) |
| `--similarity-threshold` | 0.75 (`build_knowledge_graph.DEFAULT_SIMILARITY_THRESHOLD`, shared by the CLI, the dashboard, and this eval script) | Cosine similarity cutoff shared by both `merge_similar_entities()` (predicted-vs-predicted, feeds the graph) and the semantic entity-linking match (predicted-vs-gold, feeds the eval metric) for `issue`/`decision`/`task` — higher merges/matches more conservatively, lower more aggressively. Lowered from an initial 0.85 after live results; see `pipeline-flow-and-results.md` "Semantic Entity Merging" for the full threshold sweep and why 0.75 was chosen |
| `--no-semantic-merge` | off (both passes enabled) | Skips `merge_similar_entities()` *and* semantic entity-linking; entity-linking and graph coherence fall back entirely to raw `node_key()` string matching |

## Experiment Tracking

- Predictions: `output/kg_triples/*.kg_triples.json`
- Reports: `eval/results/kg_extraction/kg_extraction_report.md`, `kg_extraction_raw.json`
- Extraction cache: `cache_kg/`; judge cache: `cache_kg_eval/`; embedding cache: `cache_kg_embeddings/` — shared across `graph_search.semantic_search()`, `build_knowledge_graph.merge_similar_entities()`, and `evaluate_kg_extraction._semantic_match_counts()`, all keyed by `md5(model + text)` so any mention text in common is only ever embedded once

## Known Limitations

1. **Small, synthetic dataset**: 10 meetings, 40 gold triples — same scale caveat as Phase 2, directional not precise.
2. **Judge is itself an LLM**: not a ground-truth oracle, though model-distinct from the extractor.
3. **Entity resolution is heuristic, not learned**: `resolve_person_alias()` only handles substring-level aliasing against known speakers; it will not resolve pronouns, nicknames unrelated to the speaker label, or cross-meeting aliasing of the same person under different display names.
4. **Entity-linking paraphrase tolerance for free-text types is now implemented as two separate passes, live-evaluated at two thresholds, with different results.** The first live run (`node_key()`-only baseline) confirmed the underlying gap: `person` entities link well (63.6% gold recall) via `resolve_person_alias()`'s deterministic anchor, but `issue`/`decision`/`task` entities essentially never exact-matched gold phrasing (0–23.1% recall). Two fixes were built, both reusing `build_knowledge_graph.cluster_by_similarity()` (cosine similarity, default threshold `DEFAULT_SIMILARITY_THRESHOLD`), both unit-tested with mocked embeddings (`04_knowledgegraph_dashboard/test_build_knowledge_graph.py`, `eval/test_evaluate_kg_extraction.py`), both re-run live end-to-end at threshold 0.85 and then again at 0.75 (the threshold was lowered after the first live run — see `pipeline-flow-and-results.md` "Semantic Entity Merging" for the full sweep and the closest-pair evidence behind that decision):
   - **`merge_similar_entities()`** (predicted-vs-predicted, feeds `build_graph()`, also wired into `app/kg_dashboard.py`'s `_load_model()`): clusters predicted mentions with each other before `node_key()` sees them. **Live result: a null result at 0.85, modest real de-fragmentation at the current default 0.75** — 110 nodes unchanged at 0.85, 110 → 106 at 0.75 (density/modularity move correspondingly; full numbers in `pipeline-flow-and-results.md` "Graph Coherence"). This pass alone cannot raise entity-linking recall at *any* threshold, by construction — it only clusters predicted mentions with each other (gold is never embedded), so a cluster's canonical text is always drawn from the same predicted-mention pool. On the dashboard's current hand-authored seed data (`data/sample_kg_triples/`) it remains a no-op even at 0.75 (verified: 61 nodes/40 edges either way) — the data was written to already be exact-match-friendly, so this pass's real effect will only show up with live-extracted transcripts.
   - **Semantic entity linking vs. gold labels** (predicted-vs-gold, feeds `entity_linking_metrics_by_type()`, eval-only), added specifically to close the recall gap the above couldn't: embeds predicted *and* gold mention texts together per type per meeting and clusters them, so a predicted mention can match a differently-worded gold mention directly. **Live result: the primary improvement** — pooled entity-linking recall moved from 27.9% (baseline) to 31.1% at 0.85, then to **42.6% at the current default 0.75**, with all three types moving (`task` 23.1%→53.8%, `decision` 0%→30.0%, `issue` 0%→12.5%; full by-type table and the complete 0.65–0.85 sweep in `pipeline-flow-and-results.md` "Semantic Entity Merging" §2).

   Two residual caveats even after both passes: (a) neither can fix an extractor miss where no `issue`/`decision`/`task` mention was predicted at all (a recall-of-the-underlying-fact problem, upstream of linking); (b) `merge_similar_entities()`'s "most frequent mention wins" canonicalization and the 0.75 threshold are both reasonable choices informed by inspecting the actual matches at this scale, not calibrated against a larger gold set — thresholds below 0.75 (down to 0.65) showed further gains in the sweep but weren't adopted since those additional matches haven't been individually verified; a larger eval set is the natural next step to resolve this with more confidence.
5. **No public-benchmark evaluation**: the phase spec calls out DocRED and the Open Research Knowledge Graph (ORKG) as candidate benchmarks. Both use different annotation schemas (DocRED: Wikipedia document-level RE with a fixed 96-relation ontology; ORKG: scholarly-paper contribution graphs) that would need a real schema-mapping or transfer-evaluation design, which is out of scope for this pass — recorded as a residual gap in `docs/gaps_toward_academic_deliverable.md`.
6. **No formal KG-embedding baseline**: Wang et al. (2017)'s survey is cited in the original proposal as background for graph learning; this phase evaluates extraction quality and graph structure directly rather than training/comparing embedding models (e.g. TransE) as a downstream task — also recorded as a residual gap.
7. **No end-to-end human evaluation**: the dashboard's search/exploration UX is not evaluated with real users (the phase spec's "usability tests with project teams" and "task-based evaluation... A/B comparison" are unimplemented, same category of deferred work as Phase 1's human evaluation — see `docs/gaps_toward_academic_deliverable.md` §7).
