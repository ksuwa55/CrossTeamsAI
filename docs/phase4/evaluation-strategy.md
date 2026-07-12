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

`entity_linking_metrics()` compares the set of predicted entity node keys (`node_key(entity_type, text)`) against the set of gold entity node keys for a meeting:

- **Precision**: of the entities the model surfaced, how many correspond to a real gold entity
- **Recall**: of the gold entities, how many the model surfaced

This is the "Top-N precision for entity linking" metric from the phase spec, framed as set precision/recall rather than a ranked Top-N since the extractor doesn't produce a ranked entity list.

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

## Experiment Tracking

- Predictions: `output/kg_triples/*.kg_triples.json`
- Reports: `eval/results/kg_extraction/kg_extraction_report.md`, `kg_extraction_raw.json`
- Extraction cache: `cache_kg/`; judge cache: `cache_kg_eval/`; embedding cache (semantic search, not part of this eval): `cache_kg_embeddings/`

## Known Limitations

1. **Small, synthetic dataset**: 10 meetings, 40 gold triples — same scale caveat as Phase 2, directional not precise.
2. **Judge is itself an LLM**: not a ground-truth oracle, though model-distinct from the extractor.
3. **Entity resolution is heuristic, not learned**: `resolve_person_alias()` only handles substring-level aliasing against known speakers; it will not resolve pronouns, nicknames unrelated to the speaker label, or cross-meeting aliasing of the same person under different display names.
4. **No public-benchmark evaluation**: the phase spec calls out DocRED and the Open Research Knowledge Graph (ORKG) as candidate benchmarks. Both use different annotation schemas (DocRED: Wikipedia document-level RE with a fixed 96-relation ontology; ORKG: scholarly-paper contribution graphs) that would need a real schema-mapping or transfer-evaluation design, which is out of scope for this pass — recorded as a residual gap in `docs/gaps_toward_academic_deliverable.md`.
5. **No formal KG-embedding baseline**: Wang et al. (2017)'s survey is cited in the original proposal as background for graph learning; this phase evaluates extraction quality and graph structure directly rather than training/comparing embedding models (e.g. TransE) as a downstream task — also recorded as a residual gap.
6. **No end-to-end human evaluation**: the dashboard's search/exploration UX is not evaluated with real users (the phase spec's "usability tests with project teams" and "task-based evaluation... A/B comparison" are unimplemented, same category of deferred work as Phase 1's human evaluation — see `docs/gaps_toward_academic_deliverable.md` §7).
