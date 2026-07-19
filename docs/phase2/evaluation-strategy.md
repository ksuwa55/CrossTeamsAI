# Phase 2: Causal Modeling of Project Bottlenecks — Evaluation Strategy

## Objective

Measure how well the causal-event extraction pipeline (`02_causal_modeling/extract_variables.py`) recovers the true cause→effect relationships in a meeting transcript, and characterize *why* it misses the ones it misses.

## Benchmark Dataset

**Synthetic labeled meetings** (`data/synthetic_transcripts/`)

- 10 short synthetic transcripts, each built around one labeled project-bottleneck scenario (signoff blocker, ambiguous requirement, missing resource, API dependency, decision stagnation, timezone communication, scope change, QA delay/deadline, security ambiguity, resource reprioritization)
- Each meeting has a hand-written `.labels.json` with 1–3 gold cause/effect pairs
- Chosen over QMSum (Phase 1's benchmark) because QMSum has no causal-relationship annotations — this task needed purpose-built labels

### Dataset Statistics

| Category | Count |
|---|---|
| Meetings | 10 |
| Gold cause/effect pairs | 19 (1–3 per meeting) |

## Metrics

### Primary Metrics

| Metric | What it measures |
|---|---|
| Precision | Of predicted cause/effect pairs, how many match a gold pair |
| Recall | Of gold cause/effect pairs, how many were predicted |
| F1 | Harmonic mean of precision and recall |

Reported both per-meeting and pooled (micro-averaged: TP/FP/FN summed across all meetings before computing P/R/F1).

### Matching Method: LLM Judge, not Embedding Similarity

Gold and predicted phrases are short, paraphrased descriptions of a cause or effect (e.g. "legal sign-off is pending" vs. "the data-sharing agreement hasn't been approved"). Embedding cosine similarity struggles here because it mostly picks up topical overlap ("legal", "sign-off", "approval" all share a topic) and can't reliably distinguish "same causal claim, different wording" from "related but distinct claim in the same topic area" — exactly the case that determines most meetings' recall, since most have only 1–3 gold pairs.

An LLM judge (`gpt-4o-mini`, distinct from the `gpt-3.5-turbo` extraction model) reads both phrases in context and explicitly checks direction ("A causes B" vs. "B causes A"), which bag-of-embeddings similarity cannot. The dataset is small enough that judge calls stay cheap: one call per predicted pair, judged against all gold pairs for that meeting at once (batched into a single structured-JSON response).

Matching is one-to-one: a predicted pair can match at most one gold pair and vice versa. Given the judge's boolean compatibility matrix per meeting, a maximum-cardinality bipartite matching (`networkx.algorithms.bipartite`) is used rather than greedy first-match, so one meeting's matching choice can't spuriously starve a later pair of its only valid match.

### Secondary Metric: Chain-Aware Recall (Graph Reachability)

The false negative inspection below found that misses aren't wording mismatches — the judge
consistently found a *partially overlapping* predicted pair, one hop upstream or downstream
of the gold label's chosen cause/effect scoping. The strict pair-level match above can't
credit that: it requires one predicted pair to match the gold pair's cause **and** effect
at once, so a model that traces the same underlying chain through an extra intermediate
hop (or collapses two gold hops into one) is scored as both a false positive and a false
negative, even though it recovered the relationship.

Chain-aware recall relaxes this to a graph-reachability check:

1. Build a small directed graph per meeting from the predicted (cause, effect) pairs —
   raw predicted text as nodes, one edge per pair. **Not canonicalized** (that only happens
   downstream in `build_causal_graph.py`) — this stays evaluation-only.
2. For each gold pair, instead of requiring a single predicted pair to match both ends,
   check whether a *path* exists in the predicted graph from a node matching gold's cause
   to a node matching gold's effect. This credits multi-hop chains: gold's cause → some
   predicted node → gold's effect counts as a hit, even though no single predicted pair
   spans cause to effect directly.
3. Node equivalence ("is this the same real-world thing as gold's cause/effect") is judged
   by the same LLM-judge approach as the pair-level matching above — reused unmodified in
   spirit, just applied to a single phrase against a list of candidate node phrases instead
   of a whole pair against a list of candidate pairs.

**What it captures that pair-level P/R/F1 doesn't:** chain-length or chain-scoping
disagreement between the model and the gold label — the model traced the same causal
story, just through a different number of hops.

**What it doesn't capture:** it has no precision or F1 counterpart. Graph reachability
over raw, uncanonicalized text doesn't have a well-defined notion of a "false positive
edge" — a spurious node can sit anywhere in the graph without ever being on a path a gold
pair needs, so it's silently harmless to this metric even though it would count against
pair-level precision. This metric is recall-only and is reported *alongside*, not instead
of, the pair-level P/R/F1 — a meeting can score well on one and poorly on the other, and
both readings are needed to understand the extractor's behavior.

### Supporting Analysis: False Negative Inspection

For every gold pair that had no matching prediction, `eval/inspect_false_negatives.py` asks the judge to find the single closest predicted pair (even though it fell short) and classify the miss:

| Verdict | Meaning |
|---|---|
| `phrasing_mismatch` | Same relationship, just worded differently — a human would call it a match |
| `partial_match` | Captures the cause OR the effect, not both, or a related-but-narrower/broader claim |
| `genuine_miss` | No predicted pair meaningfully overlaps |

This distinguishes real extraction failures from cases where the pipeline found the relationship but the strict one-to-one matching didn't credit it.

## Evaluation Pipeline

### Step 1: Extract + Evaluate

```bash
python eval/evaluate_causal_extraction.py \
  --transcripts-dir data/synthetic_transcripts \
  --extract-model gpt-3.5-turbo \
  --judge-model gpt-4o-mini
```

This runs `enrich_transcript()` + `extract_causal_events()` (unmodified pipeline) per meeting, judges predicted-vs-gold pairs, and writes:
- `output/causal_events/<meeting_id>.causal_events.json` — raw predictions
- `eval/results/causal_extraction/causal_extraction_report.md` — per-meeting + aggregate P/R/F1 table, false positives, false negatives
- `eval/results/causal_extraction/causal_extraction_raw.json` — machine-readable matched/unmatched pairs

### Step 2: Inspect False Negatives

```bash
python eval/inspect_false_negatives.py
```

Reads `causal_extraction_raw.json`, classifies each false negative, and writes `false_negative_inspection.md` / `.json`.

## Tunable Parameters

| Parameter | Default | Impact |
|---|---|---|
| `--extract-model` | `gpt-3.5-turbo` | Extraction quality vs. cost |
| `--judge-model` | `gpt-4o-mini` | Matching judge; needs reliable structured-JSON output |
| `--window-before` / `--window-after` | 2 / 1 | How much surrounding context each candidate window gives the extractor |

## Experiment Tracking

- Predictions: `output/causal_events/*.causal_events.json`
- Reports: `eval/results/causal_extraction/causal_extraction_report.md`, `causal_extraction_raw.json`
- False negative inspection: `eval/results/causal_extraction/false_negative_inspection.md`, `.json`
- Extraction cache: `cache_causal/`; judge cache: `cache_causal_eval/`

## Known Limitations

1. **Small, synthetic dataset**: 10 meetings, 19 gold pairs. Not real meetings, and too small for tight confidence intervals — results should be read as directional, not precise.
2. **Judge is itself an LLM**: matching quality depends on `gpt-4o-mini`'s judgment; it is not a ground truth oracle, though it is model-distinct from the extractor to reduce self-consistency bias.
3. **Low precision by design of the matching strictness**: the extractor tends to split single gold relationships into multiple finer-grained predicted pairs (e.g. one gold pair vs. several predicted pairs about the same blocker), which the one-to-one matcher scores as extra false positives even when each captures a true underlying signal.
4. **No end-to-end human evaluation**: the causal DAG and intervention simulator (`build_causal_graph.py`, `simulate_intervention.py`) are not evaluated directly — only the upstream extraction step is measured against labels.
