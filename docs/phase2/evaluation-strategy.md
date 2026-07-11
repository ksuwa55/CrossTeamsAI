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
