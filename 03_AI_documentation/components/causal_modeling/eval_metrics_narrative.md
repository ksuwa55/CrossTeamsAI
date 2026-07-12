Pooled (micro-averaged), `gpt-3.5-turbo` extraction / `gpt-4o-mini` judge, over
the 10-meeting / 19-pair evaluation set (table above, read live from
`eval/results/causal_extraction/causal_extraction_raw.json`). Full per-meeting
breakdown: `docs/phase2/pipeline-flow-and-results.md`.

**False-negative inspection** (why gold pairs were missed): all 8 pooled
false negatives were classified `partial_match` — the extractor never fully
missed a relationship, but consistently traced the causal chain to a
different link than the gold label chose (e.g. one hop further upstream, or
capturing only the cause or only the effect side of the gold pair). See
`eval/results/causal_extraction/false_negative_inspection.md` for the full
per-case detail.

**Interpretation, not just numbers:**
- Low precision is largely a byproduct of one-to-one matching strictness
  combined with the extractor emitting several finer-grained pairs about the
  same underlying blocker (50 predicted vs. 19 gold pairs, pooled).
  Canonicalization (`canonicalize_node`) absorbs much of this at the graph
  layer, so the raw pair-level precision understates the graph's usefulness
  — but this has not been independently verified with a graph-level metric.
- Two meetings (`meeting_03_missing_resource`, `meeting_05_decision_stagnation`)
  scored 0% precision/recall. Flagged as a known limitation rather than
  addressed here — prompt or window-size tuning was intentionally left out
  of scope for this iteration.

**Explainability:** the dashboard's `_format_path_evidence()` traces every
edge in a shown causal path back to the source transcript quote, so a
manager-facing explanation is generated alongside every simulated
intervention — this is a documented strength relative to the source
literature's "why the prediction was made" requirement.

**No end-to-end evaluation of the DAG or intervention simulator themselves**
— only the upstream extraction step is scored against labels. No expert
(Agile coach / PM) review of the graph's face validity, and no refutation
testing (`dowhy`'s `refute_estimate`) of the SCM's do-calculus estimates.
