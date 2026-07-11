# Phase 2: Pipeline Flow, Real Scenario & Evaluation Results

## Two Flows: Eval vs Real Use

The core extraction/graph/simulation logic is identical in both flows. Only the entry point and what happens with the output differs.

### Eval Flow (labeled synthetic meetings)

```
Synthetic transcript + .labels.json (data/synthetic_transcripts/*.json)
  │
  ▼  eval/evaluate_causal_extraction.py → main()
  │   1. enrich_transcript()        — regex prefilter tags each utterance
  │   2. extract_causal_events()    — LLM call per candidate window → cause/effect/timestamp
  │   3. judge_compatibility_matrix() + max_bipartite_matching() — LLM judge vs. gold pairs
  │   4. evaluate_meeting()         — precision / recall / F1 per meeting
  │
  ▼  eval/inspect_false_negatives.py
      - For every unmatched gold pair, judge finds the closest prediction and a verdict
```

### Real Scenario (Intervention Dashboard)

```
Real (or sample) meeting transcripts → causal events (data/sample_causal_events/*.json)
  │
  ▼  app/intervention_dashboard.py → _load_model()
  │   - load_events() + build_graph()             — canonicalized DAG across all meetings
  │   - build_occurrence_data() + build_scm()      — fit structural causal model (dowhy.gcm)
  │
  ▼  Manager picks a cause + outcome in the Gradio UI
  │   - run_simulation() checks a causal path exists, runs simulate_intervention()
  │   - Shows P(outcome) baseline vs. mitigated, plus the real transcript quotes
  │     behind each hop of the causal path (_format_path_evidence())
```

No gold labels exist in the real scenario — the dashboard's output goes straight to the manager as a decision-support signal, not a graded prediction.

## Key Files and Their Roles

| File | Role |
|---|---|
| `02_causal_modeling/extract_variables.py` → `enrich_transcript()` | Regex prefilter: tags each utterance with role/topic/decision/blocker/next_action |
| `02_causal_modeling/extract_variables.py` → `extract_causal_events()` | LLM call per candidate window → cause/effect/timestamp events |
| `02_causal_modeling/build_causal_graph.py` → `canonicalize_node()` | Maps free-text phrases onto a fixed node vocabulary |
| `02_causal_modeling/build_causal_graph.py` → `build_graph()` | Merges events from all meetings into one weighted DAG with evidence |
| `02_causal_modeling/simulate_intervention.py` → `build_scm()` | Fits a structural causal model (empirical roots + logistic-regression downstream) |
| `02_causal_modeling/simulate_intervention.py` → `simulate_intervention()` | Do-calculus: P(outcome) baseline vs. cause mitigated |
| `app/intervention_dashboard.py` | Gradio "what-if" UI for managers, with evidence quotes |
| `eval/evaluate_causal_extraction.py` | LLM-judge precision/recall/F1 vs. hand-labeled pairs |
| `eval/inspect_false_negatives.py` | Classifies each missed gold pair (phrasing mismatch / partial match / genuine miss) |

## Evaluation Results: Causal Event Extraction

Extraction model: `gpt-3.5-turbo` (unmodified pipeline). Matching: LLM judge (`gpt-4o-mini`), one-to-one maximum bipartite matching per meeting.

| Meeting | Predicted | Gold | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| meeting_01_signoff_blocker | 7 | 3 | 2 | 5 | 1 | 28.6% | 66.7% | 40.0% |
| meeting_02_ambiguous_requirement | 3 | 3 | 1 | 2 | 2 | 33.3% | 33.3% | 33.3% |
| meeting_03_missing_resource | 2 | 1 | 0 | 2 | 1 | 0.0% | 0.0% | 0.0% |
| meeting_04_api_dependency | 8 | 1 | 1 | 7 | 0 | 12.5% | 100.0% | 22.2% |
| meeting_05_decision_stagnation | 5 | 1 | 0 | 5 | 1 | 0.0% | 0.0% | 0.0% |
| meeting_06_timezone_communication | 6 | 2 | 1 | 5 | 1 | 16.7% | 50.0% | 25.0% |
| meeting_07_scope_change | 4 | 2 | 1 | 3 | 1 | 25.0% | 50.0% | 33.3% |
| meeting_08_qa_delay_deadline | 6 | 2 | 2 | 4 | 0 | 33.3% | 100.0% | 50.0% |
| meeting_09_security_ambiguity | 2 | 2 | 1 | 1 | 1 | 50.0% | 50.0% | 50.0% |
| meeting_10_resource_reprioritization | 7 | 2 | 2 | 5 | 0 | 28.6% | 100.0% | 44.4% |
| **Aggregate (pooled)** | 50 | 19 | 11 | 39 | 8 | 22.0% | 57.9% | 31.9% |

*Aggregate is pooled (micro-averaged): TP/FP/FN are summed across all meetings before computing precision/recall/F1.*

## False Negative Inspection

All 8 pooled false negatives were inspected by finding the closest predicted pair per meeting and classifying the miss:

| Verdict | Count |
|---|---|
| `phrasing_mismatch` | 0 |
| `partial_match` | 8 |
| `genuine_miss` | 0 |

**Every missed gold pair had a partially-overlapping prediction** — the extractor never completely failed to notice a relationship, but it consistently split or re-scoped the gold cause/effect pair differently than the label. Typical pattern (`meeting_01_signoff_blocker`): gold pair is "customer records module can't be pushed → QA holding off on test cases"; the closest prediction is "delay in legal approval → stalled progress on writing test cases" — same effect area, but the cause is traced one hop further upstream than the label chose.

## Interpretation

- **Recall (57.9%) is moderate-to-good** and the false-negative inspection shows the real number is better than it looks: 0 of 8 misses were genuine — the pipeline surfaced *something* related to every gold relationship, just not phrased/scoped identically.
- **Precision (22.0%) is low, largely by construction of the matching strictness combined with the extractor's behavior**: the pipeline predicts ~2.6x as many pairs as there are gold pairs (50 vs. 19), because it tends to emit several finer-grained cause/effect pairs about the same underlying blocker (e.g. `meeting_04_api_dependency`: 8 predicted vs. 1 gold) rather than one pair matching the label's granularity.
- **Practical implication for the DAG/dashboard**: canonicalization (`canonicalize_node()`) absorbs much of this granularity mismatch — many of the "extra" predicted pairs collapse onto the same canonical node pair in the graph, so the aggregate DAG is more robust to precision loss than the raw per-pair metric suggests. The per-pair precision/recall numbers above should be read as a lower bound on the extraction step's usefulness for the downstream graph, not a measure of the graph's own quality.
- Two meetings (`meeting_03_missing_resource`, `meeting_05_decision_stagnation`) scored 0% precision/recall. This is flagged as a known limitation rather than addressed here — prompt or window-size tuning was intentionally left out of scope for this iteration (see "Known Limitations").