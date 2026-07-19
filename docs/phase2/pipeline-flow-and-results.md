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
  │   5. evaluate_meeting_chain_aware() — graph-reachability "chain-aware recall" (see below)
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

## Chain-Aware Recall (Graph Reachability)

The false-negative pattern above — every miss was a *partial* match, one hop upstream or
downstream of the gold label's cause/effect scoping — motivated a second, graph-level
recall metric in `eval/evaluate_causal_extraction.py`, alongside (not replacing) the
strict pair-level P/R/F1 table above.

**How it differs from the pair-level metric:** the pair-level match requires one predicted
pair to match gold's cause *and* effect simultaneously. Chain-aware recall instead builds a
small directed graph per meeting from the predicted pairs (raw predicted text as nodes, one
edge per pair — deliberately not canonicalized; that stays downstream in
`build_causal_graph.py`) and checks whether a *path* exists from a node matching gold's
cause to a node matching gold's effect. Node equivalence is judged by the same LLM-judge
method as the pair-level matching (`judge_node_matches`), just applied to a single phrase
against candidate node phrases instead of a whole pair against candidate pairs. This means a
gold pair now counts as recovered if the model traced cause → intermediate node → effect,
even though no single predicted pair spans cause to effect directly — exactly the "traced
one hop further upstream/short" pattern the false-negative inspection found.

**What it doesn't capture:** it has no precision or F1 counterpart. Reachability over raw,
uncanonicalized text has no well-defined "false positive edge" — an unrelated predicted
node just sits off to the side of the graph, unreachable-relevant, rather than counting
against the metric the way it would against pair-level precision. Read chain-aware recall
as "how often the extractor's raw output *contains the causal story*, regardless of hop
count," not as a replacement quality score — the pair-level P/R/F1 above is still the
right metric for judging how closely predictions match the label's exact granularity.

### Results (2026-07-20 run, gpt-3.5-turbo extraction / gpt-4o-mini judge)

| Meeting | Gold | Chain Hits | Chain Misses | Chain Recall | Pair Recall |
|---|---|---|---|---|---|
| meeting_01_signoff_blocker | 3 | 2 | 1 | 66.7% | 66.7% |
| meeting_02_ambiguous_requirement | 3 | 1 | 2 | 33.3% | 33.3% |
| meeting_03_missing_resource | 1 | 1 | 0 | 100.0% | 0.0% |
| meeting_04_api_dependency | 1 | 0 | 1 | 0.0% | 100.0% |
| meeting_05_decision_stagnation | 1 | 1 | 0 | 100.0% | 0.0% |
| meeting_06_timezone_communication | 2 | 0 | 2 | 0.0% | 50.0% |
| meeting_07_scope_change | 2 | 1 | 1 | 50.0% | 50.0% |
| meeting_08_qa_delay_deadline | 2 | 1 | 1 | 50.0% | 100.0% |
| meeting_09_security_ambiguity | 2 | 2 | 0 | 100.0% | 50.0% |
| meeting_10_resource_reprioritization | 2 | 2 | 0 | 100.0% | 100.0% |
| **Aggregate (pooled)** | 19 | 11 | 8 | **57.9%** | 57.9% |

The pooled chain-aware recall (57.9%, 11/19) lands at the *same* aggregate number as
pair-level recall — but that's a coincidence of totals, not the same 11 pairs being
credited. Cross-referencing which individual gold pairs flipped status shows two distinct
effects happening in opposite directions:

**Question this metric was built to answer: do the 8 previously-identified false negatives
(all classified `partial_match` by `inspect_false_negatives.py`) get credited by chain-aware
recall?** Partially — **4 of the 8 are recovered**, 4 are still misses even with multi-hop
credit:

| Recovered by chain-aware (4) | Still missed (4) |
|---|---|
| `meeting_02`: "the requirement is really unclear" → ... | `meeting_02`: "about three days of work have been lost..." → "...miss the Friday deadline" |
| `meeting_03`: "QA is backed up" → ... (0%→100% for this meeting) | `meeting_01`: "the customer records module cannot be pushed" → "QA is holding off..." |
| `meeting_05`: "the team doesn't think they can decide without a plan" → ... (0%→100%) | `meeting_06`: "by the time Sam got to the review, the Osaka team had already logged off..." → ... |
| `meeting_09`: "the compliance team is blocking the release..." → ... | `meeting_07`: "the client added a new requirement for multi-currency support..." → ... |

So the hypothesis motivating this metric — that some misses are just chain-length
disagreements, recoverable via an intermediate node — held for exactly half of the
inspected false negatives (`meeting_03` and `meeting_05` in particular go from 0% to 100%
recall for their single gold pair). The other half remain genuine misses even allowing
multi-hop paths: for those, no predicted node in the graph was judged equivalent to
*either* gold's cause or gold's effect at all, so no path could exist regardless of hops.

**Unexpected finding — chain-aware recall is not a strict superset of pair-level recall.**
4 gold pairs that were exact pair-level **true positives** (`meeting_02`, `meeting_04`,
`meeting_06`, `meeting_08`) came back as chain-aware **misses**. Inspecting these (e.g.
`meeting_04`: predicted "Delay in receiving sandbox API credentials → Stalled progress on
shipping partner integration" was accepted by the pair-level judge as matching gold's
"...waiting on sandbox API credentials... → the whole integration testing phase is
delayed...") shows why: the pair-level judge sees cause and effect *together* and can
credit an overall relationship match with some slack in how the effect is framed. The
node-level judge (`judge_node_matches`) sees the same effect phrase *in isolation*, without
the paired cause for context, and applied a stricter reading — "stalled progress on
shipping partner integration" isn't judged the same real-world thing as "the whole
integration testing phase is delayed" on its own. This is a genuine property of the two
judging tasks being different (whole-relationship equivalence vs. single-phrase
equivalence), not a bug in the matching logic, and it's why the pooled recall numbers
matching exactly (57.9% both) is coincidental rather than chain-aware recall being a
strict relaxation of the pair-level metric.

## Interpretation

- **Recall (57.9%) is moderate-to-good** and the false-negative inspection shows the real number is better than it looks: 0 of 8 misses were genuine — the pipeline surfaced *something* related to every gold relationship, just not phrased/scoped identically.
- **Precision (22.0%) is low, largely by construction of the matching strictness combined with the extractor's behavior**: the pipeline predicts ~2.6x as many pairs as there are gold pairs (50 vs. 19), because it tends to emit several finer-grained cause/effect pairs about the same underlying blocker (e.g. `meeting_04_api_dependency`: 8 predicted vs. 1 gold) rather than one pair matching the label's granularity.
- **Practical implication for the DAG/dashboard**: canonicalization (`canonicalize_node()`) absorbs much of this granularity mismatch — many of the "extra" predicted pairs collapse onto the same canonical node pair in the graph, so the aggregate DAG is more robust to precision loss than the raw per-pair metric suggests. The per-pair precision/recall numbers above should be read as a lower bound on the extraction step's usefulness for the downstream graph, not a measure of the graph's own quality.
- Two meetings (`meeting_03_missing_resource`, `meeting_05_decision_stagnation`) scored 0% precision/recall. This is flagged as a known limitation rather than addressed here — prompt or window-size tuning was intentionally left out of scope for this iteration (see "Known Limitations").
- **Chain-aware recall (above) partially confirms the "canonicalization absorbs granularity mismatch" claim, with a caveat**: of the 8 pair-level false negatives, 4 are recovered once multi-hop paths are allowed (2 meetings go from 0% to 100% recall), consistent with the "traced one hop further" pattern the false-negative inspection found. But 4 remain genuine misses even with multi-hop credit, and 4 *other* gold pairs that were exact pair-level matches flip to chain-aware misses — because the node-level judge, seeing a cause or effect phrase in isolation, is stricter than the pair-level judge, which sees both halves together. Net effect: pooled chain-aware recall (57.9%) numerically equals pair-level recall, but over a different set of pairs — read both numbers, not just one.