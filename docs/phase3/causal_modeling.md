# Model Card: Phase 2 — Causal Modeling of Project Bottlenecks

This card documents `02_causal_modeling/`, `app/intervention_dashboard.py`, `eval/evaluate_causal_extraction.py` following the four documentation-requirement categories identified in Königstorfer & Thalmann (2022), *AI Documentation: A Path to Accountability*, Journal of Responsible Technology, 11:100043 — supplemented by the Model Cards framework (Mitchell et al., 2019).

## Intended Use

Extracts cause→effect relationships from meeting transcripts, merges them
across meetings into a shared causal graph, fits a structural causal model
over that graph, and lets a manager simulate "what if we mitigated X" through
a Gradio dashboard.

**Intended as a decision-support signal, not a validated causal claim.** The
dashboard surfaces a hypothesis about root causes, traceable back to the
original transcript quotes behind each hop — it does not certify that the
relationship is causal in a statistically rigorous sense. See "Known
Limitations & Risks" before using output to justify a resourcing or process
decision.

## 1. Model Design Choices

| Decision | Value | Rationale |
|---|---|---|
| Extraction model | `gpt-3.5-turbo` | Consistent with Phase 1; reuses `MeetingSummarizer`'s call path and disk cache |
| Extraction temperature | 0.0 | Reproducibility of extracted cause/effect pairs |
| Candidate window selection | Regex prefilter (`blocker`/`decision`/`next_action` flags) + ±N-utterance window (`select_candidate_windows`) | Keeps LLM calls limited to plausibly relevant context, controlling cost |
| Node canonicalization | Fixed 8-category vocabulary (`blocked_dependency`, `decision_stagnation`, `ambiguous_requirement`, `scope_change`, `missing_resource`, `qa_delay`, `integration_delay`, `missed_deadline`) + `other:<phrase>` fallback | Prevents the graph fragmenting into one node per unique LLM phrasing |
| SCM mechanism | Root nodes: empirical (observed-frequency) distribution. Downstream nodes: logistic regression over parents (`dowhy.gcm.ClassifierFCM`) | Simplest mechanism that still lets a hard intervention propagate through do-calculus |
| Evaluation judge model | `gpt-4o-mini`, deliberately different from the `gpt-3.5-turbo` extractor | Reduces self-consistency bias between extraction and evaluation |
| Matching method | LLM judge + one-to-one maximum bipartite matching (`networkx`), not embedding similarity | Embedding cosine similarity could not reliably separate "same causal claim, different wording" from "topically related but distinct claim," and cannot check cause/effect directionality |

**Alternatives considered but not adopted:**
- Embedding-based similarity matching for evaluation — rejected for the
  directionality/precision reasons above (see
  `docs/phase2/evaluation-strategy.md`).
- Post-hoc deduplication of near-identical predicted pairs — considered, but
  not implemented; canonicalization at the graph layer already absorbs most
  of this granularity mismatch (see "Evaluation Metrics" below), and adding a
  separate deduplication step risked conflicting with that documented design
  rationale.

## 2. Data Characteristics

- **Evaluation dataset:** 10 synthetic meeting transcripts
  (`data/synthetic_transcripts/`), each authored to instantiate one labeled
  failure mode: signoff blocker, ambiguous requirement, missing resource, API
  dependency, decision stagnation, timezone/cross-border communication, scope
  change, QA delay/deadline, security ambiguity, resource reprioritization —
  chosen to cover the 8-category canonical node vocabulary above, plus one
  cross-border-specific scenario reflecting the original research proposal's
  focus on cross-border Agile teams.
- **Gold labels:** 19 hand-written cause/effect pairs across the 10 meetings
  (1–3 per meeting), in `*.labels.json`.
- **Known bias:** all gold labels were written by a single annotator (a
  non-native English speaker, self-reported). No second annotator or
  inter-annotator agreement score (e.g. Cohen's kappa) exists. No written
  labeling rubric was used to standardize what counts as "the cause" vs. an
  intermediate effect in a causal chain — see the granularity-mismatch
  finding under "Evaluation Metrics."
- **Dashboard seed data:** `data/sample_causal_events/` — pre-extracted
  events used only to populate the Gradio dashboard at startup; distinct from
  the labeled evaluation set and not itself evaluated against gold labels.
- **Real transcripts:** none used. All development and evaluation data is
  synthetic.

## 3. Evaluation Metrics

| Run | Precision | Recall | F1 |
|---|---|---|---|
| Pooled (10 meetings / 19 gold pairs) | 22.0% | 57.9% | 31.9% |

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

## 4. System Modification Log

No automated change-tracking platform is used; this section is maintained
manually as of this iteration.

| Change | Notes |
|---|---|
| Regex-only prefilter (`extract_variables.py`) | Initial state on `main` |
| LLM-based extraction, DAG builder, SCM/DoWhy simulator, Gradio dashboard | Developed on a feature branch |
| Feature branch merged into `main` | Brings `02_causal_modeling/build_causal_graph.py`, `simulate_intervention.py`, `app/intervention_dashboard.py` into the tracked repo |
| Evaluation harness added | `eval/evaluate_causal_extraction.py`, `eval/inspect_false_negatives.py`, plus the 10 synthetic transcripts + gold labels |
| `docs/phase2/` four-document set added | architecture, data-design, evaluation-strategy, pipeline-flow-and-results |
| README updated | Phase 2 section added, mirroring Phase 1's documentation style |
| `gaps_toward_academic_deliverable.md` added | Cross-phase record of what remains for academic-quality rigor |
| Migrated to `03_AI_documentation/generate_model_card.py` | This card is now rendered from `components/causal_modeling/` instead of hand-maintained |

## Known Limitations & Risks

See `docs/phase2/evaluation-strategy.md` ("Known Limitations") and
`gaps_toward_academic_deliverable.md` for the full list. Summary of the
highest-relevance items for someone relying on this system's output:

- **Small, synthetic evaluation set** (10 meetings, 19 pairs) — results are
  directional, not statistically precise, and do not generalize to real
  meeting transcripts, which have not been tested.
- **No baseline comparison** — there is currently no evidence that the LLM
  extractor outperforms a simpler method (e.g. keyword co-occurrence) on this
  task.
- **Single-annotator gold labels**, no inter-annotator agreement measured.
- **No refutation/robustness testing** of the causal estimates.
- **No comparison against Bayesian networks**, despite the original research
  proposal specifying "Bayesian networks + DoWhy/CausalML" rather than DoWhy
  alone.
- **No expert (Agile coach / PM) review** of whether the graph's causal
  claims are plausible to a domain practitioner.
- **Not integrated with Phase 1** — there is no "causal querying" of past
  discussions through the summarization pipeline, as originally proposed.

---
*Generated 2026-07-12 06:22 UTC by `03_AI_documentation/generate_model_card.py` from commit `08ef053` (uncommitted changes present) on `phase3-impl`. Do not hand-edit — edit `03_AI_documentation/components/causal_modeling/` and re-run the generator instead.*
