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
