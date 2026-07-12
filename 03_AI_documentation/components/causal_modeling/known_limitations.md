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
