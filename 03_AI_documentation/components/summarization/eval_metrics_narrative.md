Table above is read live from `eval/results/metrics_eval_original.json`,
`eval/results/metrics_eval_modified.json`, `eval/results/model_comparison/`,
and `eval/results/ablations/` (produced by `eval/evaluate_qmsum.py`: ROUGE-1/2/L
+ BERTScore F1 with bootstrap 95% CIs). Rows are not all on the same sample
size (n=10 for the two LLM-pipeline rows vs. n=22–23 for model swaps vs.
n=281 for extractive baselines run over the fuller set) — treat the
comparison as directional, not a controlled A/B.

**Interpretation:**
- The "modified" prompt improves ROUGE-1/2/L and BERTScore F1 over "original"
  while also reducing the prediction/reference length ratio (over-generation)
  — the length-control changes (`--max_sentences`, `preserve_ngrams`) appear
  to help lexical overlap rather than trade it off.
- Extractive baselines (`eval/baselines.py`: lead-n, TF-IDF, random) are
  wired into the same evaluation harness specifically so a claim like "the
  LLM pipeline is worth its cost over simple extraction" can be checked
  against a number, not assumed — see
  `docs/gaps_toward_academic_deliverable.md §3`, which notes this is
  something Phase 2 (causal modeling) still lacks.
- No human evaluation of faithfulness/coherence has been run
  (`eval/human_eval/google_form_blueprint.md` is designed but not
  administered) — BERTScore is a semantic-similarity proxy, not a
  faithfulness measure.
