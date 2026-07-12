See `docs/phase1/evaluation-strategy.md` ("Known Limitations") for the full
list from the evaluation design; summarized here:

- **QMSum-only evaluation** — no cross-dataset generalization testing (e.g.
  AMI/ICSI corpus directly, or real non-benchmark meetings).
- **Reference bias** — ROUGE rewards lexical overlap with a single human
  reference and may penalize valid but differently-worded summaries.
- **No human evaluation** — `eval/human_eval/google_form_blueprint.md` is
  designed but was never administered; BERTScore is a semantic-similarity
  proxy, not a substitute for human judgment on faithfulness.
- **Subset sampling by default** — headline numbers in Section 3 are on a
  ~10-meeting sample for cost control, not the full 32-meeting test split
  (`--sample_ratio 1.0 --max_meetings 999` for the full run).
- **No AI-generated-content disclosure in the UI** (`app/ui_runner.py`) — end
  users are not told the summary is LLM-generated and may be inaccurate.
- **English-only, benchmark-domain only** — no testing on other languages or
  meeting domains outside QMSum's academic/product/committee distribution.
