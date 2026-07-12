No automated change-tracking platform is used; this section is maintained
manually as of this iteration.

| Change | Notes |
|---|---|
| Core `MeetingSummarizer` + single-transcript path (`app/main.py`) | Initial implementation |
| QMSum loader + batch CLI runner (`app/cli_runner.py`) with map-reduce chunking | Enables benchmark-scale evaluation beyond a single transcript |
| Prompt tuning pass (prefilter, `preserve_ngrams`, length control, revise pass) | Produced the "modified" vs. "original" comparison in Section 3 |
| Extractive baselines added (`eval/baselines.py`: lead-n, TF-IDF, random) | Gives a non-LLM reference point for the ROUGE/BERTScore numbers |
| Model-comparison and ablation sweeps (`eval/results/model_comparison/`, `eval/results/ablations/`) | Empirically tests alternatives (gpt-4o, gpt-4-turbo, chunk size, temperature, prefilter, revise) instead of deciding by default |
| Gradio UI (`app/ui_runner.py`) | Interactive path alongside the batch CLI |
| `docs/phase1/` four-document set added | architecture, data-design, evaluation-strategy, pipeline-flow-and-results |
| This card (`03_AI_documentation/components/summarization/`) added | Brings Phase 1 into the same accountability-documentation format as Phase 2 |
