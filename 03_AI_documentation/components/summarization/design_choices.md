| Decision | Value | Rationale |
|---|---|---|
| Base model | `gpt-3.5-turbo` (`--model`, configurable) | Cost/latency baseline; alternatives benchmarked empirically (see below) rather than assumed |
| Decoding temperature | `0.0` (CLI batch default, `app/cli_runner.py`) / `0.2` (`MeetingSummarizer.run_summarizer` default) | Low temperature stabilizes n-gram overlap, which ROUGE-2/L directly reward |
| Prompting strategy | Map-reduce: chunk (`--chunk_chars 12000`, `--overlap_chars 1000`) → per-chunk "map" summary → single "reduce" merge | QMSum transcripts routinely exceed a single-call context budget |
| Context narrowing | Keyword-based prefilter (`keyword_prefilter()`), on by default (`--prefilter on`) | Cheap retrieval step to keep only query-relevant lines before chunking, reducing cost and off-topic drift |
| Caching | Disk cache keyed by `md5(model + system_prompt + prompt)` (`get_cache_path()`) | Avoids redundant paid API calls across repeated runs/evals; also the traceability mechanism referenced under "Known Limitations & Risks" |
| Optional polish pass | `revise_summary()`, off by default (`--revise off`) | Extra LLM call trades cost for coherence |

**Alternatives measured, not just considered** (`eval/results/model_comparison/`,
`eval/results/ablations/` — the numbers in the Evaluation Metrics table above
are pulled directly from these files):
- Model swap: `gpt-4o`, `gpt-4-turbo` vs. the `gpt-3.5-turbo` default.
- Chunk size: `8k`/`16k` chars vs. the `12k` default (`eval/results/ablations/results_ablation_chunk8k_*.json`, `..._chunk16k_*.json`).
- Temperature: `0.2`/`0.5` vs. the `0.0` batch default.
- Prefilter on vs. off.
- Revise pass on vs. off.
- Extractive, non-LLM baselines: lead-n, TF-IDF, random (`eval/baselines.py`) —
  the reference point for whether the LLM pipeline is worth its cost at all.
