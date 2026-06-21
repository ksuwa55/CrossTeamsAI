# Phase 1: Pipeline Flow, Real Scenario & Evaluation Results

## Two Flows: Eval vs Real Use

The core summarization logic is identical in both flows. Only the input source differs.

### Eval Flow (QMSum benchmark)

```
QMSum JSON file (data/QMSum/data/ALL/test/*.json)
  │
  ▼  qmsum_loader.py → iter_qmsum()
  │   - Loads transcript + query + reference summary
  │   - Builds input_text (query + length hint + transcript)
  │
  ▼  app/cli_runner.py → main()
  │   1. extract_transcript_from_input_text()  — strips query prefix, gets raw transcript
  │   2. keyword_prefilter()                   — narrows transcript to query-relevant lines
  │   3. split_chunks()                        — splits long transcript into overlapping chunks
  │   4. build_query_prompt() × N chunks       — MAP: prompt per chunk → LLM call → partial summary
  │   5. build_reduce_prompt()                 — REDUCE: merge partials → LLM call → final summary
  │
  ▼  summarizer.py → MeetingSummarizer.run_summarizer()
  │   - Sends prompt to GPT-3.5-turbo via OpenAI API
  │   - Caches result to disk (cache_qmsum/)
  │   - Returns summary string
  │
  ▼  eval/evaluate_qmsum.py
      - Compares prediction vs reference summary
      - Outputs ROUGE-1/2/L and BERTScore F1
```

### Real Scenario (Gradio UI)

```
User pastes real meeting transcript into browser
  │
  ▼  app/ui_runner.py → ui_summarize()
  │   - Same split_chunks → build_query_prompt → run_summarizer → build_reduce_prompt logic
  │   - User can select task type: general / decisions / blockers / action_items / risks / followups
  │   - User can optionally write a custom query
  │
  ▼  summarizer.py → MeetingSummarizer.run_summarizer()
      - Calls GPT-3.5-turbo, caches to cache_ui/
      - Returns summary shown directly in the browser
```

No reference summary exists in the real scenario — the output goes straight to the user.

## Key Files and Their Roles

| File | Role |
|---|---|
| `01_summarization/qmsum_loader.py` → `iter_qmsum()` | Loads QMSum data, builds input_text with query + transcript |
| `app/cli_runner.py` → `build_query_prompt()` | Builds the actual MAP prompt sent to the model |
| `app/cli_runner.py` → `build_reduce_prompt()` | Builds the REDUCE prompt to merge partial summaries |
| `app/cli_runner.py` → `extract_transcript_from_input_text()` | Strips query prefix from input_text to get raw transcript |
| `01_summarization/summarizer.py` → `run_summarizer()` | Makes OpenAI API call and **returns the summary** |
| `app/ui_runner.py` → `ui_summarize()` | UI callback — same pipeline for real user input |
| `eval/evaluate_qmsum.py` | Computes ROUGE + BERTScore against reference |

## Length Fix (2026-04)

**Problem:** Summaries were too long (~134 words avg vs ~71 word references), hurting ROUGE scores.

**Changes made:**
- `01_summarization/qmsum_loader.py`: Added `"Write 2–4 sentences, around 60–80 words."` to `input_text`
- `01_summarization/summarizer.py`: Reduced `max_tokens` from 220 → 150

The `max_tokens` change in `run_summarizer()` is the main driver of length reduction. The prompt hint in `qmsum_loader.py` has limited effect because `cli_runner.py` strips the query prefix before building the actual prompt — the length constraint in the real prompt comes from `SYSTEM_PROMPT` and `max_sentences`.

## Evaluation Results: Before vs After Length Fix

Measured on the 10 examples selected for human evaluation (QMSum test set).

| Metric | Before fix | After fix | Change |
|---|---|---|---|
| ROUGE-1 | 0.2891 | 0.2984 | +0.009 |
| ROUGE-2 | 0.0591 | 0.0621 | +0.003 |
| ROUGE-L | 0.1593 | 0.1714 | +0.012 |
| BERTScore F1 | 0.5900 | 0.6001 | +0.010 |
| Avg pred words | 133.9 | 107.0 | -27 words |
| Length ratio (pred/ref) | 1.88x | 1.50x | closer to reference |

All metrics improved after the fix. Summaries are still ~50% longer than references (71 words avg), so further length tuning may be possible.

## Human Evaluation

10 examples selected from QMSum test set, covering:
- Mixed meeting types (government, academic, product design)
- Mix of general and specific queries
- Spread of ROUGE quality levels (low to high)

The human evaluation form (`eval/human_eval/google_form_blueprint_v2.md`) uses the post-fix summaries and rates each on 4 dimensions: Faithfulness, Relevance, Conciseness, Completeness (1–5 scale). Each evaluator rates 10 examples in ~25 minutes.
